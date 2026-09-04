"""Losses for the centre head.

Three things here are taken from published recipes rather than chosen by hand,
after the hand-chosen version was measured to be the weak point:

1. Rotation as a FLIP-INVARIANT sine error plus a direction classifier
   (SECOND, arXiv:1812.05784; the same `add_sin_difference` the Mixed Signals
   devkit's own PointPillars loss uses). Measured motivation: with plain L1 on
   (sin, cos) the predicted vector collapsed to norm 0.608 and median yaw error
   was 41.4 deg, which halved to 22.4 deg when direction was ignored -- i.e. the
   network knew the axis and was hedging between two opposite answers.

2. Loss weights from CenterPoint (OpenPCDet nuScenes config): cls_weight 1.0,
   loc_weight 0.25, and code weights of 0.2 on the rotation components. The
   previous uniform 1.0 came from nowhere and weighted rotation ~20x higher than
   the reference -- a term the model could not solve, so it dominated gradients
   with noise. Regression is normalised per OBJECT (as CenterNet does), not per
   element.

3. IoU-aware classification (RangeDet, arXiv:2103.10039, one of its four named
   components). The heatmap target at a positive becomes the detached BEV IoU of
   the decoded prediction against the ground-truth box instead of a constant 1,
   so the score the ranking uses reflects localisation quality. AP is a ranking
   metric, so this targets it directly.
"""
import torch
import torch.nn.functional as F

# CenterPoint / OpenPCDet nuScenes: LOSS_WEIGHTS cls_weight 1.0, loc_weight 0.25,
# code_weights [...,0.2,0.2] on the sin/cos pair.
# Rotation weighting, arrived at by measurement rather than by copying a config.
#
# CenterPoint-nuScenes (loc 0.25, rot code 0.2 -> rotation 0.05) under-weights
# heading for a metric that is BEV rotated IoU: we measured that a vehicle needs
# |yaw err| <= 22 deg and a bike <= 13 deg to clear IoU 0.7, and we sit at ~25.
#
# But copying CenterPoint-Waymo wholesale (loc 2.0, all code 1.0) COLLAPSED
# training: val mAP fell to exactly 0.000 by epoch 2. The reason is that Waymo's
# config is paired with Adam, which normalises per parameter, whereas we use SGD
# where a loss weight acts directly as a learning-rate multiplier. That change
# raised offset/z/dim by 8x and rotation by 40x at once; `dim` is predicted in
# log space and exponentiated at decode, so once it drifted the boxes became
# enormous and matched nothing.
#
# So: leave the other regression terms at the nuScenes scale that was training
# stably, and raise ONLY rotation, and only 5x (0.05 -> 0.25).
CLS_WEIGHT = 1.0
LOC_WEIGHT = 0.25
CODE_W = {"offset": 1.0, "z": 1.0, "dim": 1.0, "rot": 1.0}
DIR_WEIGHT = 0.5


def focal_loss(pred, gt, quality=None, class_weights=None):
    """Penalty-reduced focal loss (CornerNet/CenterNet), optionally IoU-aware.

    quality=None  -> the usual formulation: positives are pushed toward 1.
    quality given -> Quality Focal Loss (GFL, arXiv:2006.04388) at positives,
                     where the target is the box's IoU rather than a constant 1,
                     so the score the ranking uses reflects localisation quality.
                     The negative term keeps CenterNet's (1-gt)^4 penalty
                     reduction around each centre either way.
    class_weights: optional (C,) tensor -- per-class multiplier on both the
                     positive and negative loss terms. Used to fight the mAP
                     drag from small classes (Bike, Pedestrian) without changing
                     the per-object normaliser.
    """
    p = torch.sigmoid(pred).clamp(1e-3, 1 - 1e-3)
    pos = gt.eq(1).float()
    neg = 1.0 - pos

    if quality is None:
        pos_loss = torch.log(p) * torch.pow(1 - p, 2) * pos
    else:
        q = quality.clamp(0, 1)
        pos_loss = (q * torch.log(p) + (1 - q) * torch.log(1 - p)) \
                   * torch.pow((q - p).abs(), 2) * pos

    neg_loss = torch.log(1 - p) * torch.pow(p, 2) * torch.pow(1 - gt, 4) * neg

    if class_weights is not None:
        w = class_weights.to(pred.device).view(1, -1, 1, 1)
        pos_loss = pos_loss * w
        neg_loss = neg_loss * w

    n = pos.sum()
    pos_loss, neg_loss = pos_loss.sum(), neg_loss.sum()
    return -neg_loss if n == 0 else -(pos_loss + neg_loss) / n


def gather_feat(feat, ind):
    """feat: (B,C,H,W) -> values at flat spatial indices ind: (B,K) -> (B,K,C)."""
    B, C = feat.shape[0], feat.shape[1]
    feat = feat.view(B, C, -1).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(-1).expand(B, ind.shape[1], C))


def reg_l1(pred, ind, mask, target):
    """L1 at ground-truth centres, normalised by object count (CenterNet)."""
    p = gather_feat(pred, ind)
    m = mask.unsqueeze(-1)
    return (torch.abs(p - target) * m).sum() / (mask.sum() + 1e-4)


def rot_sine_loss(pred, ind, mask, target):
    """Flip-invariant heading error.

    |sin(theta_pred - theta_gt)| = |sp*tc - cp*ts| for unit vectors, so a
    prediction rotated by exactly 180 degrees costs nothing and the direction
    classifier is left to resolve the sign. The prediction is L2-normalised
    first: without it the network could drive the loss down by shrinking the
    vector rather than by pointing it correctly.
    """
    p = gather_feat(pred, ind)
    p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
    t = target / (target.norm(dim=-1, keepdim=True) + 1e-6)
    cross = p[..., 0] * t[..., 1] - p[..., 1] * t[..., 0]
    return (cross.abs() * mask).sum() / (mask.sum() + 1e-4)


def dir_loss(pred, ind, mask, target):
    """Cross-entropy over the two half-turn bins."""
    p = gather_feat(pred, ind)                       # (B,K,2)
    ce = F.cross_entropy(p.reshape(-1, 2), target.reshape(-1), reduction="none")
    return (ce * mask.reshape(-1)).sum() / (mask.sum() + 1e-4)


@torch.no_grad()
def iou_aware_targets(out, batch):
    """RangeDet IoU-aware classification: replace the 1.0 at each positive cell
    with the BEV IoU of that cell's decoded box against its ground-truth box.

    Returns (soft_gt, pos_mask). Detached by construction -- this reshapes the
    classification TARGET so the score used for ranking reflects localisation
    quality; it is not a differentiable box loss.
    """
    from rangeview.compat import boxes_iou_bev
    from rangeview.models.net import PC_RANGE, BEV_RES, OUT_STRIDE

    hm_gt = batch["hm"]
    ind, mask = batch["ind"], batch["mask"]
    B, K = ind.shape
    cell = BEV_RES * OUT_STRIDE
    Wm = hm_gt.shape[-1]

    col = (ind % Wm).float()
    row = (ind // Wm).float()

    def to_boxes(off, z, logdim, rot):
        d = logdim.exp()
        return torch.stack([
            (col + off[..., 0]) * cell + PC_RANGE[0],
            (row + off[..., 1]) * cell + PC_RANGE[1],
            z[..., 0], d[..., 0], d[..., 1], d[..., 2],
            torch.atan2(rot[..., 0], rot[..., 1]),
        ], dim=-1)

    pred = to_boxes(gather_feat(out["offset"], ind).float(),
                    gather_feat(out["z"], ind).float(),
                    gather_feat(out["dim"], ind).float(),
                    gather_feat(out["rot"], ind).float())
    gt = to_boxes(batch["offset"].float(), batch["z"].float(),
                  batch["dim"].float(), batch["rot"].float())

    quality = torch.zeros_like(hm_gt)
    flat = quality.view(B, quality.shape[1], -1)
    cls_idx = batch["cls_idx"]
    m = mask.bool()

    for b in range(B):
        sel = m[b]
        if not bool(sel.any()):
            continue
        iou = boxes_iou_bev(pred[b][sel].contiguous(),
                            gt[b][sel].contiguous()).diagonal().clamp(0, 1)
        # a floor keeps an early, badly-placed box from driving its own target
        # to zero, which would stop it ever being learned
        flat[b, cls_idx[b][sel], ind[b][sel]] = iou.clamp(min=0.1).to(flat.dtype)
    return quality


IOU_WEIGHT = 1.0


def iou_head_loss(pred_iou, ind, mask, target_iou):
    """Supervise the IoU head at ground-truth centres.

    Target is the BEV IoU between the cell's own decoded box and its GT box,
    mapped from [0,1] to [-1,1] as AFDetV2 does, and trained with smooth L1.
    Detached target: this teaches the head to PREDICT localisation quality, it
    is not a differentiable box loss.
    """
    p = gather_feat(pred_iou, ind).squeeze(-1)
    t = target_iou * 2.0 - 1.0
    return (F.smooth_l1_loss(p, t, reduction="none") * mask).sum() / (mask.sum() + 1e-4)


@torch.no_grad()
def decoded_bev_iou(out, batch):
    """BEV IoU of each positive cell's decoded box against its GT box."""
    from rangeview.compat import boxes_iou_bev
    from rangeview.models.net import PC_RANGE, BEV_RES, OUT_STRIDE
    ind, mask = batch["ind"], batch["mask"]
    B, K = ind.shape
    cell = BEV_RES * OUT_STRIDE
    Wm = batch["hm"].shape[-1]
    col = (ind % Wm).float(); row = (ind // Wm).float()

    def to_boxes(off, z, logdim, rot):
        d = logdim.exp()
        return torch.stack([
            (col + off[..., 0]) * cell + PC_RANGE[0],
            (row + off[..., 1]) * cell + PC_RANGE[1],
            z[..., 0], d[..., 0], d[..., 1], d[..., 2],
            torch.atan2(rot[..., 0], rot[..., 1])], dim=-1)

    pred = to_boxes(gather_feat(out["offset"], ind).float(),
                    gather_feat(out["z"], ind).float(),
                    gather_feat(out["dim"], ind).float(),
                    gather_feat(out["rot"], ind).float())
    gt = to_boxes(batch["offset"].float(), batch["z"].float(),
                  batch["dim"].float(), batch["rot"].float())
    tgt = torch.zeros(B, K, device=ind.device)
    m = mask.bool()
    for b in range(B):
        sel = m[b]
        if not bool(sel.any()):
            continue
        iou = boxes_iou_bev(pred[b][sel].contiguous(),
                            gt[b][sel].contiguous()).diagonal().clamp(0, 1)
        tgt[b][sel] = iou.to(tgt.dtype)
    return tgt


def compute_loss(out, batch, iou_aware=False, class_weights=None):
    ind, mask = batch["ind"], batch["mask"]

    quality = iou_aware_targets(out, batch) if iou_aware else None

    losses = {
        "hm": CLS_WEIGHT * focal_loss(out["hm"], batch["hm"], quality, class_weights),
        "off": LOC_WEIGHT * CODE_W["offset"] * reg_l1(out["offset"], ind, mask, batch["offset"]),
        "z": LOC_WEIGHT * CODE_W["z"] * reg_l1(out["z"], ind, mask, batch["z"]),
        "dim": LOC_WEIGHT * CODE_W["dim"] * reg_l1(out["dim"], ind, mask, batch["dim"]),
        "rot": LOC_WEIGHT * CODE_W["rot"] * rot_sine_loss(out["rot"], ind, mask, batch["rot"]),
    }
    if "dir" in out:
        losses["dir"] = DIR_WEIGHT * dir_loss(out["dir"], ind, mask, batch["dir"])
    if "iou" in out:
        losses["iou"] = IOU_WEIGHT * iou_head_loss(
            out["iou"], ind, mask, decoded_bev_iou(out, batch))
    losses["total"] = sum(losses.values())
    return losses
