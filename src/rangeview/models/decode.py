"""Turn CenterHead outputs into 3D boxes, then run per-class rotated BEV NMS."""
import torch
import torch.nn.functional as F

from rangeview.models.net import PC_RANGE, BEV_RES, OUT_STRIDE, NUM_CLASSES

DIR_OFFSET = 3.141592653589793 / 4      # must match dataset.DIR_OFFSET
DIR_NUM_BINS = 2


def _limit_period(val, offset=0.5, period=2 * 3.141592653589793):
    return val - torch.floor(val / period + offset) * period


def _gather(feat, ind):
    """feat: (B, D, H, W); ind: (B, N) flat spatial -> (B, N, D)."""
    B, D = feat.shape[:2]
    feat = feat.view(B, D, -1).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(-1).expand(B, ind.shape[1], D))


# Per-class score thresholds. These are NOT tuned to hit a box count -- they are
# the points where the model's own confidence becomes informative. Measured
# precision by score band on held-out data:
#     0.01-0.05 -> 0.002    0.20-0.30 -> 0.486
#     0.05-0.10 -> 0.023    0.30-0.50 -> 0.823
#     0.10-0.20 -> 0.112    0.50+     -> 0.904
# Below ~0.2 a detection is more likely wrong than right, and 73% of all raw
# detections landed in the 0.2%-precision band. The old decode took the top 100
# peaks PER CLASS unconditionally, so it manufactured those regardless of what
# the model believed; NMS could not remove them because isolated false peaks
# overlap nothing.
# PER-CLASS thresholds, swept under the DEVKIT evaluator on held-out scenes
# 8/16/26/34 (800 frames). Uniform thresholds leave real score on the table
# because the classes have very different false-positive profiles:
#
#   thr    Vehicle    Bike   Pedestrian
#   0.30    0.7725   0.7956    0.7476
#   0.35    0.7843   0.7829    0.7604
#   0.50    0.4744   0.7231    0.7850      <- vehicle collapses, pedestrian peaks
#
#   per-class optimum  0.7883   vs  0.7759 uniform 0.35  /  0.7719 uniform 0.30
#
# Every peak is interior to the swept range 0.25-0.75, so these are real optima
# rather than grid edges. Pedestrian wants a much stricter threshold than the
# others -- it carries the most false positives -- and Vehicle AP falls off a
# cliff above 0.40, so do not raise it globally.
SCORE_THRESH = {1: 0.35, 2: 0.30, 3: 0.50}


# IoU-aware confidence rectification (AFDetV2 arXiv:2112.09205, CIA-SSD
# arXiv:2012.03015):   f = score^(1-alpha) * iou_pred^alpha
# alpha=0 disables it. Measured with an ORACLE IoU on held-out scenes, re-tuning
# the per-class threshold at each alpha: 0.7868 (alpha 0) -> 0.8953 (0.5) ->
# 0.9159 (1.0). A trained head will not reach the oracle, so alpha is swept
# against the devkit metric on held-out data before use.
IOU_ALPHA = 0.0


# Ground-relative z. When a model is trained with geometry=True its z head
# regresses the box centre's height above the mapped ground, not absolute z
# (target std 0.766 -> 0.186 over 124,667 boxes, because most of the spread is
# just the road sloping across the scene). Decode MUST add the surface back at
# the decoded xy or every box lands about a metre out with no error raised, so
# this is switched on from the checkpoint rather than by hand -- see
# `configure_from_state_dict`.
_GROUND = None


def set_ground_map(path="ground_map.npz"):
    """Enable ground-relative z decoding; pass None to disable."""
    global _GROUND
    if path is None:
        _GROUND = None
        return None
    from rangeview.data.geometry import GroundMap
    _GROUND = GroundMap(path)
    return _GROUND


def configure_from_state_dict(sd, path="ground_map.npz"):
    """Turn ground-relative decoding on iff the checkpoint was trained with it.

    The geometry pack widens the encoder stem to 14 input channels, so the stem
    weight is an unambiguous marker and the two settings cannot be mismatched by
    accident.
    """
    w = sd.get("encoder.stem.net.0.weight")
    geom = w is not None and w.shape[1] == 14
    set_ground_map(path if geom else None)
    return geom


@torch.no_grad()
def decode(out, score_thresh=None, max_det=None, iou_alpha=None):
    """Heatmap -> 3D boxes, keeping every local maximum the model is confident
    about and nothing else.

    There is deliberately no top-K and no fixed output count. The number of
    detections is whatever the model asserts, which is the quantity a detector
    is supposed to decide.

    score_thresh: float, or dict {class_id: float}, or None to use SCORE_THRESH.
    max_det: optional safety cap; None means unbounded.
    """
    hm = torch.sigmoid(out["hm"])
    B, C, H, W = hm.shape

    alpha = IOU_ALPHA if iou_alpha is None else iou_alpha
    if alpha and "iou" in out:
        # the head regresses IoU mapped to [-1,1]; undo that, then rectify
        iou = ((out["iou"].float() + 1.0) * 0.5).clamp(1e-4, 1.0)
        hm = hm.clamp(min=1e-6).pow(1.0 - alpha) * iou.pow(alpha)

    # 3x3 max-pool keeps only local maxima, so neighbouring cells of one peak do
    # not each become a detection
    peak = (F.max_pool2d(hm, 3, stride=1, padding=1) - hm).abs() < 1e-6
    hm = hm * peak

    if score_thresh is None:
        score_thresh = SCORE_THRESH
    if not isinstance(score_thresh, dict):
        score_thresh = {c: float(score_thresh) for c in range(1, NUM_CLASSES + 1)}
    thr = torch.tensor([score_thresh[c] for c in range(1, NUM_CLASSES + 1)],
                       device=hm.device).view(1, C, 1, 1)

    keep = hm >= thr                                  # (B, C, H, W)
    cell = BEV_RES * OUT_STRIDE
    results = []
    for b in range(B):
        cls_i, row_i, col_i = keep[b].nonzero(as_tuple=True)
        if cls_i.numel() == 0:
            results.append({"boxes": hm.new_zeros((0, 7)),
                            "scores": hm.new_zeros((0,)),
                            "labels": torch.zeros(0, dtype=torch.long, device=hm.device)})
            continue
        flat = (row_i * W + col_i).unsqueeze(0)
        g = lambda k: _gather(out[k][b:b + 1], flat)[0]
        off, z, dim, rot = g("offset"), g("z"), g("dim").exp(), g("rot")

        yaw = torch.atan2(rot[..., 0], rot[..., 1])
        if "dir" in out:
            import math
            period = 2 * math.pi / DIR_NUM_BINS
            dl = _gather(out["dir"][b:b + 1], flat)[0].argmax(-1).float()
            yaw = _limit_period(yaw - DIR_OFFSET, 0.0, period) + DIR_OFFSET + period * dl

        bx = (col_i.float() + off[..., 0]) * cell + PC_RANGE[0]
        by = (row_i.float() + off[..., 1]) * cell + PC_RANGE[1]
        bz = z[..., 0]
        if _GROUND is not None:
            g = _GROUND.to(bz.device)
            bz = bz + g.at(torch.stack([bx, by], -1))
        boxes = torch.stack([
            bx, by, bz, dim[..., 0], dim[..., 1], dim[..., 2], yaw], -1)
        scores = hm[b][cls_i, row_i, col_i]
        order = scores.argsort(descending=True)
        if max_det is not None:
            order = order[:max_det]
        results.append({"boxes": boxes[order].float(),
                        "scores": scores[order].float(),
                        "labels": (cls_i[order] + 1).long()})
    return results





@torch.no_grad()
def cross_class_nms(det, iou_thresh=0.55):
    """Suppress boxes of DIFFERENT classes that describe the same object.

    The heatmap uses an independent sigmoid per class, so nothing forces the
    three classes to compete: one cell can fire on bike and pedestrian at once.
    `nms_per_class` then keeps both, by design ("a pedestrian never suppresses an
    overlapping vehicle"). Measured consequence on held-out data: of the Bike
    false positives scoring >= 0.5, **90.9% overlap a ground-truth box of a
    DIFFERENT class** -- they are real objects wearing the wrong label, not
    hallucinations.

    The threshold is deliberately high (0.55, vs 0.2 within a class). Genuinely
    adjacent objects of different classes overlap far less than that -- a
    pedestrian box (1.00 x 0.88 m) fully inside a vehicle box (3.47 x 1.82 m) is
    only IoU 0.14 -- so this removes near-duplicates while leaving a pedestrian
    standing beside a car untouched.
    """
    from rangeview.compat import boxes_bev_iou_cpu

    boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
    if len(boxes) < 2:
        return det
    order = scores.argsort(descending=True)
    b = boxes[order]
    M = torch.from_numpy(boxes_bev_iou_cpu(b.cpu().numpy(), b.cpu().numpy())).to(b.device)
    diff = (labels[order].view(-1, 1) != labels[order].view(1, -1)).to(b.device)
    keep, taken = [], torch.zeros(len(b), dtype=torch.bool, device=b.device)
    for i in range(len(b)):
        if taken[i]:
            continue
        keep.append(i)
        taken |= (M[i] > iou_thresh) & diff[i]      # only across classes
    idx = order[torch.tensor(keep, dtype=torch.long)]
    return {"boxes": boxes[idx], "scores": scores[idx], "labels": labels[idx]}


@torch.no_grad()
def nms_per_class(det, iou_thresh=0.2, max_out=None):
    """Rotated BEV NMS, applied within each class so a pedestrian never
    suppresses an overlapping vehicle."""
    # Rotated BEV NMS via the Shapely fallback in _compat. The OpenCOOD CUDA
    # kernel is unavailable here; the CPU/Shapely path matches its semantics
    # (score-sorted greedy suppression at `iou_thresh`) and is fast enough for
    # eval/submission where box counts per frame are small.
    from rangeview.compat import boxes_bev_iou_cpu

    boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
    if boxes.numel() == 0:
        return det

    keep_all = []
    for c in range(1, NUM_CLASSES + 1):
        sel = (labels == c).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        b, s = boxes[sel], scores[sel]
        order = s.argsort(descending=True)
        b, s, sel = b[order], s[order], sel[order]
        M = torch.from_numpy(boxes_bev_iou_cpu(b.cpu().numpy(), b.cpu().numpy()))
        keep, taken = [], torch.zeros(len(b), dtype=torch.bool)
        for i in range(len(b)):                    # b is already score-sorted
            if taken[i]:
                continue
            keep.append(i)
            taken |= M[i] > iou_thresh
        keep = torch.tensor(keep, dtype=torch.long, device=b.device)
        keep_all.append(sel[keep])

    if not keep_all:
        return {"boxes": boxes[:0], "scores": scores[:0], "labels": labels[:0]}

    idx = torch.cat(keep_all)
    idx = idx[scores[idx].argsort(descending=True)]
    if max_out is not None:
        idx = idx[:max_out]
    return {"boxes": boxes[idx], "scores": scores[idx], "labels": labels[idx]}
