"""mAP evaluation on held-out scenes.

Reports per-class AP at BEV-IoU thresholds, using all-point (VOC-2010 style)
interpolation rather than the 11-point variant, which is what modern 3D
detection benchmarks use.

Loss alone cannot tell us whether detection works: the regression branches are
only scored at ground-truth centres, so a model that misses objects entirely
still shows a healthy regression loss. AP measures precision *and* recall.
"""
import argparse
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset import RangeViewDataset, collate
from model import RangeViewFusionNet, NUM_CLASSES
from decode import decode, nms_per_class

from tqdm import tqdm

CLASS_NAME = {1: "Vehicle", 2: "Bike", 3: "Pedestrian"}
# Looser for the small classes, as KITTI/nuScenes do -- a 0.7 IoU on a 0.8m
# pedestrian demands centimetre accuracy that no LiDAR detector achieves.
DEFAULT_IOU = {1: 0.5, 2: 0.5, 3: 0.3}


def bev_iou(a, b):
    """a: (N,7), b: (M,7) -> (N,M) rotated BEV IoU."""
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev, boxes_bev_iou_cpu
    if len(a) == 0 or len(b) == 0:
        return torch.zeros(len(a), len(b))
    if not torch.cuda.is_available():
        return torch.from_numpy(boxes_bev_iou_cpu(a.cpu().numpy().astype("float32"),
                                                  b.cpu().numpy().astype("float32")))
    return boxes_iou_bev(a.cuda().contiguous().float(),
                         b.cuda().contiguous().float()).cpu()


def average_precision(scores, tp, n_gt):
    """All-point interpolated AP from a score-sorted TP indicator."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # make precision monotonically non-increasing, then integrate over recall
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(np.diff(np.concatenate([[0.0], recall])) > 0)[0]
    return float(np.sum((recall[idx] - np.concatenate([[0.0], recall])[idx]) * precision[idx]))


@torch.no_grad()
def evaluate(model, loader, device, iou_thresh=None, score_thresh=0.01,
             nms_iou=0.2, verbose=True):
    iou_thresh = iou_thresh or DEFAULT_IOU
    model.eval()

    per_cls = defaultdict(lambda: {"scores": [], "tp": [], "n_gt": 0})

    for batch in tqdm(loader, desc="Evaluating"):
        imgs = {k: v.to(device) for k, v in batch["images"].items()}
        valid = {k: v.to(device) for k, v in batch["valid"].items()}
        dets = decode(model({"images": imgs, "valid": valid}), score_thresh=score_thresh)

        for bi, det in enumerate(dets):
            det = nms_per_class(det, iou_thresh=nms_iou)
            gt_b = batch["gt_boxes_raw"][bi]
            gt_c = batch["gt_class_raw"][bi]

            for c in range(1, NUM_CLASSES + 1):
                g = gt_b[gt_c == c]
                per_cls[c]["n_gt"] += len(g)

                sel = det["labels"] == c
                p, s = det["boxes"][sel].cpu(), det["scores"][sel].cpu()
                if len(p) == 0:
                    continue
                order = s.argsort(descending=True)
                p, s = p[order], s[order]

                if len(g) == 0:
                    per_cls[c]["scores"] += s.tolist()
                    per_cls[c]["tp"] += [0] * len(s)
                    continue

                ious = bev_iou(p, g)
                taken = set()
                for i in range(len(p)):
                    j = int(ious[i].argmax())
                    hit = ious[i, j] >= iou_thresh[c] and j not in taken
                    if hit:
                        taken.add(j)
                    per_cls[c]["scores"].append(float(s[i]))
                    per_cls[c]["tp"].append(1 if hit else 0)

    aps = {}
    for c in range(1, NUM_CLASSES + 1):
        d = per_cls[c]
        aps[c] = average_precision(d["scores"], d["tp"], d["n_gt"])

    valid_aps = [v for v in aps.values() if not np.isnan(v)]
    mAP = float(np.mean(valid_aps)) if valid_aps else float("nan")

    if verbose:
        print(f"{'class':<12}{'IoU':>6}{'AP':>9}{'#GT':>8}{'#det':>8}")
        for c in range(1, NUM_CLASSES + 1):
            print(f"{CLASS_NAME[c]:<12}{iou_thresh[c]:>6.2f}{aps[c]:>9.4f}"
                  f"{per_cls[c]['n_gt']:>8}{len(per_cls[c]['scores']):>8}")
        print(f"{'mAP':<12}{'':>6}{mAP:>9.4f}")
    return mAP, aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="/home/mehul/mixed_signals/rangeview/cache")
    ap.add_argument("--val-scenes", type=int, nargs="*", default=[31, 34])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only N frames")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    ds = RangeViewDataset(a.cache, keep_scenes=a.val_scenes)
    if a.limit:
        ds.files = ds.files[: a.limit]
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, collate_fn=collate)

    sd = torch.load(a.ckpt, map_location=device)
    sd = sd["model"] if "model" in sd else sd
    # build with the norm this checkpoint was trained with: model.py now
    # defaults to GroupNorm, older checkpoints are BatchNorm
    import norm as _norm
    _norm.NORM = "batch" if any(k.endswith("running_mean") for k in sd) else "group"
    model = RangeViewFusionNet().to(device)
    model.load_state_dict(sd)
    print(f"ckpt {a.ckpt} (epoch {sd.get('epoch','?')}) | {len(ds)} val frames "
          f"from scenes {a.val_scenes}\n")
    evaluate(model, dl, device, score_thresh=a.score_thresh, nms_iou=a.nms_iou)


if __name__ == "__main__":
    main()
