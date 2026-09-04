"""Score a trained checkpoint on held-out scenes using the OFFICIAL scorer.

Uses the per-class score thresholds from `models.decode.SCORE_THRESH` and the
real scoring program (`scoring_program/scoring.py`) via `eval.score_exact` —
true rotated BEV IoU, frame-ordered AP, EVAL_RANGE filter.

Do NOT use `experiments/rangeview_legacy/evaluate_devkit.py` for decisions: it
compares overlap area in m² against the IoU threshold, and an older in-house
evaluator sorted detections globally, which reads ~+0.10 optimistic.

Example:
    uv run python -m rangeview.eval.score_ckpt \
        --ckpt  artifacts/checkpoints/rangeview_epoch7.pth \
        --cache artifacts/cache_train \
        --val-scenes 31 34
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from rangeview.data.dataset import RangeViewDataset, collate
from rangeview.data.geometry import batch_to_device
from rangeview.models import decode as _dec
from rangeview.models import norm as _norm
from rangeview.models.decode import SCORE_THRESH, cross_class_nms, decode, nms_per_class
from rangeview.models.net import RangeViewFusionNet
from rangeview.eval import score_exact as S

CN = {1: "Vehicle", 2: "Bike", 3: "Pedestrian"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-scenes", type=int, nargs="+", required=True,
                    help="Scene ids held out for evaluation (e.g. 31 34)")
    ap.add_argument("--frames", type=int, default=0,
                    help="0 = all held-out frames. Positive N picks N evenly-spaced frames.")
    ap.add_argument("--nms-iou", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cross-class-nms", type=float, default=0.0,
                    help="0 = off. Suppress boxes of different classes on the same object.")
    a = ap.parse_args()

    dev = torch.device(a.device)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    epoch = sd.get("epoch") if isinstance(sd, dict) else None
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd

    _norm.NORM = "batch" if any(k.endswith("running_mean") for k in sd) else "group"
    cin = sd["encoder.stem.net.0.weight"].shape[1]
    _geom = _dec.configure_from_state_dict(sd)
    cfg = dict(use_vfe=any(k.startswith("vfe.") for k in sd),
               use_residuals=cin not in (6, 14),
               use_geometry=_geom,
               use_iou_head=any(k.startswith("head.iou") for k in sd),
               ms_fusion=any(k.startswith("backbone.fuse2") for k in sd))
    model = RangeViewFusionNet(grad_ckpt=False, **cfg).to(dev)
    model.load_state_dict(sd)
    model.eval()

    ds = RangeViewDataset(a.cache, keep_scenes=a.val_scenes, augment=False,
                          residuals=cfg["use_residuals"], geometry=_geom)
    if a.frames and a.frames < len(ds.files):
        idx = np.linspace(0, len(ds.files) - 1, a.frames).round().astype(int)
        ds.files = [ds.files[i] for i in sorted(set(idx.tolist()))]
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, collate_fn=collate)

    print(f"{a.ckpt}  (epoch {epoch})")
    print(f"  " + " | ".join(f"{k.replace('use_','')}={v}" for k, v in cfg.items()))
    print(f"  {len(ds.files)} frames from scenes {a.val_scenes} on {dev}")
    print(f"  thresholds {SCORE_THRESH} | nms_iou {a.nms_iou}"
          + (f" | cross-class {a.cross_class_nms}" if a.cross_class_nms else ""))

    frames = []
    t0 = time.time()
    with torch.no_grad():
        for b in dl:
            out = model(batch_to_device(b, dev))
            for bi, det in enumerate(decode(out, score_thresh=SCORE_THRESH)):
                det = nms_per_class(det, iou_thresh=a.nms_iou)
                if a.cross_class_nms:
                    det = cross_class_nms(det, iou_thresh=a.cross_class_nms)
                frames.append((det["boxes"].cpu().numpy(),
                               det["scores"].cpu().numpy(),
                               det["labels"].cpu().numpy(),
                               b["gt_boxes_raw"][bi].numpy(),
                               b["gt_class_raw"][bi].numpy()))
            del out
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    nb = sum(len(f[0]) for f in frames)
    ngt = sum(len(f[3]) for f in frames)
    print(f"  inference {time.time() - t0:.0f}s | "
          f"{nb / max(len(frames), 1):.1f} boxes/frame vs "
          f"{ngt / max(len(frames), 1):.1f} GT/frame\n")

    mAP, head, ap_tbl = S.score_frames(frames, thresholds=(0.3, 0.5))
    print(f"  {'class':<12}{'AP@0.3':>9}{'AP@0.5':>9}{'headline':>10}")
    for c in CN:
        print(f"  {CN[c]:<12}{ap_tbl[c][0.3]:>9.4f}{ap_tbl[c][0.5]:>9.4f}{head[c]:>10.4f}")
    print(f"  {'-' * 40}")
    print(f"  {'mAP':<12}{'':>9}{'':>9}{mAP:>10.4f}")


if __name__ == "__main__":
    main()
