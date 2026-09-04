"""Held-out evaluation under the COMPETITION's own scorer.

Drop-in replacement for evaluate_devkit.evaluate_devkit, with the same
signature and return type, so train.py can import it in place.

Why this exists: evaluate_devkit compares `boxes_overlap_bev_gpu` -- an AREA in
m^2 -- directly against the "iou_thresh", so its "AP@0.5" means overlap >= 0.5
m^2, about 8% of a car. It also scores at a flat 0.30 threshold rather than the
per-class values in decode.SCORE_THRESH. Every training curve produced before
this file was written used that metric, which is why tb showed ~0.86 where the
official scorer said ~0.78, and why epoch selection off those curves could not
be trusted.

This routes through score_exact, which imports scoring_program/scoring.py
directly: true rotated BEV IoU, frame-ordered AP (GLOBAL_SORT_DETECTIONS =
False), the EVAL_RANGE filter, and HEADLINE = {1: 0.5, 2: 0.5, 3: 0.3}.
"""
import numpy as np
import torch

import rangeview.eval.score_exact as S
from rangeview.models.decode import SCORE_THRESH, cross_class_nms, decode, nms_per_class
from rangeview.data.geometry import batch_to_device

CLASS_NAME = {1: "Vehicle", 2: "Bike", 3: "Pedestrian"}


@torch.no_grad()
def evaluate_official(model, loader, device="cuda", score_thresh=None,
                      nms_iou=0.2, verbose=True, cross_class=0.0):
    """Returns (mAP, {class: headline AP}) under scoring_program/scoring.py.

    score_thresh=None uses the per-class decode.SCORE_THRESH, which is what a
    submission uses; passing a float overrides all three classes at once.
    """
    model.eval()
    thr = SCORE_THRESH if score_thresh is None else score_thresh
    frames = []
    nbox = 0
    for b in loader:
        out = model(batch_to_device(b, device))
        for bi, det in enumerate(decode(out, score_thresh=thr)):
            det = nms_per_class(det, iou_thresh=nms_iou)
            if cross_class:
                det = cross_class_nms(det, iou_thresh=cross_class)
            nbox += len(det["boxes"])
            frames.append((det["boxes"].cpu().numpy(),
                           det["scores"].cpu().numpy(),
                           det["labels"].cpu().numpy(),
                           b["gt_boxes_raw"][bi].numpy(),
                           b["gt_class_raw"][bi].numpy()))
        del out

    # only the two headline thresholds; the full sweep triples scoring cost
    mAP, head, _ = S.score_frames(frames, thresholds=(0.3, 0.5))
    if verbose:
        print(f"{'class':<12}{'thr':>6}{'AP':>9}")
        for c in CLASS_NAME:
            print(f"{CLASS_NAME[c]:<12}{S.HEADLINE[c]:>6}{head[c]:>9.4f}")
        print(f"{'mAP':<12}{'':>6}{mAP:>9.4f}   "
              f"({nbox/max(len(frames),1):.1f} boxes/frame, thr={thr})")
    return float(mAP), head
