"""Exact local scoring: imports the challenge's own scoring.py functions.

Uses `bev_iou`, `accumulate_frame`, `calculate_ap` and `HEADLINE` straight from
`scoring_program/scoring.py`, so the numbers are produced by the leaderboard's
code rather than a reimplementation. Two earlier evaluators in this project were
each wrong in a different way (see FINDINGS.md section 13), which is why this
imports rather than reimplements.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_SP = Path("/home/mehul/mixed_signals/scoring_program/scoring.py")
_spec = importlib.util.spec_from_file_location("official_scoring", _SP)
official = importlib.util.module_from_spec(_spec)
sys.modules["official_scoring"] = official
_spec.loader.exec_module(official)

CLASSES = official.CLASSES
HEADLINE = official.HEADLINE
EVAL_RANGE = official.EVAL_RANGE


def in_range(boxes):
    if len(boxes) == 0:
        return np.zeros(0, bool)
    x, y = boxes[:, 0], boxes[:, 1]
    return ((x >= EVAL_RANGE[0]) & (x <= EVAL_RANGE[2])
            & (y >= EVAL_RANGE[1]) & (y <= EVAL_RANGE[3]))


def score_frames(frames, thresholds=None):
    """frames: list of (det_boxes, det_scores, det_labels, gt_boxes, gt_labels),
    all numpy. Returns (mAP, {class: headline AP}, full ap table)."""
    thr = thresholds or official.IOU_THRESHOLDS
    stats = {c: {t: {"score": [], "tp": [], "fp": [], "gt": 0} for t in thr}
             for c in CLASSES}
    for db, ds_, dl, gb, gl in frames:
        for c in CLASSES:
            gm = gl == c
            g = gb[gm]
            g = g[in_range(g)] if len(g) else g.reshape(-1, 7)
            dm = dl == c
            d, s = db[dm], ds_[dm]
            if len(d):
                keep = in_range(d)
                d, s = d[keep], s[keep]
            for t in thr:
                official.accumulate_frame(stats[c], d.reshape(-1, 7), s, g, t)
    ap = {c: {t: official.calculate_ap(stats[c][t]) for t in thr} for c in CLASSES}
    head = {c: ap[c][HEADLINE[c]] for c in CLASSES}
    return float(np.mean(list(head.values()))), head, ap
