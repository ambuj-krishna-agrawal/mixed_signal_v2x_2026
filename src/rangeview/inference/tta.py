"""Test-time augmentation for the range-view detector.

Four techniques, each testable in isolation:

  Y-flip     : reverse azimuth columns + negate y in point coords + un-flip
               predicted cy and yaw.  IN training-aug distribution.
  X-flip     : negate x in point coords + un-flip predicted cx and yaw.
               NOT in training aug; expected to hurt.
  yaw-rot    : rotate all point (x, y) around ego origin by theta radians,
               un-rotate predicted (cx, cy) and yaw.  Training aug is <=45 deg,
               so 90/180/270 are OOD.
  WBF        : weighted box fusion in place of NMS within a single pass.
               Groups overlapping boxes, averages coordinates weighted by score.

Per-class NMS-IoU sweep is not TTA proper -- it's just calling nms_per_class
with a different iou_thresh.  Handled inline in the driver.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Input-space transforms.  Each takes a batch dict from `collate()` and returns
# a transformed dict + an inverse function that maps predicted boxes back into
# the original ego frame.

def _flip_boxes_y(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 7) -> (N, 7), inverting a y-negation.  Columns: cx cy cz dx dy dz yaw."""
    out = boxes.clone()
    out[..., 1] = -out[..., 1]
    out[..., 6] = -out[..., 6]      # yaw' = -yaw under y -> -y
    return out


def _flip_boxes_x(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = -out[..., 0]
    # x -> -x sends yaw to pi - yaw (heading reflected across the y axis)
    import math
    out[..., 6] = math.pi - out[..., 6]
    return out


def _rot_boxes(boxes: torch.Tensor, theta: float) -> torch.Tensor:
    out = boxes.clone()
    c, s = float(np.cos(theta)), float(np.sin(theta))
    x, y = out[..., 0].clone(), out[..., 1].clone()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    out[..., 6] = out[..., 6] + theta
    return out


def _clone_agent_map(m: dict) -> dict:
    return {k: v.clone() for k, v in m.items()}


def _flip_images_y(batch: dict) -> dict:
    """Reverse azimuth columns + negate y in stored ego coords, per agent.

    Matches augment.py's train-time y-flip which pairs column reversal with
    a negation of y in the point coords.
    """
    out = {"images": {}, "valid": {}}
    for a, img in batch["images"].items():
        # img shape (B, H, W, C); reverse W then negate y column (index 1)
        im = torch.flip(img, dims=[2])
        im = im.clone()
        im[..., 1] = -im[..., 1]
        out["images"][a] = im
        out["valid"][a] = torch.flip(batch["valid"][a], dims=[2])
    for k in ("hag", "origins"):
        if k in batch and isinstance(batch[k], dict) and batch[k]:
            out[k] = _clone_agent_map(batch[k])  # not touched here
    return out


def _flip_images_x(batch: dict) -> dict:
    """Negate x in stored ego coords, no column change.

    X-flip is NOT part of training augmentation.  The range-view lattice
    represents azimuth in columns and beam in rows: negating x alone moves
    a point to a different azimuth, so the lattice ordering no longer matches
    the coordinate.  We do this anyway for the ablation, but expect degradation.
    """
    out = {"images": {}, "valid": {}}
    for a, img in batch["images"].items():
        im = img.clone()
        im[..., 0] = -im[..., 0]
        out["images"][a] = im
        out["valid"][a] = batch["valid"][a].clone()
    for k in ("hag", "origins"):
        if k in batch and isinstance(batch[k], dict) and batch[k]:
            out[k] = _clone_agent_map(batch[k])
    return out


def _rotate_images(batch: dict, theta: float) -> dict:
    """Rotate stored ego xyz by theta around z; leave the lattice unchanged.

    Like x-flip this desyncs the lattice azimuth ordering from the coordinate
    contents.  Only in-training-distribution (small theta) is expected to help.
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    out = {"images": {}, "valid": {}}
    for a, img in batch["images"].items():
        im = img.clone()
        x = im[..., 0].clone()
        y = im[..., 1].clone()
        im[..., 0] = c * x - s * y
        im[..., 1] = s * x + c * y
        out["images"][a] = im
        out["valid"][a] = batch["valid"][a].clone()
    for k in ("hag", "origins"):
        if k in batch and isinstance(batch[k], dict) and batch[k]:
            out[k] = _clone_agent_map(batch[k])
    return out


# Registry: (transform_batch, unwarp_boxes)
TRANSFORMS = {
    "identity":  (lambda b: b,                       lambda boxes: boxes),
    "y_flip":    (_flip_images_y,                    _flip_boxes_y),
    "x_flip":    (_flip_images_x,                    _flip_boxes_x),
    "rot_p30":   (lambda b: _rotate_images(b, +np.deg2rad(30)),
                  lambda boxes: _rot_boxes(boxes, -np.deg2rad(30))),
    "rot_m30":   (lambda b: _rotate_images(b, -np.deg2rad(30)),
                  lambda boxes: _rot_boxes(boxes, +np.deg2rad(30))),
    "rot_p90":   (lambda b: _rotate_images(b, +np.pi / 2),
                  lambda boxes: _rot_boxes(boxes, -np.pi / 2)),
    "rot_180":   (lambda b: _rotate_images(b, +np.pi),
                  lambda boxes: _rot_boxes(boxes, -np.pi)),
}


# ---------------------------------------------------------------------------
# Weighted Box Fusion (Solovyev et al., 2019).  Replacement for NMS: rather
# than keeping the top-score box in a cluster and dropping the rest, cluster
# them by BEV IoU and take a score-weighted average of their (cx, cy, cz,
# dx, dy, dz, yaw) coordinates.  Yaw is averaged in sin/cos.
#
# NMS is a special case (weights all-or-nothing).  WBF preserves the score
# ranking of the top-of-cluster box, so per-frame AP scoring is unaffected
# by ranking changes -- only the coordinates shift.

from rangeview.compat import boxes_bev_iou_cpu


def wbf_per_class(boxes: torch.Tensor, scores: torch.Tensor,
                  labels: torch.Tensor, iou_thresh: float = 0.55,
                  score_weight: bool = True) -> dict:
    """Group by (class, IoU>=iou_thresh), fuse each cluster.

    boxes:  (N, 7) cx cy cz dx dy dz yaw
    scores: (N,)
    labels: (N,) 1..3
    """
    from rangeview.models.net import NUM_CLASSES
    keep_boxes, keep_scores, keep_labels = [], [], []
    device = boxes.device
    for c in range(1, NUM_CLASSES + 1):
        sel = (labels == c).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        b, s = boxes[sel], scores[sel]
        order = s.argsort(descending=True)
        b, s = b[order], s[order]
        b_np = b.detach().cpu().numpy()
        s_np = s.detach().cpu().numpy()
        M = boxes_bev_iou_cpu(b_np, b_np)
        used = np.zeros(len(b_np), bool)
        for i in range(len(b_np)):
            if used[i]:
                continue
            cluster = np.where((M[i] >= iou_thresh) & (~used))[0]
            used[cluster] = True
            if len(cluster) == 1:
                # single-box cluster: keep as is
                keep_boxes.append(b_np[i])
                keep_scores.append(float(s_np[i]))
            else:
                w = s_np[cluster] if score_weight else np.ones_like(s_np[cluster])
                w = w / w.sum()
                # average xyz, dxdydz
                fused = np.zeros(7, np.float32)
                for k in (0, 1, 2, 3, 4, 5):
                    fused[k] = float((w * b_np[cluster, k]).sum())
                # yaw via sin/cos
                sin_avg = float((w * np.sin(b_np[cluster, 6])).sum())
                cos_avg = float((w * np.cos(b_np[cluster, 6])).sum())
                fused[6] = float(np.arctan2(sin_avg, cos_avg))
                keep_boxes.append(fused)
                # score: keep the cluster's max (preserves per-frame ranking)
                keep_scores.append(float(s_np[cluster].max()))
            keep_labels.append(c)
    if not keep_boxes:
        return {"boxes": boxes[:0], "scores": scores[:0], "labels": labels[:0]}
    out_b = torch.from_numpy(np.stack(keep_boxes)).to(device=device, dtype=boxes.dtype)
    out_s = torch.tensor(keep_scores, dtype=scores.dtype, device=device)
    out_l = torch.tensor(keep_labels, dtype=labels.dtype, device=device)
    # re-sort by score so downstream per-frame ranking is intact
    order = out_s.argsort(descending=True)
    return {"boxes": out_b[order], "scores": out_s[order], "labels": out_l[order]}


# ---------------------------------------------------------------------------
# Multi-pass merge: given a list of per-pass detection dicts (each with boxes/
# scores/labels for one frame), merge them into a single set.
#
# Two strategies:
#   "nms"  : concatenate all detections, run per-class rotated NMS at iou_thresh
#   "wbf"  : concatenate, run WBF at iou_thresh
#
# In both cases, order across passes doesn't matter since NMS/WBF are order-
# invariant (given consistent tie-breaking).

def merge_multipass(dets_per_pass: list, mode: str = "nms",
                    iou_thresh: float = 0.5) -> dict:
    if not dets_per_pass:
        empty = torch.zeros(0)
        return {"boxes": empty.reshape(0, 7), "scores": empty, "labels": empty.long()}
    all_b = torch.cat([d["boxes"]  for d in dets_per_pass], dim=0)
    all_s = torch.cat([d["scores"] for d in dets_per_pass], dim=0)
    all_l = torch.cat([d["labels"] for d in dets_per_pass], dim=0)
    if mode == "wbf":
        return wbf_per_class(all_b, all_s, all_l, iou_thresh=iou_thresh)
    else:
        # NMS across passes: fall through to nms_per_class from decode.
        from rangeview.models.decode import nms_per_class
        return nms_per_class({"boxes": all_b, "scores": all_s, "labels": all_l},
                             iou_thresh=iou_thresh)
