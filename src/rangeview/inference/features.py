"""Extract per-prediction features from val cache for TP/FP discrimination.

For every prediction (after cc-NMS 0.30) and every GT, compute:
  point_count      : LiDAR points inside the (rotated) box, all agents fused
  point_density    : points / box_volume  (points/m^3)
  intensity_mean   : mean intensity of in-box points (channel 3 of _img)
  intensity_std    : std intensity
  z_top / z_bot    : max/min z of in-box points
  z_span           : z_top - z_bot
  n_agents         : how many of the 5 agents contributed >=1 point inside box
  best_iou_same    : IoU to nearest same-class GT (0 if none)
  best_iou_any     : IoU to nearest GT of ANY class (0 if none)
  best_iou_diff    : IoU to nearest DIFFERENT-class GT (class confusion signal)
  ego_dist         : sqrt(x^2 + y^2) box centre
  box_score        : original detection score
  box_class        : detection class (1/2/3)
  is_tp            : ground-truth label (1=TP, 0=FP) at HEADLINE IoU thresholds

Also computed for GTs (to compare distributions for coverage / diagnostics).

Saves everything to `val_features.pt` for downstream analysis / refiner train.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rangeview.models.decode import cross_class_nms
from rangeview.paths import ARTIFACTS, SCORING

_SP = SCORING / "scoring.py"
_spec = importlib.util.spec_from_file_location("official_scoring", _SP)
_official = importlib.util.module_from_spec(_spec)
sys.modules["official_scoring"] = _official
_spec.loader.exec_module(_official)
iou_matrix = _official.iou_matrix
HEADLINE = _official.HEADLINE

AGENT_KEYS = ["top", "dome", "laser", "003", "004"]
PC_Z_MIN, PC_Z_MAX = -6.0, 4.0  # keep points that could plausibly be objects


def load_frame_points(npz_path):
    """Return per-agent (N,4) arrays: xyz + intensity.  Also fused (M,4)."""
    d = np.load(npz_path)
    per_agent = {}
    for a in AGENT_KEYS:
        key_img = f"{a}_img"; key_val = f"{a}_valid"
        if key_img not in d.files:
            continue
        img = d[key_img].astype(np.float32)
        val = d[key_val].astype(bool) if key_val in d.files else \
              np.ones(img.shape[:2], dtype=bool)
        xyz = img[..., :3][val]
        inten = img[..., 3][val] if img.shape[-1] > 3 else np.zeros(len(xyz))
        m = (np.abs(xyz[:, 0]) < 60) & (np.abs(xyz[:, 1]) < 60) \
            & (xyz[:, 2] > PC_Z_MIN) & (xyz[:, 2] < PC_Z_MAX)
        pts = np.concatenate([xyz[m], inten[m][:, None]], axis=1)
        per_agent[a] = pts
    fused = np.concatenate(list(per_agent.values()), axis=0)
    return per_agent, fused


def points_in_box(points_xyz, box, margin=0.0):
    """Return bool mask of which points fall inside the rotated 3D box.
    box = (cx, cy, cz, dx, dy, dz, yaw).  margin expands each side.
    """
    cx, cy, cz, dx, dy, dz, yaw = box[:7]
    dx += 2 * margin; dy += 2 * margin; dz += 2 * margin
    dx_local = points_xyz[:, 0] - cx
    dy_local = points_xyz[:, 1] - cy
    cs, sn = np.cos(-yaw), np.sin(-yaw)  # rotate INTO box frame
    x_l = cs * dx_local - sn * dy_local
    y_l = sn * dx_local + cs * dy_local
    z_l = points_xyz[:, 2] - cz
    return ((np.abs(x_l) <= dx / 2) & (np.abs(y_l) <= dy / 2)
            & (np.abs(z_l) <= dz / 2))


def box_features(box, per_agent, fused):
    """Per-box features from point-cloud membership.  Points are xyz + intensity.

    Adds height-profile percentiles and expanded-context ratio to help
    discriminate pedestrian hallucinations (which have similar point count
    inside vs just-outside the box, unlike real objects).
    """
    cx, cy, cz, dx, dy, dz, yaw = box[:7]
    fused_xyz = fused[:, :3]
    # coarse gate on x,y first for speed
    coarse = ((np.abs(fused_xyz[:, 0] - cx) < (dx + dy) + 1.5) &
              (np.abs(fused_xyz[:, 1] - cy) < (dx + dy) + 1.5) &
              (np.abs(fused_xyz[:, 2] - cz) < dz + 1))
    fx = fused[coarse]
    inside = points_in_box(fx[:, :3], box) if len(fx) else np.zeros(0, dtype=bool)
    ipts = fx[inside]
    vol = max(dx * dy * dz, 1e-3)
    if len(ipts):
        n = len(ipts)
        z_top = float(ipts[:, 2].max()); z_bot = float(ipts[:, 2].min())
        z_span = z_top - z_bot
        z10, z25, z50, z75, z90 = np.percentile(ipts[:, 2], [10, 25, 50, 75, 90])
        i_mean = float(ipts[:, 3].mean()); i_std = float(ipts[:, 3].std())
    else:
        n = 0; z_top = z_bot = z_span = i_mean = i_std = 0.0
        z10 = z25 = z50 = z75 = z90 = 0.0

    # expanded box context: points in a shell just outside the box
    # (real objects have a density cliff; hallucinations don't)
    shell_inside = points_in_box(fx[:, :3], box, margin=1.0) if len(fx) else np.zeros(0, dtype=bool)
    n_shell = int(shell_inside.sum() - inside.sum())  # points in shell only
    shell_vol = max((dx + 2.0) * (dy + 2.0) * (dz + 2.0) - vol, 1e-3)
    density_inside = n / vol
    density_shell  = n_shell / shell_vol
    density_contrast = density_inside / (density_shell + 1.0)  # >>1 = sharp cliff

    # per-agent count
    n_agents = 0
    for a, pa in per_agent.items():
        if not len(pa):
            continue
        pa_xyz = pa[:, :3]
        coarse_a = ((np.abs(pa_xyz[:, 0] - cx) < (dx + dy)) &
                    (np.abs(pa_xyz[:, 1] - cy) < (dx + dy)) &
                    (np.abs(pa_xyz[:, 2] - cz) < dz + 1))
        if not coarse_a.any():
            continue
        if points_in_box(pa_xyz[coarse_a], box).any():
            n_agents += 1

    ego = float(np.hypot(cx, cy))
    return {
        "point_count":   int(n),
        "point_density": float(density_inside),
        "intensity_mean": i_mean,
        "intensity_std":  i_std,
        "z_top": z_top, "z_bot": z_bot, "z_span": z_span,
        "z10": float(z10), "z25": float(z25), "z50": float(z50),
        "z75": float(z75), "z90": float(z90),
        "n_shell": n_shell,
        "density_contrast": float(density_contrast),
        "n_agents": int(n_agents),
        "ego_dist": ego,
        "box_volume": float(vol),
    }


def context_features(pred_boxes, pred_scores, pred_labels, idx, cls):
    """For pred at index `idx` (class `cls`), compute nearby-pred stats.

    Signals whether this box is 'clustered' (double detection) or 'isolated'.
    """
    cx, cy = pred_boxes[idx, 0], pred_boxes[idx, 1]
    dists = np.hypot(pred_boxes[:, 0] - cx, pred_boxes[:, 1] - cy)
    dists[idx] = 1e9  # exclude self
    same_cls = (pred_labels == cls)
    same_cls[idx] = False
    diff_cls = (pred_labels != cls)

    # nearest same-class pred (distance + score ratio)
    if same_cls.any():
        near_same_idx = int(np.argmin(dists + (~same_cls).astype(float) * 1e6))
        near_same_dist = float(dists[near_same_idx])
        near_same_score = float(pred_scores[near_same_idx])
        higher_score_near_same = (
            (pred_scores > pred_scores[idx]) & same_cls & (dists < 5.0)
        )
        has_higher_near = bool(higher_score_near_same.any())
        n_same_within_5m = int(((dists < 5.0) & same_cls).sum())
    else:
        near_same_dist = 50.0
        near_same_score = 0.0
        has_higher_near = False
        n_same_within_5m = 0

    # nearest diff-class pred (distance)
    if diff_cls.any():
        near_diff_dist = float(dists[diff_cls].min())
        n_any_within_3m = int((dists < 3.0).sum())
    else:
        near_diff_dist = 50.0
        n_any_within_3m = 0

    return {
        "near_same_dist":  near_same_dist,
        "near_same_score": near_same_score,
        "has_higher_near": int(has_higher_near),
        "n_same_within_5m": n_same_within_5m,
        "near_diff_dist":  near_diff_dist,
        "n_any_within_3m": n_any_within_3m,
    }


def match_labels(preds, gt_boxes, gt_labels):
    """Return (is_tp, best_iou_same, best_iou_any, best_iou_diff) per pred."""
    p_boxes  = preds["boxes"].numpy()
    p_scores = preds["scores"].numpy()
    p_labels = preds["labels"].numpy()
    N = len(p_scores)
    is_tp = np.zeros(N, dtype=bool)
    best_same = np.zeros(N); best_any = np.zeros(N); best_diff = np.zeros(N)
    if N == 0:
        return is_tp, best_same, best_any, best_diff
    for c in (1, 2, 3):
        thr = HEADLINE[c]
        pi = np.nonzero(p_labels == c)[0]
        gi_same = np.nonzero(gt_labels == c)[0]
        gi_diff = np.nonzero(gt_labels != c)[0]
        if len(pi) == 0:
            continue
        d_boxes  = p_boxes[pi]; d_scores = p_scores[pi]
        # match to same class
        gt_used = np.zeros(len(gi_same), dtype=bool)
        order = np.argsort(-d_scores)
        if len(gi_same):
            ious_same = iou_matrix(d_boxes, gt_boxes[gi_same])
            for k in order:
                row = ious_same[k].copy()
                row[gt_used] = -1
                j = int(np.argmax(row))
                v = float(row[j])
                if v >= thr:
                    gt_used[j] = True
                    is_tp[pi[k]] = True
                best_same[pi[k]] = max(best_same[pi[k]], v)
        # nearest GT any class
        if len(gt_labels):
            ious_any = iou_matrix(d_boxes, gt_boxes)
            best_any[pi] = ious_any.max(axis=1)
        # nearest GT of different class (class-confusion)
        if len(gi_diff):
            ious_diff = iou_matrix(d_boxes, gt_boxes[gi_diff])
            best_diff[pi] = ious_diff.max(axis=1)
    return is_tp, best_same, best_any, best_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-cache", default=str(ARTIFACTS / "val_preds_cache.pt"))
    ap.add_argument("--frame-cache", default=str(ARTIFACTS / "cache_train"))
    ap.add_argument("--val-scenes", type=int, nargs="+", default=[31, 34])
    ap.add_argument("--out", default=str(ARTIFACTS / "val_features.pt"))
    a = ap.parse_args()

    blob = torch.load(a.pred_cache, weights_only=False)
    preds_raw = blob["preds"]; gts = blob["gts"]; scene_ids = blob["scene_ids"]
    frame_cache = Path(a.frame_cache)
    if a.val_scenes == [-1]:
        # test mode: use all files, don't filter by scene id
        frame_paths = sorted(frame_cache.glob("*.npz"))
    else:
        frame_paths = sorted(
            [f for f in frame_cache.glob("*.npz")
             if int(f.name.split("_")[0][3:]) in set(a.val_scenes)])
    assert len(frame_paths) == len(preds_raw)

    print(f"processing {len(preds_raw)} frames...", flush=True)
    t0 = time.time()
    all_pred_rows = []
    all_gt_rows = []
    for i, (raw, (gb, gl), fp) in enumerate(zip(preds_raw, gts, frame_paths)):
        # cc-NMS 0.30 to match scoring pipeline
        d = cross_class_nms(raw, iou_thresh=0.30)
        order = d["scores"].argsort(descending=True)
        pred = {k: v[order] for k, v in d.items()}

        per_agent, fused = load_frame_points(fp)

        is_tp, biou_same, biou_any, biou_diff = match_labels(pred, gb, gl)

        # features per pred
        p_boxes  = pred["boxes"].numpy()
        p_scores = pred["scores"].numpy()
        p_labels = pred["labels"].numpy()
        for k in range(len(p_scores)):
            f = box_features(p_boxes[k], per_agent, fused)
            f.update(context_features(p_boxes, p_scores, p_labels,
                                       k, int(p_labels[k])))
            f.update({
                "frame": i, "scene": int(scene_ids[i]),
                "cls": int(p_labels[k]), "score": float(p_scores[k]),
                "is_tp": bool(is_tp[k]),
                "best_iou_same": float(biou_same[k]),
                "best_iou_any":  float(biou_any[k]),
                "best_iou_diff": float(biou_diff[k]),
            })
            all_pred_rows.append(f)
        # features per GT (for diagnostics)
        for k in range(len(gl)):
            f = box_features(gb[k], per_agent, fused)
            f.update({
                "frame": i, "scene": int(scene_ids[i]),
                "cls": int(gl[k]),
            })
            all_gt_rows.append(f)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(preds_raw)}  "
                  f"({(i+1)/(time.time()-t0):.1f} f/s)", flush=True)

    print(f"processed in {time.time()-t0:.0f}s "
          f"({len(all_pred_rows)} pred rows, {len(all_gt_rows)} gt rows)",
          flush=True)

    torch.save({"pred_rows": all_pred_rows, "gt_rows": all_gt_rows},
               a.out)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
