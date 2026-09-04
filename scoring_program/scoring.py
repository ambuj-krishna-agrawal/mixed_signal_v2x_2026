#!/usr/bin/env python3
"""Scoring program for the Mixed Signals V2X 3D Object Detection Challenge.

This is a *result* (prediction) scoring program: participants upload per-frame
detections and this script compares them against held-out ground truth.

It reproduces the official Mixed Signals / OpenCOOD evaluation:

  * 3 meta-classes:        1 = vehicle, 2 = bike, 3 = pedestrian
  * Matching metric:       Bird's-Eye-View (BEV) rotated-box IoU
  * AP definition:         VOC-2010 "all-points" interpolation (area under the
                           precision-recall curve), per class
  * IoU thresholds:        0.3, 0.5, 0.7 (computed for every class)
  * Headline metric (mAP): mean of Vehicle AP@0.5, Bike AP@0.5, Pedestrian AP@0.3
  * Evaluation range:      x, y in [-51.2, 51.2] m  (boxes outside are dropped)
  * Matching rule:         per-frame, greedy, detections sorted by descending
                           score, each ground-truth box matched at most once

The implementation is intentionally dependency-light: only numpy is required.
Rotated-rectangle intersection is computed with a pure-numpy Sutherland-Hodgman
convex-polygon clip + shoelace area, so no shapely / CUDA / OpenCOOD is needed.

Codabench runtime layout (CodaLab-compatible):
  /app/input/ref/   ground-truth files (the task's reference_data)
  /app/input/res/   the participant's uploaded predictions
  /app/output/      where scores.json (+ optional detailed_results.html) go

File formats (one .txt per frame, whitespace-separated, # comments allowed):
  ground truth :  cls cx cy cz dx dy dz yaw
  prediction   :  cls cx cy cz dx dy dz yaw score
where cls in {1,2,3}; (cx,cy,cz) box centre [m]; (dx,dy,dz) length,width,height
[m]; yaw heading about +z [rad]; score in [0,1]. Frames are matched by filename
stem, so prediction frame_000123.txt is compared against ref frame_000123.txt.
A frame present in ref but missing from res is scored as zero detections.
"""

import os
import sys
import json
import glob

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration: keep in sync with the dataset's official benchmark.
# --------------------------------------------------------------------------- #
CLASSES = {1: "vehicle", 2: "bike", 3: "pedestrian"}
IOU_THRESHOLDS = (0.3, 0.5, 0.7)
EVAL_RANGE = (-51.2, -51.2, 51.2, 51.2)  # xmin, ymin, xmax, ymax (BEV)

# Headline per-class operating points (class_id -> iou used for mAP).
HEADLINE = {1: 0.5, 2: 0.5, 3: 0.3}

# Match the dataset's default evaluation (per-frame ranking, no global sort).
GLOBAL_SORT_DETECTIONS = False

# --------------------------------------------------------------------------- #
# Geometry: rotated-box BEV IoU (pure numpy)
# --------------------------------------------------------------------------- #


def _box_to_corners(cx, cy, dx, dy, yaw):
    """Return the 4 BEV corners (CCW) of a rotated box. dx=length, dy=width."""
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = dx / 2.0, dy / 2.0
    # local corners in counter-clockwise order (required by the clip below):
    # front-right, front-left, rear-left, rear-right
    local = np.array([[hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject, clip_poly):
    """Sutherland-Hodgman: clip convex `subject` polygon by convex `clip_poly`."""
    output = list(subject)
    cp = list(clip_poly)
    for i in range(len(cp)):
        if not output:
            break
        a = cp[i]
        b = cp[(i + 1) % len(cp)]
        edge = (b[0] - a[0], b[1] - a[1])

        def inside(p):
            # left-hand side of directed edge a->b (clip poly is CCW)
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0.0

        def intersect(p, q):
            d1 = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            d2 = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            denom = d1 - d2
            if abs(denom) < 1e-12:
                return p
            t = d1 / denom
            return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

        new_output = []
        prev = output[-1]
        prev_in = inside(prev)
        for cur in output:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    new_output.append(intersect(prev, cur))
                new_output.append(cur)
            elif prev_in:
                new_output.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        output = new_output
    return np.array(output) if output else np.empty((0, 2))


def bev_iou(box_a, box_b):
    """BEV IoU between two boxes given as (cx,cy,cz,dx,dy,dz,yaw)."""
    ca = _box_to_corners(box_a[0], box_a[1], box_a[3], box_a[4], box_a[6])
    cb = _box_to_corners(box_b[0], box_b[1], box_b[3], box_b[4], box_b[6])
    inter_poly = _clip(ca, cb)
    inter = _poly_area(inter_poly)
    if inter <= 0.0:
        return 0.0
    union = _poly_area(ca) + _poly_area(cb) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(dets, gts):
    """(N,M) BEV IoU between N detection boxes and M ground-truth boxes."""
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)))
    out = np.zeros((len(dets), len(gts)))
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            out[i, j] = bev_iou(d, g)
    return out


# --------------------------------------------------------------------------- #
# AP computation (VOC-2010 all-points)
# --------------------------------------------------------------------------- #


def voc_ap(rec, prec):
    """Area under the PR curve using VOC-2010 all-point interpolation."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def calculate_ap(stat):
    """stat: dict with lists 'score','tp','fp' and int 'gt'. Returns AP."""
    if stat["gt"] == 0:
        return 0.0
    score = np.array(stat["score"], dtype=float)
    tp = np.array(stat["tp"], dtype=float)
    fp = np.array(stat["fp"], dtype=float)
    if len(tp) == 0:
        return 0.0
    if GLOBAL_SORT_DETECTIONS:
        order = np.argsort(-score)
        tp, fp = tp[order], fp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / float(stat["gt"])
    prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return voc_ap(rec, prec)


def accumulate_frame(result_stat, dets, scores, gts, iou_thr):
    """Greedy per-frame TP/FP matching, appended in descending-score order."""
    stat = result_stat[iou_thr]
    stat["gt"] += len(gts)
    if len(dets) == 0:
        return
    order = np.argsort(-scores)
    dets = dets[order]
    scores = scores[order]
    ious = iou_matrix(dets, gts)
    gt_used = np.zeros(len(gts), dtype=bool)
    for i in range(len(dets)):
        stat["score"].append(float(scores[i]))
        # Match this detection to its highest-IoU *still-unused* ground-truth
        # box, and count it as a TP iff that IoU >= threshold. This mirrors the
        # reference OpenCOOD matching (eval_utils.caluclate_tp_fp / _gpu): each
        # GT is consumed at most once and a detection competes only for the GTs
        # that remain free, in descending-score order.
        matched = False
        if len(gts) > 0:
            row = ious[i].copy()
            row[gt_used] = -1.0  # exclude GTs already taken by a higher-score det
            j = int(np.argmax(row))
            if row[j] >= iou_thr:
                gt_used[j] = True
                matched = True
        stat["tp"].append(1 if matched else 0)
        stat["fp"].append(0 if matched else 1)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def _in_range(box):
    x, y = box[0], box[1]
    return EVAL_RANGE[0] <= x <= EVAL_RANGE[2] and EVAL_RANGE[1] <= y <= EVAL_RANGE[3]


def load_boxes(path, with_score):
    """Return {cls: (boxes[N,7], scores[N])}. Rows out of range are dropped."""
    ncols = 9 if with_score else 8
    out = {c: ([], []) for c in CLASSES}
    if not path or not os.path.isfile(path):
        return {c: (np.zeros((0, 7)), np.zeros((0,))) for c in CLASSES}
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < ncols:
                raise ValueError(
                    f"{os.path.basename(path)} line {ln}: expected {ncols} "
                    f"columns, got {len(parts)}: '{line}'"
                )
            cls = int(float(parts[0]))
            if cls not in CLASSES:
                continue
            box = [float(v) for v in parts[1:8]]
            if not _in_range(box):
                continue
            out[cls][0].append(box)
            out[cls][1].append(float(parts[8]) if with_score else 1.0)
    return {
        c: (
            np.array(b, dtype=float).reshape(-1, 7),
            np.array(s, dtype=float).reshape(-1),
        )
        for c, (b, s) in out.items()
    }


def find_io_dirs():
    """Resolve (ref_dir, res_dir, output_dir) across Codabench/CodaLab layouts."""
    args = sys.argv[1:]
    if len(args) >= 2:
        input_dir, output_dir = args[0], args[1]
    else:
        input_dir = "/app/input"
        output_dir = "/app/output"
    ref_dir = os.path.join(input_dir, "ref")
    res_dir = os.path.join(input_dir, "res")
    if not os.path.isdir(ref_dir):
        ref_dir = input_dir
    if not os.path.isdir(res_dir):
        res_dir = input_dir
    return ref_dir, res_dir, output_dir


def _index_by_stem(directory):
    out = {}
    for p in glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ref_dir, res_dir, output_dir = find_io_dirs()
    os.makedirs(output_dir, exist_ok=True)

    ref_files = _index_by_stem(ref_dir)
    res_files = _index_by_stem(res_dir)
    if not ref_files:
        raise RuntimeError(f"No ground-truth .txt files found under {ref_dir}")

    # Per (class, iou) accumulators.
    stats = {
        c: {t: {"score": [], "tp": [], "fp": [], "gt": 0} for t in IOU_THRESHOLDS}
        for c in CLASSES
    }

    missing = 0
    for stem, ref_path in sorted(ref_files.items()):
        gt = load_boxes(ref_path, with_score=False)
        res_path = res_files.get(stem)
        if res_path is None:
            missing += 1
        pred = load_boxes(res_path, with_score=True)
        for c in CLASSES:
            gts = gt[c][0]
            dets, scores = pred[c]
            for t in IOU_THRESHOLDS:
                accumulate_frame(stats[c], dets, scores, gts, t)

    # AP per class per threshold.
    ap = {c: {t: calculate_ap(stats[c][t]) for t in IOU_THRESHOLDS} for c in CLASSES}

    headline = [ap[c][HEADLINE[c]] for c in CLASSES]
    mAP = float(np.mean(headline))

    scores_out = {
        "mAP": round(mAP, 6),
        "vehicle_ap_30": round(ap[1][0.3], 6),
        "vehicle_ap_50": round(ap[1][0.5], 6),
        "vehicle_ap_70": round(ap[1][0.7], 6),
        "bike_ap_30": round(ap[2][0.3], 6),
        "bike_ap_50": round(ap[2][0.5], 6),
        "bike_ap_70": round(ap[2][0.7], 6),
        "pedestrian_ap_30": round(ap[3][0.3], 6),
        "pedestrian_ap_50": round(ap[3][0.5], 6),
        "pedestrian_ap_70": round(ap[3][0.7], 6),
    }

    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(scores_out, f, indent=2)

    write_detailed_html(os.path.join(output_dir, "detailed_results.html"),
                        ap, scores_out, len(ref_files), len(res_files), missing)

    print("Scoring complete:")
    print(json.dumps(scores_out, indent=2))
    if missing:
        print(f"WARNING: {missing}/{len(ref_files)} frames had no prediction file.")


def write_detailed_html(path, ap, scores_out, n_ref, n_res, missing):
    rows = "".join(
        f"<tr><td>{CLASSES[c]}</td>"
        f"<td>{ap[c][0.3]:.4f}</td><td>{ap[c][0.5]:.4f}</td><td>{ap[c][0.7]:.4f}</td></tr>"
        for c in CLASSES
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:1.5em;color:#222}}
table{{border-collapse:collapse;margin:1em 0}}
th,td{{border:1px solid #ccc;padding:6px 14px;text-align:center}}
th{{background:#f4f4f4}} .m{{font-size:1.4em;font-weight:700}}
</style></head><body>
<h2>Mixed Signals V2X: Detection Results</h2>
<p class="m">mAP = {scores_out['mAP']:.4f}</p>
<p>Headline mAP = mean(Vehicle AP@0.5, Bike AP@0.5, Pedestrian AP@0.3).</p>
<table>
<tr><th>Class</th><th>AP@0.3</th><th>AP@0.5</th><th>AP@0.7</th></tr>
{rows}
</table>
<p>Frames scored: {n_ref} &nbsp;|&nbsp; prediction files: {n_res}
&nbsp;|&nbsp; frames missing predictions: {missing}</p>
<p style="color:#888">BEV rotated-box IoU · VOC-2010 all-point AP ·
eval range x,y &isin; [-51.2, 51.2] m.</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
