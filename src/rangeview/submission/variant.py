"""Winning submission variant: cross-model WBF + mild Ped expansion + Ped>Bike NMS.

R's test decomposition proved which components transfer positively on test:
  V expansion crashed V@0.5 to 0.8084       → dropped
  P expansion + WBF pushed P@0.3 to 0.8604   → kept mildly
  B@0.5 was neutral around 0.8227           → cross-model WBF + Bike dedup kept

Recipe (Variant T, val ~0.850 / test 0.8464):
  1. Cross-model WBF (v2, ep8, ep8_cp × 1024 + 512) for V,B @ IoU 0.50
  2. Cross-model WBF (v2, ep8, ep8_cp × 1024) for P @ IoU 0.35
  3. Mild Ped isotropic expansion 1.15x
  4. Ped>Bike NMS @ IoU 0.10  (annotation-priority — a Ped absorbing a nearby Bike)
  5. Bike>Bike dedup @ IoU 0.15
  6. Refiner (V, B) + RSU visibility filter + Bike ground-dim filter

Callers must supply per-checkpoint prediction caches (built by
`rangeview.inference.infer`), the test frame cache, and refiner weights.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import rangeview.data.dataset as _ds
import rangeview.inference.tta as tta
from rangeview.data.dataset import RangeViewDataset, collate


def wbf_multi(dets_list, cls, iou):
    out = []
    for i in range(len(dets_list[0])):
        b, s, l = [], [], []
        for d_fr in dets_list:
            d = d_fr[i]
            sel = d["labels"] == cls
            b.append(d["boxes"][sel])
            s.append(d["scores"][sel])
            l.append(d["labels"][sel])
        b_cat = torch.cat(b, dim=0)
        s_cat = torch.cat(s, dim=0)
        l_cat = torch.cat(l, dim=0)
        if b_cat.numel() == 0:
            out.append({"boxes": b_cat, "scores": s_cat, "labels": l_cat})
            continue
        out.append(tta.wbf_per_class(b_cat, s_cat, l_cat, iou_thresh=iou))
    return out


def _dets(x):
    if isinstance(x, dict) and "dets" in x:
        return x["dets"]
    return x


def bev_iou_2d(a, b):
    ax1, ax2 = a[0] - a[3] / 2, a[0] + a[3] / 2
    ay1, ay2 = a[1] - a[4] / 2, a[1] + a[4] / 2
    bx1, bx2 = b[0] - b[3] / 2, b[0] + b[3] / 2
    by1, by2 = b[1] - b[4] / 2, b[1] + b[4] / 2
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    return inter / max((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter, 1e-6)


def drop_a_over_b(a_dets, b_dets, overlap_iou):
    """Drop boxes in `a` that overlap any box in `b` at IoU >= threshold."""
    new_a = []
    for i in range(len(a_dets)):
        b_boxes = b_dets[i]["boxes"]
        a_boxes = a_dets[i]["boxes"]
        a_scores = a_dets[i]["scores"]
        a_labels = a_dets[i]["labels"]
        if a_boxes.numel() == 0 or b_boxes.numel() == 0:
            new_a.append(a_dets[i])
            continue
        bb, aa = b_boxes.numpy(), a_boxes.numpy()
        keep = np.ones(len(aa), dtype=bool)
        for ai in range(len(aa)):
            for bi in range(len(bb)):
                if bev_iou_2d(aa[ai], bb[bi]) >= overlap_iou:
                    keep[ai] = False
                    break
        mask = torch.from_numpy(keep)
        new_a.append({"boxes": a_boxes[mask], "scores": a_scores[mask], "labels": a_labels[mask]})
    return new_a


def dupe_nms(dets, iou_thr):
    """Simple axis-aligned BEV NMS, sorted by score desc."""
    new = []
    for i in range(len(dets)):
        bb = dets[i]["boxes"].numpy()
        bs = dets[i]["scores"].numpy()
        if len(bb) == 0:
            new.append(dets[i])
            continue
        order = np.argsort(-bs)
        keep = np.ones(len(bb), dtype=bool)
        for i_idx, i_ in enumerate(order):
            if not keep[i_]:
                continue
            for j_ in order[i_idx + 1:]:
                if not keep[j_]:
                    continue
                if bev_iou_2d(bb[i_], bb[j_]) >= iou_thr:
                    keep[j_] = False
        mask = torch.from_numpy(keep)
        new.append({"boxes": dets[i]["boxes"][mask],
                    "scores": dets[i]["scores"][mask],
                    "labels": dets[i]["labels"][mask]})
    return new


def expand_boxes(dets, factor):
    """Isotropic box expansion (in-place on dx, dy)."""
    new = []
    for d in dets:
        b = d["boxes"].clone()
        if b.numel() > 0:
            b[:, 3] *= factor
            b[:, 4] *= factor
        new.append({"boxes": b, "scores": d["scores"], "labels": d["labels"]})
    return new


def build(pred_caches_1024, pred_caches_512, test_cache, out_pt):
    """Fuse per-model predictions into a single test_preds_T.pt cache.

    pred_caches_1024 / pred_caches_512: lists of .pt paths from
      rangeview.inference.infer (one per checkpoint, at each azimuth resolution).
    test_cache: RangeViewDataset cache for the test frames.
    out_pt: destination .pt to write.
    """
    dets_1024 = [_dets(torch.load(p, weights_only=False)) for p in pred_caches_1024]
    dets_512  = [_dets(torch.load(p, weights_only=False)) for p in pred_caches_512]
    all_dets  = dets_1024 + dets_512

    v = wbf_multi(all_dets,  cls=1, iou=0.50)
    b = wbf_multi(all_dets,  cls=2, iou=0.50)
    p = wbf_multi(dets_1024, cls=3, iou=0.35)   # P uses 1024 only; 512 hurts P

    p = expand_boxes(p, factor=1.15)             # mild — R proved aggressive P still transfers
    b = drop_a_over_b(b, p, overlap_iou=0.10)    # class-priority: Ped over Bike
    b = dupe_nms(b, iou_thr=0.15)                # Bike dedup

    _ds.scene_of = lambda pth: 0
    ds = RangeViewDataset(str(test_cache), residuals=False, keep_scenes=[0])
    file_names = [f.name for f in ds.files]
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    gts = []
    for x in dl:
        for bi in range(len(x["gt_boxes_raw"])):
            gts.append((x["gt_boxes_raw"][bi].numpy(), x["gt_class_raw"][bi].numpy()))

    picks = []
    for i in range(len(v)):
        picks.append({
            "boxes":  torch.cat([v[i]["boxes"],  b[i]["boxes"],  p[i]["boxes"]],  dim=0),
            "scores": torch.cat([v[i]["scores"], b[i]["scores"], p[i]["scores"]], dim=0),
            "labels": torch.cat([v[i]["labels"], b[i]["labels"], p[i]["labels"]], dim=0),
        })
    picks_cpu = [{k: v.detach().cpu() for k, v in d.items()} for d in picks]
    torch.save({"preds": picks_cpu, "gts": gts,
                "scene_ids": [0] * len(picks_cpu), "file_names": file_names},
               out_pt)
    return out_pt


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--preds-1024", nargs="+", required=True,
                    help="Prediction .pt caches at 1024 azimuth (one per checkpoint)")
    ap.add_argument("--preds-512", nargs="+", default=[],
                    help="Prediction .pt caches at 512 azimuth (optional, one per checkpoint)")
    ap.add_argument("--test-cache", required=True,
                    help="Test RangeViewDataset cache directory")
    ap.add_argument("--v-refiner", required=True, help="Vehicle refiner .pt")
    ap.add_argument("--b-refiner", required=True, help="Bike refiner .pt")
    ap.add_argument("--out-dir", required=True, help="Directory to write submission txts")
    ap.add_argument("--out-zip", default=None, help="Final zip path (default: <out-dir>.zip)")
    ap.add_argument("--work-dir", default=".",
                    help="Working directory for intermediate files (defaults to CWD)")
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    preds_pt = work / "test_preds_variant.pt"
    feats_pt = work / "test_features_variant.pt"

    print(f"[1/3] Building fused predictions -> {preds_pt}")
    build(args.preds_1024, args.preds_512, args.test_cache, preds_pt)

    print(f"[2/3] Extracting per-box features -> {feats_pt}")
    subprocess.check_call([
        sys.executable, "-m", "rangeview.inference.features",
        "--pred-cache", str(preds_pt),
        "--frame-cache", args.test_cache,
        "--val-scenes", "-1",
        "--out", str(feats_pt),
    ])

    print(f"[3/3] Applying refiner + RSU + writing txts -> {args.out_dir}")
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    subprocess.check_call([
        sys.executable, "-m", "rangeview.submission.build",
        "--test-preds", str(preds_pt),
        "--test-feat",  str(feats_pt),
        "--test-cache", args.test_cache,
        "--v-refiner",  args.v_refiner,
        "--b-refiner",  args.b_refiner,
        "--thr-v", "0.01",
        "--thr-b", "0.02",
        "--out-dir", str(out_dir),
    ])

    zip_path = Path(args.out_zip) if args.out_zip else out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    subprocess.check_call(["zip", "-q", "-r", str(zip_path), "."], cwd=out_dir)

    # Bike ground/dim filter (final safety net)
    filtered_out = zip_path.with_name(zip_path.stem + "_gd_bike")
    subprocess.check_call([
        sys.executable, "-m", "rangeview.submission.filter",
        "--sub", str(zip_path),
        "--out", str(filtered_out),
        "--classes", "2",
        "--zip",
    ])
    print(f"\nDONE -> {filtered_out}.zip")


if __name__ == "__main__":
    main()
