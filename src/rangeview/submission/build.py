"""Write submission using DIFFERENT refiners per class.

Vehicle refined by refiner_ckpts_2x.pt (thr=0.07)
Bike    refined by refiner_ckpts.pt    (thr=0.02)
Pedestrian: no refiner (fallback to raw score)

Then applies the RSU-visibility filter (drop boxes with <=5 top+dome points
after excluding no-return locus).  The filter is the dataset's annotation
rule: `num_min_points_in_boxes = 5` evaluated on top+dome, so a box with
fewer RSU points cannot be matched by any GT and is charged as an FP.
Toggle with --no-rsu-filter to disable.

Match the val mAP 0.8337 config.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rangeview.models.decode import cross_class_nms
from rangeview.paths import ARTIFACTS
from rangeview.refiner.refiner import FEATURES, transform, Refiner, predict_prob
from rangeview.refiner.ensemble import load_mlp_ckpt as load_refiner_ckpt
from rangeview.refiner.util import frame_id

# RSU-visibility filter constants (mirrors /Users/amagrawal/Downloads/filter_rsu.py).
# Kept in-file rather than imported so this script has no external dep.
RSU_ORIGINS = {"top": np.array([0.0, 0.0, 0.0], np.float32),
               "dome": np.array([0.044, 0.014, -0.263], np.float32)}
LOCUS_R = 0.05


def _points_in_box(pts, box):
    q = pts - box[:3]
    c, s = np.cos(-box[6]), np.sin(-box[6])
    lx = q[:, 0] * c - q[:, 1] * s
    ly = q[:, 0] * s + q[:, 1] * c
    return int(((np.abs(lx) <= box[3] / 2) & (np.abs(ly) <= box[4] / 2)
                & (np.abs(q[:, 2]) <= box[5] / 2)).sum())


def _rsu_points(cache_dir, stem):
    z = np.load(cache_dir / f"{stem}.npz", allow_pickle=True)
    out = []
    for agent, origin in RSU_ORIGINS.items():
        if f"{agent}_img" not in z:
            continue
        img = z[f"{agent}_img"][..., :3].astype(np.float32).reshape(-1, 3)
        valid = z[f"{agent}_valid"].reshape(-1).astype(bool)
        p = img[valid]
        out.append(p[np.linalg.norm(p - origin, axis=1) > LOCUS_R])
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-preds", required=True)
    ap.add_argument("--test-feat", required=True)
    ap.add_argument("--test-cache", default=str(ARTIFACTS / "cache_test_full"))
    ap.add_argument("--v-refiner", required=True,
                    help="Vehicle refiner ckpt (e.g. refiner_ckpts_2x.pt)")
    ap.add_argument("--b-refiner", required=True,
                    help="Bike refiner ckpt (e.g. refiner_ckpts.pt)")
    ap.add_argument("--thr-v", type=float, required=True)
    ap.add_argument("--thr-v-far", type=float, default=None,
                    help="Vehicle refiner threshold at 30-40m specifically "
                         "(QW-V1a: use higher e.g. 0.12).  If None, uses --thr-v.")
    ap.add_argument("--thr-b", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rsu-filter", dest="rsu_filter", action="store_true", default=True,
                    help="apply RSU-visibility filter after refiner (default ON)")
    ap.add_argument("--no-rsu-filter", dest="rsu_filter", action="store_false",
                    help="skip the RSU-visibility filter")
    ap.add_argument("--rsu-thresh", type=int, default=5,
                    help="drop boxes with <= this many top+dome points (default 5)")
    ap.add_argument("--rsu-classes", type=int, nargs="+", default=[1, 2, 3],
                    help="classes the RSU filter applies to (default: all)")
    a = ap.parse_args()

    print(f"loading test preds {a.test_preds}", flush=True)
    blob = torch.load(a.test_preds, weights_only=False)
    preds_raw = blob["preds"]
    print(f"  {len(preds_raw)} frames", flush=True)

    ff = torch.load(a.test_feat, weights_only=False)
    feat_rows = ff["pred_rows"]

    print(f"loading Vehicle refiner {a.v_refiner}", flush=True)
    v_models, v_stats, _ = load_refiner_ckpt(a.v_refiner)
    print(f"loading Bike refiner {a.b_refiner}", flush=True)
    b_models, b_stats, _ = load_refiner_ckpt(a.b_refiner)

    frame_paths = sorted(Path(a.test_cache).glob("*.npz"))
    assert len(frame_paths) == len(preds_raw)

    # per-row prob + keep mask, using per-class refiner
    keep = np.ones(len(feat_rows), dtype=bool)
    for cls_id, models, stats, thr in [
            (1, v_models, v_stats, a.thr_v),
            (2, b_models, b_stats, a.thr_b),
    ]:
        rows_c = [(i, r) for i, r in enumerate(feat_rows) if r["cls"] == cls_id]
        if not rows_c or models.get(cls_id) is None:
            continue
        X = np.array([[r[f] for f in FEATURES[cls_id]] for _, r in rows_c],
                     dtype=np.float32)
        X = transform(X, cls_id).astype(np.float32)
        m_stat, s_stat = stats[cls_id]
        X = ((X - m_stat) / s_stat).astype(np.float32)
        p = predict_prob(models[cls_id], X)
        for (i, r), pp in zip(rows_c, p):
            thr_here = thr
            # QW-V1a: Vehicle threshold at 30-40m can be higher
            if cls_id == 1 and a.thr_v_far is not None and 30 <= r["ego_dist"] < 40:
                thr_here = a.thr_v_far
            keep[i] = (float(pp) >= thr_here)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.txt"):
        f.unlink()

    test_cache_dir = Path(a.test_cache)
    rsu_classes = set(a.rsu_classes)
    fi_ptr = 0; total_boxes = 0; rsu_dropped = 0
    for fi, (raw, fp) in enumerate(zip(preds_raw, frame_paths)):
        d = cross_class_nms(raw, iou_thresh=0.30)
        order = d["scores"].argsort(descending=True)
        d = {k: v[order] for k, v in d.items()}
        pb = d["boxes"].numpy(); ps = d["scores"].numpy(); pl = d["labels"].numpy()
        n = len(pl)
        keep_frame = keep[fi_ptr:fi_ptr + n]
        fi_ptr += n
        pb = pb[keep_frame]; ps = ps[keep_frame]; pl = pl[keep_frame]
        srt = np.argsort(-ps)
        pb = pb[srt]; ps = ps[srt]; pl = pl[srt]

        # RSU-visibility filter (dataset annotation rule; applied AFTER refiner)
        if a.rsu_filter and len(pb):
            stem = frame_id(fp.name)
            npz_path = test_cache_dir / f"{stem}.npz"
            if npz_path.exists():
                rsu_pts = _rsu_points(test_cache_dir, stem)
                keep_rsu = np.ones(len(pb), dtype=bool)
                for bi in range(len(pb)):
                    if int(pl[bi]) not in rsu_classes:
                        continue
                    if _points_in_box(rsu_pts, pb[bi]) <= a.rsu_thresh:
                        keep_rsu[bi] = False
                rsu_dropped += int((~keep_rsu).sum())
                pb = pb[keep_rsu]; ps = ps[keep_rsu]; pl = pl[keep_rsu]

        lines = []
        for bi in range(len(pb)):
            cx, cy, cz, dx, dy, dz, yaw = pb[bi]
            yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
            yaw = float(np.clip(yaw, -3.1415, 3.1415))
            lines.append(
                f"{int(pl[bi])} {cx:.4f} {cy:.4f} {cz:.4f} "
                f"{dx:.4f} {dy:.4f} {dz:.4f} {yaw:.4f} {float(ps[bi]):.4f}"
            )
        (out_dir / f"{frame_id(fp.name)}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        total_boxes += len(pb)
    rsu_msg = f"  RSU filter dropped {rsu_dropped}" if a.rsu_filter else "  RSU filter OFF"
    print(f"wrote {len(preds_raw)} frames, {total_boxes} boxes  "
          f"(refiner dropped {int(len(feat_rows) - keep.sum())}){rsu_msg}",
          flush=True)


if __name__ == "__main__":
    main()
