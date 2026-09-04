"""Apply the trained refiner to test predictions and write a submission zip.

Pipeline:
  1. Load test_preds_cache.pt (raw ensemble preds on test) + test_features.pt
  2. Load refiner ckpts + stats
  3. Compute refined_score per prediction: base^(1-alpha) * refiner_prob^alpha
  4. cc-NMS 0.30, sort by refined score
  5. Write one <frame_id>.txt per frame  cls cx cy cz dx dy dz yaw score
  6. Zip up.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rangeview.models.decode import cross_class_nms
from rangeview.paths import ARTIFACTS
from rangeview.refiner.refiner import (
    FEATURES, transform, Refiner, predict_prob,
)


def frame_id(cache_name: str) -> str:
    stem = Path(cache_name).stem
    if "_idx" not in stem:
        return stem
    seq, idx = stem.replace("seq", "").split("_idx")
    return f"mini_{seq}_{idx}"


def load_refiner_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    stats = {int(c): (np.array(v[0], dtype=np.float32),
                      np.array(v[1], dtype=np.float32))
             for c, v in ck["stats"].items()}
    models = {}
    for c_str, sd_list in ck["state_dicts"].items():
        c = int(c_str)
        if sd_list is None:
            models[c] = None; continue
        n_in = len(ck["feature_lists"][c_str])
        models_c = []
        for sd in sd_list:
            m = Refiner(n_in).to(device); m.load_state_dict(sd); m.eval()
            models_c.append(m)
        models[c] = models_c
    return (models, stats, ck["best_alpha"],
            ck.get("best_mode", "geom"),
            ck.get("disable_classes", []),
            ck["baseline_val_map"], ck["best_val_map"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-preds", required=True,
                    help="pt file from run_ensemble_scenes on cache_test_full")
    ap.add_argument("--test-feat", required=True,
                    help="pt file from extract_pred_features on test preds")
    ap.add_argument("--test-cache", default=str(ARTIFACTS / "cache_test_full"))
    ap.add_argument("--refiner-ckpt", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="e.g. submission_refined")
    ap.add_argument("--alpha", type=float, default=None,
                    help="if omitted, use best_alpha from refiner ckpt")
    a = ap.parse_args()

    print(f"loading test preds {a.test_preds}", flush=True)
    blob = torch.load(a.test_preds, weights_only=False)
    preds_raw = blob["preds"]
    print(f"  {len(preds_raw)} frames", flush=True)

    print(f"loading test features {a.test_feat}", flush=True)
    ff = torch.load(a.test_feat, weights_only=False)
    feat_rows = ff["pred_rows"]

    print(f"loading refiner {a.refiner_ckpt}", flush=True)
    (models, stats, best_alpha, best_mode, disable_classes,
     base_map, best_map) = load_refiner_ckpt(a.refiner_ckpt)
    alpha = a.alpha if a.alpha is not None else best_alpha
    mode = best_mode
    print(f"  baseline val {base_map:.4f}, best val {best_map:.4f}, "
          f"using mode {mode} alpha {alpha} disabled {disable_classes}",
          flush=True)

    # Frame paths (deterministic order = sorted glob same as run_ensemble_scenes)
    test_cache = Path(a.test_cache)
    frame_paths = sorted(test_cache.glob("*.npz"))
    assert len(frame_paths) == len(preds_raw), \
        f"mismatch: {len(frame_paths)} paths vs {len(preds_raw)} preds"

    # Precompute per-class refined score + keep mask for each row
    refined_by_index = np.zeros(len(feat_rows))
    keep_by_index    = np.ones(len(feat_rows), dtype=bool)
    for c in (1, 2, 3):
        rows_c = [(i, r) for i, r in enumerate(feat_rows) if r["cls"] == c]
        if not rows_c or models[c] is None or c in disable_classes:
            for i, r in rows_c:
                refined_by_index[i] = r["score"]
            continue
        X = np.array([[r[f] for f in FEATURES[c]] for _, r in rows_c],
                     dtype=np.float32)
        X = transform(X, c).astype(np.float32)
        m_stat, s_stat = stats[c]
        X = ((X - m_stat) / s_stat).astype(np.float32)
        p = predict_prob(models[c], X)
        for (i, r), pp in zip(rows_c, p):
            base = max(r["score"], 1e-4)
            pp = float(pp)
            if mode == "geom":
                refined_by_index[i] = (base ** (1 - alpha)) * (pp ** alpha)
            elif mode == "addit":
                refined_by_index[i] = (1 - alpha) * base + alpha * pp
            elif mode == "product":
                refined_by_index[i] = base * (pp ** alpha)
            elif mode == "hardfilt":
                refined_by_index[i] = base
                keep_by_index[i] = (pp >= alpha)
            else:
                raise ValueError(f"unknown mode {mode}")

    # Write per-frame files
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.txt"):
        f.unlink()  # clean previous
    print(f"writing to {out_dir}...", flush=True)

    fi_ptr = 0
    total_boxes = 0
    for fi, (raw, fp) in enumerate(zip(preds_raw, frame_paths)):
        d = cross_class_nms(raw, iou_thresh=0.30)
        order = d["scores"].argsort(descending=True)
        d = {k: v[order] for k, v in d.items()}
        pb = d["boxes"].numpy(); pl = d["labels"].numpy()
        n = len(pl)
        new_scores = np.array([refined_by_index[fi_ptr + k] for k in range(n)],
                              dtype=np.float32)
        keep = np.array([keep_by_index[fi_ptr + k] for k in range(n)],
                        dtype=bool)
        fi_ptr += n
        pb = pb[keep]; new_scores = new_scores[keep]; pl = pl[keep]
        n = len(pl)
        srt = np.argsort(-new_scores)
        pb = pb[srt]; new_scores = new_scores[srt]; pl = pl[srt]

        lines = []
        for bi in range(n):
            cx, cy, cz, dx, dy, dz, yaw = pb[bi]
            yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
            yaw = float(np.clip(yaw, -3.1415, 3.1415))
            lines.append(
                f"{int(pl[bi])} {cx:.4f} {cy:.4f} {cz:.4f} "
                f"{dx:.4f} {dy:.4f} {dz:.4f} {yaw:.4f} {float(new_scores[bi]):.4f}"
            )
        (out_dir / f"{frame_id(fp.name)}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        total_boxes += n
    print(f"  wrote {len(preds_raw)} frames ({total_boxes} boxes)",
          flush=True)

    # zip
    zip_path = str(out_dir) + ".zip"
    if Path(zip_path).exists():
        Path(zip_path).unlink()
    subprocess.check_call(
        ["zip", "-q", "-r", zip_path, "."], cwd=out_dir)
    print(f"zip -> {zip_path}", flush=True)


if __name__ == "__main__":
    main()
