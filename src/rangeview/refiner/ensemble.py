"""Try more sophisticated deploy strategies to squeeze extra val mAP:

  A. Per-class hard-filter threshold sweep (independent per class)
  B. Finer threshold grid (0.02..0.15 step 0.01)
  C. MLP + GBM probability ensemble (average sigmoid outputs before threshold)
  D. Combinations of the above

Uses saved refiner_ckpts.pt (MLP) and refiner_gbm.pt (GBM).  All experiments
are score_frames on the CACHED val predictions -- fast.
"""
import argparse
import pickle
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch

from rangeview.models.decode import cross_class_nms
import rangeview.eval.score_exact as S
from rangeview.paths import ARTIFACTS
from rangeview.refiner.refiner import FEATURES, transform, Refiner, predict_prob

CLASS_NAME = {1: "Vehicle", 2: "Bike", 3: "Pedestrian"}


def load_mlp_ckpt(path):
    ck = torch.load(path, weights_only=False)
    stats = {int(c): (np.array(v[0], dtype=np.float32),
                      np.array(v[1], dtype=np.float32))
             for c, v in ck["stats"].items()}
    models = {}
    for c_str, sd_list in ck["state_dicts"].items():
        c = int(c_str)
        if sd_list is None:
            models[c] = None; continue
        n_in = len(ck["feature_lists"][c_str])
        seed_models = []
        for sd in sd_list:
            m = Refiner(n_in); m.load_state_dict(sd); m.eval()
            seed_models.append(m)
        models[c] = seed_models
    return models, stats, ck.get("disable_classes", [])


def load_gbm_ckpt(path):
    ck = torch.load(path, weights_only=False)
    stats = {int(c): (np.array(v[0], dtype=np.float32),
                      np.array(v[1], dtype=np.float32))
             for c, v in ck["stats"].items()}
    gbm_by_class = pickle.loads(ck["gbm_pickled"])
    gbm_by_class = {int(c): v for c, v in gbm_by_class.items()}
    return gbm_by_class, stats, ck.get("disable_classes", [])


def refiner_probs_mlp(feat_rows, mlp_models, mlp_stats):
    """Compute MLP refiner prob for every row in feat_rows.  Rows where models[c]
    is None or c in disable get prob = row['score'] (fallback)."""
    probs = np.zeros(len(feat_rows))
    for c in (1, 2, 3):
        rows_c = [(i, r) for i, r in enumerate(feat_rows) if r["cls"] == c]
        if not rows_c:
            continue
        if mlp_models[c] is None:
            for i, r in rows_c: probs[i] = r["score"]
            continue
        X = np.array([[r[f] for f in FEATURES[c]] for _, r in rows_c],
                     dtype=np.float32)
        X = transform(X, c).astype(np.float32)
        m_stat, s_stat = mlp_stats[c]
        X = ((X - m_stat) / s_stat).astype(np.float32)
        p = predict_prob(mlp_models[c], X)
        for (i, r), pp in zip(rows_c, p):
            probs[i] = float(pp)
    return probs


def refiner_probs_gbm(feat_rows, gbm_by_class, gbm_stats):
    probs = np.zeros(len(feat_rows))
    for c in (1, 2, 3):
        rows_c = [(i, r) for i, r in enumerate(feat_rows) if r["cls"] == c]
        if not rows_c:
            continue
        if gbm_by_class.get(c) is None:
            for i, r in rows_c: probs[i] = r["score"]
            continue
        X = np.array([[r[f] for f in FEATURES[c]] for _, r in rows_c],
                     dtype=np.float32)
        X = transform(X, c).astype(np.float32)
        m_stat, s_stat = gbm_stats[c]
        X = ((X - m_stat) / s_stat).astype(np.float32)
        gbms = gbm_by_class[c]
        p_sum = np.zeros(len(X))
        for gbm in gbms:
            p_sum = p_sum + gbm.predict_proba(X)[:, 1]
        p = p_sum / len(gbms)
        for (i, r), pp in zip(rows_c, p):
            probs[i] = float(pp)
    return probs


def score_with_probs(preds_raw, gts, feat_rows, probs, thresholds_by_class):
    """probs[i] is the per-row refiner probability. thresholds_by_class[c] is the
    per-class hard-filter threshold; rows dropped if probs[i] < thr[c] (only for
    classes where refiner active)."""
    keep = np.ones(len(feat_rows), dtype=bool)
    for i, r in enumerate(feat_rows):
        c = r["cls"]
        thr_c = thresholds_by_class.get(c)
        if thr_c is None:
            continue
        if probs[i] < thr_c:
            keep[i] = False
    frames = []; fi_ptr = 0
    for fi, raw in enumerate(preds_raw):
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
        gb, gl = gts[fi]
        frames.append((pb, ps, pl, gb, gl))
    mAP, head, _ = S.score_frames(frames, thresholds=(0.3, 0.5))
    return float(mAP), head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-feat", default=str(ARTIFACTS / "val_features_v2.pt"))
    ap.add_argument("--val-preds", default=str(ARTIFACTS / "val_preds_cache.pt"))
    ap.add_argument("--mlp-ckpt", default=str(ARTIFACTS / "refiner_ckpts.pt"))
    ap.add_argument("--gbm-ckpt", default=str(ARTIFACTS / "refiner_gbm.pt"))
    a = ap.parse_args()

    v = torch.load(a.val_feat, weights_only=False)
    val_rows = v["pred_rows"]
    blob = torch.load(a.val_preds, weights_only=False)
    preds_raw = blob["preds"]; gts = blob["gts"]

    mlp_models, mlp_stats, mlp_disable = load_mlp_ckpt(a.mlp_ckpt)
    gbm_models, gbm_stats, gbm_disable = load_gbm_ckpt(a.gbm_ckpt)
    print(f"MLP disabled classes: {mlp_disable}   GBM disabled: {gbm_disable}",
          flush=True)

    # ---- compute per-row probs ----
    print("computing MLP probs...", flush=True)
    p_mlp = refiner_probs_mlp(val_rows, mlp_models, mlp_stats)
    print("computing GBM probs...", flush=True)
    p_gbm = refiner_probs_gbm(val_rows, gbm_models, gbm_stats)
    # ensemble mean
    p_mean = (p_mlp + p_gbm) / 2

    # ---- Baseline (no filter) ----
    m0, h0 = score_with_probs(preds_raw, gts, val_rows, p_mlp, {})
    print(f"\nbaseline: mAP {m0:.4f}  V {h0[1]:.4f} B {h0[2]:.4f} P {h0[3]:.4f}",
          flush=True)

    # ---- Global threshold sweep for MLP-alone (matches earlier) ----
    print("\n=== A: MLP-alone global threshold (finer grid) ===", flush=True)
    thr_grid_fine = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                     0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    for t in thr_grid_fine:
        # Vehicle + Bike only (Ped disabled)
        m, h = score_with_probs(preds_raw, gts, val_rows, p_mlp,
                                 {1: t, 2: t})
        tag = " *" if m > m0 + 0.001 else ""
        print(f"  thr {t:.3f}: mAP {m:.4f} (Δ {m-m0:+.4f})  "
              f"V {h[1]:.4f} B {h[2]:.4f} P {h[3]:.4f}{tag}", flush=True)

    # ---- B: per-class threshold search (MLP) ----
    print("\n=== B: MLP per-class thresholds (V × B grid) ===", flush=True)
    thr_v_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    thr_b_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    best_m = m0; best_v = None; best_b = None
    for tv, tb in product(thr_v_grid, thr_b_grid):
        m, h = score_with_probs(preds_raw, gts, val_rows, p_mlp,
                                 {1: tv, 2: tb})
        if m > best_m + 1e-5:
            best_m = m; best_v = tv; best_b = tb
    print(f"  MLP best: V thr={best_v} B thr={best_b}  "
          f"mAP {best_m:.4f} (Δ {best_m-m0:+.4f})", flush=True)

    # ---- C: GBM per-class thresholds ----
    print("\n=== C: GBM per-class thresholds ===", flush=True)
    best_m_g = m0; best_vg = None; best_bg = None
    for tv, tb in product(thr_v_grid, thr_b_grid):
        m, h = score_with_probs(preds_raw, gts, val_rows, p_gbm,
                                 {1: tv, 2: tb})
        if m > best_m_g + 1e-5:
            best_m_g = m; best_vg = tv; best_bg = tb
    print(f"  GBM best: V thr={best_vg} B thr={best_bg}  "
          f"mAP {best_m_g:.4f} (Δ {best_m_g-m0:+.4f})", flush=True)

    # ---- D: MLP+GBM mean probs, per-class thresholds ----
    print("\n=== D: MLP+GBM mean probs, per-class thresholds ===", flush=True)
    best_m_e = m0; best_ve = None; best_be = None
    for tv, tb in product(thr_v_grid, thr_b_grid):
        m, h = score_with_probs(preds_raw, gts, val_rows, p_mean,
                                 {1: tv, 2: tb})
        if m > best_m_e + 1e-5:
            best_m_e = m; best_ve = tv; best_be = tb
    print(f"  Ensemble best: V thr={best_ve} B thr={best_be}  "
          f"mAP {best_m_e:.4f} (Δ {best_m_e-m0:+.4f})", flush=True)

    # ---- E: mixed source per class (MLP for one class, GBM for other) ----
    print("\n=== E: mixed refiner source per class ===", flush=True)
    for vf, vname, bf, bname in [
        (p_mlp, "MLP", p_mlp, "MLP"),
        (p_mlp, "MLP", p_gbm, "GBM"),
        (p_gbm, "GBM", p_mlp, "MLP"),
        (p_gbm, "GBM", p_gbm, "GBM"),
        (p_mean, "ENS", p_mean, "ENS"),
        (p_mlp, "MLP", p_mean, "ENS"),
        (p_mean, "ENS", p_mlp, "MLP"),
    ]:
        best_m2 = m0; best_v2 = None; best_b2 = None
        for tv, tb in product(thr_v_grid, thr_b_grid):
            # apply different probs per class -- score_with_probs uses one probs
            # array, so we splice: use vf for Vehicle rows, bf for Bike rows.
            probs = np.array(p_mlp)  # copy shape, ignored where not V/B
            for i, r in enumerate(val_rows):
                if r["cls"] == 1: probs[i] = vf[i]
                elif r["cls"] == 2: probs[i] = bf[i]
            m, h = score_with_probs(preds_raw, gts, val_rows, probs,
                                     {1: tv, 2: tb})
            if m > best_m2 + 1e-5:
                best_m2 = m; best_v2 = tv; best_b2 = tb
        print(f"  V={vname} B={bname}: V thr={best_v2} B thr={best_b2}  "
              f"mAP {best_m2:.4f} (Δ {best_m2-m0:+.4f})", flush=True)

    # save the best config as a JSON summary
    import json
    summary = {
        "baseline_map": m0,
        "mlp_perclass_best": {"v_thr": best_v, "b_thr": best_b, "map": best_m},
        "gbm_perclass_best": {"v_thr": best_vg, "b_thr": best_bg, "map": best_m_g},
        "ensemble_perclass_best": {"v_thr": best_ve, "b_thr": best_be, "map": best_m_e},
    }
    with open(ARTIFACTS / "ensemble_refiners_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> ensemble_refiners_result.json", flush=True)


if __name__ == "__main__":
    main()
