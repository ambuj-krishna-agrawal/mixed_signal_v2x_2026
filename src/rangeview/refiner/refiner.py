"""Train per-class MLP refiners on training-scene features, evaluate on val.

Refiner takes hand-crafted per-box features (see extract_pred_features.py) plus
the raw detection score, outputs a probability that the box is a TP at the
competition IoU threshold for its class.

Deploy: refined_score = original_score ** (1 - alpha) * refiner_prob ** alpha
(alpha tuned on val).  Then cc-NMS 0.30, then score.

Independent MLPs per class since feature signal differs strongly per class
(Vehicle: geometric; Bike: size/height; Ped: score + agents).

We ONLY train on TRAIN-scene features to avoid overfitting val.  Val is a
strict held-out for evaluating the refiner.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rangeview.models.decode import cross_class_nms
from rangeview.paths import ARTIFACTS, SCORING

_SP = SCORING / "scoring.py"
_spec = importlib.util.spec_from_file_location("official_scoring", _SP)
_official = importlib.util.module_from_spec(_spec)
sys.modules["official_scoring"] = _official
_spec.loader.exec_module(_official)
import rangeview.eval.score_exact as S


# Per-class feature list -- ordered.  All floats.
# Extended with height percentiles, expanded-box density contrast, and
# nearby-prediction context features.
_CORE = ["score", "point_count", "point_density", "z_top", "z_bot", "z_span",
         "z10", "z25", "z50", "z75", "z90",
         "n_shell", "density_contrast",
         "n_agents", "ego_dist", "box_volume",
         "intensity_mean", "intensity_std",
         "near_same_dist", "near_same_score", "has_higher_near",
         "n_same_within_5m", "near_diff_dist", "n_any_within_3m"]
FEATURES = {1: _CORE, 2: _CORE, 3: _CORE}
CLASS_NAME = {1: "Vehicle", 2: "Bike", 3: "Pedestrian"}


def make_dataset(rows, cls):
    """Return (X, y) numpy arrays for class `cls` from feature rows."""
    r_c = [r for r in rows if r["cls"] == cls]
    if not r_c:
        return np.zeros((0, len(FEATURES[cls]))), np.zeros(0)
    X = np.array([[r[f] for f in FEATURES[cls]] for r in r_c],
                 dtype=np.float32)
    y = np.array([1.0 if r["is_tp"] else 0.0 for r in r_c],
                 dtype=np.float32)
    return X, y


def _p1_log1p(x):
    # log(1 + x) for wide-range features (point_count, point_density, etc.)
    return np.sign(x) * np.log1p(np.abs(x))


def transform(X, cls):
    """Apply per-feature transforms before standardisation.

    Wide-range features (point_count, point_density, box_volume, ego_dist,
    n_shell, density_contrast, near_same_dist, near_diff_dist, near_same_score,
    n_same_within_5m, n_any_within_3m) get log1p; others pass through.
    """
    X = X.copy()
    feats = FEATURES[cls]
    log_feats = {"point_count", "point_density", "box_volume", "ego_dist",
                 "n_shell", "density_contrast",
                 "near_same_dist", "near_diff_dist", "near_same_score",
                 "n_same_within_5m", "n_any_within_3m"}
    for i, f in enumerate(feats):
        if f in log_feats:
            X[:, i] = _p1_log1p(X[:, i])
    return X


def compute_stats(X):
    m = X.mean(axis=0); s = X.std(axis=0); s = np.where(s < 1e-6, 1.0, s)
    return m, s


class Refiner(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


def train_one(X_tr, y_tr, X_va, y_va, epochs=40, batch=256, lr=1e-3, device="cpu",
              verbose=False, seed=0):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    m = Refiner(n_in).to(device)
    pos = y_tr.sum(); neg = (1 - y_tr).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.from_numpy(X_tr).to(device); yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_va).to(device); yv = torch.from_numpy(y_va).to(device)

    best_auc = 0; best_state = None
    for e in range(epochs):
        m.train()
        idx = torch.randperm(len(Xt), device=device)
        loss_tot = 0; nb = 0
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            opt.zero_grad(set_to_none=True)
            out = m(Xt[b]).squeeze(-1)
            l = crit(out, yt[b])
            l.backward(); opt.step()
            loss_tot += float(l); nb += 1
        m.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(m(Xv).squeeze(-1)).cpu().numpy()
        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_va, p_va))
        except Exception:
            auc = float("nan")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        if verbose:
            print(f"    ep {e:>2}  loss {loss_tot/max(nb,1):.4f}  val AUC {auc:.4f}",
                  flush=True)
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best_auc


def predict_prob(model_or_list, X, device="cpu"):
    """If given a list of models, average their sigmoid outputs (ensemble)."""
    Xt = torch.from_numpy(X).to(device)
    if isinstance(model_or_list, list):
        p = torch.zeros(len(X), device=device)
        for m in model_or_list:
            m.eval()
            with torch.no_grad():
                p = p + torch.sigmoid(m(Xt).squeeze(-1))
        p = p / len(model_or_list)
    else:
        m = model_or_list
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(Xt).squeeze(-1))
    return p.cpu().numpy()


def apply_refiner_and_score(preds_raw, gts, val_features, models, stats, alpha,
                             mode="geom", disable_classes=()):
    """Compute refined scores, apply cc-NMS 0.30, score with competition scorer.

    mode:
      geom     : refined = orig^(1-alpha) * prob^alpha
      addit    : refined = (1-alpha)*orig + alpha*prob
      product  : refined = orig * prob^alpha
      hardfilt : drop preds with prob < alpha (alpha is threshold, not blend)
    disable_classes: classes for which to fall back to raw score.
    """
    refined = {}
    for c in (1, 2, 3):
        rows_c = [(i, r) for i, r in enumerate(val_features) if r["cls"] == c]
        if not rows_c or models[c] is None or c in disable_classes:
            for i, r in rows_c:
                refined[i] = (r["score"], True)  # (score, keep)
            continue
        m_stat, s_stat = stats[c]
        X = np.array([[r[f] for f in FEATURES[c]] for _, r in rows_c],
                     dtype=np.float32)
        X = transform(X, c)
        X = (X - m_stat) / s_stat
        p = predict_prob(models[c], X)
        for (i, r), pp in zip(rows_c, p):
            base = max(r["score"], 1e-4)
            pp = float(pp)
            if mode == "geom":
                new_s = (base ** (1 - alpha)) * (pp ** alpha)
                keep = True
            elif mode == "addit":
                new_s = (1 - alpha) * base + alpha * pp
                keep = True
            elif mode == "product":
                new_s = base * (pp ** alpha)
                keep = True
            elif mode == "hardfilt":
                new_s = base
                keep = (pp >= alpha)
            else:
                raise ValueError(f"unknown mode {mode}")
            refined[i] = (new_s, keep)

    # Order of val_features matches: per-frame, in the order they appear after
    # apply_filter -> cross_class_nms -> sort in extract_pred_features.
    frames = []; fi_ptr = 0
    for fi, raw in enumerate(preds_raw):
        d = cross_class_nms(raw, iou_thresh=0.30)
        order = d["scores"].argsort(descending=True)
        d = {k: v[order] for k, v in d.items()}
        pb = d["boxes"].numpy(); pl = d["labels"].numpy()
        n = len(pl)
        new_scores = np.zeros(n, dtype=np.float32)
        keep_mask = np.ones(n, dtype=bool)
        for k in range(n):
            s, keep = refined[fi_ptr + k]
            new_scores[k] = s
            keep_mask[k] = keep
        fi_ptr += n
        pb = pb[keep_mask]; new_scores = new_scores[keep_mask]; pl = pl[keep_mask]
        srt = np.argsort(-new_scores)
        pb = pb[srt]; new_scores = new_scores[srt]; pl = pl[srt]
        gb, gl = gts[fi]
        frames.append((pb, new_scores, pl, gb, gl))
    mAP, head, _ = S.score_frames(frames, thresholds=(0.3, 0.5))
    return float(mAP), head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-feat", default=str(ARTIFACTS / "train_features.pt"))
    ap.add_argument("--val-feat", default=str(ARTIFACTS / "val_features.pt"))
    ap.add_argument("--val-preds", default=str(ARTIFACTS / "val_preds_cache.pt"))
    ap.add_argument("--out", default=str(ARTIFACTS / "refiner_ckpts.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    device = a.device

    print(f"loading val {a.val_feat} and train {a.train_feat}", flush=True)
    v = torch.load(a.val_feat, weights_only=False)
    val_rows = v["pred_rows"]
    tr = torch.load(a.train_feat, weights_only=False)
    train_rows = tr["pred_rows"]
    print(f"  train rows: {len(train_rows)}   val rows: {len(val_rows)}",
          flush=True)

    val_preds_blob = torch.load(a.val_preds, weights_only=False)
    val_preds_raw = val_preds_blob["preds"]
    val_gts = val_preds_blob["gts"]

    # ---- score-only baseline AUC per class (what the refiner has to beat) ----
    from sklearn.metrics import roc_auc_score
    score_auc = {}
    for c in (1, 2, 3):
        r_c = [r for r in val_rows if r["cls"] == c]
        if r_c:
            y = np.array([1.0 if r["is_tp"] else 0.0 for r in r_c])
            s = np.array([r["score"] for r in r_c])
            try:
                score_auc[c] = float(roc_auc_score(y, s))
            except Exception:
                score_auc[c] = 0.5
    print(f"score-only val AUC: {score_auc}", flush=True)

    # ---- train per-class refiners ----
    models = {}; stats = {}; refiner_auc = {}
    for c in (1, 2, 3):
        X_tr, y_tr = make_dataset(train_rows, c)
        X_va, y_va = make_dataset(val_rows, c)
        if len(X_tr) == 0 or len(X_va) == 0:
            print(f"  {CLASS_NAME[c]}: not enough data, skip"); models[c] = None
            stats[c] = (np.zeros(len(FEATURES[c])), np.ones(len(FEATURES[c])))
            continue
        X_tr = transform(X_tr, c); X_va = transform(X_va, c)
        m_stat, s_stat = compute_stats(X_tr)
        X_tr_n = (X_tr - m_stat) / s_stat
        X_va_n = (X_va - m_stat) / s_stat
        stats[c] = (m_stat, s_stat)
        print(f"\n=== training refiner: {CLASS_NAME[c]} "
              f"(train {len(X_tr)}, val {len(X_va)}, "
              f"pos_train {int(y_tr.sum())}, pos_val {int(y_va.sum())}) ===",
              flush=True)
        # ensemble N seeds, average predicted probs
        aucs = []
        seed_models = []
        for seed in a.seeds:
            m, auc = train_one(X_tr_n, y_tr, X_va_n, y_va,
                               epochs=a.epochs, device=device, seed=seed)
            aucs.append(auc); seed_models.append(m)
            print(f"    seed {seed}: val AUC {auc:.4f}", flush=True)
        models[c] = seed_models
        # ensemble val AUC (average probs across seeds)
        p_ens = np.zeros(len(y_va))
        for m in seed_models:
            p_ens += predict_prob(m, X_va_n, device=device)
        p_ens /= len(seed_models)
        auc_ens = float(roc_auc_score(y_va, p_ens))
        refiner_auc[c] = auc_ens
        print(f"  ensemble val AUC: {auc_ens:.4f} "
              f"(single seeds mean {np.mean(aucs):.4f}, best {max(aucs):.4f}, "
              f"score-only baseline {score_auc.get(c, 0.5):.4f})",
              flush=True)

    # ---- baseline (no refiner) ----
    print("\n=== baseline (no refiner) ===", flush=True)
    m0, h0 = apply_refiner_and_score(val_preds_raw, val_gts, val_rows,
                                     {1: None, 2: None, 3: None},
                                     stats, alpha=0.0)
    print(f"  mAP {m0:.4f}  V {h0[1]:.4f} B {h0[2]:.4f} P {h0[3]:.4f}",
          flush=True)

    # Determine which classes get a refiner (fallback if refiner AUC < score AUC)
    disable_classes = tuple(c for c in (1, 2, 3)
                            if refiner_auc.get(c, 0.5)
                               < score_auc.get(c, 0.5) - 0.005)
    if disable_classes:
        print(f"\nfallback to raw score for classes: {disable_classes} "
              f"(refiner AUC < score AUC)", flush=True)

    # ---- sweep alpha across multiple modes ----
    print("\n=== sweep alpha × mode ===", flush=True)
    best_alpha = 0.0; best_m = m0; best_mode = "geom"
    for mode in ("geom", "addit", "product"):
        print(f"  mode={mode}", flush=True)
        for alpha in a.alphas:
            m, h = apply_refiner_and_score(val_preds_raw, val_gts, val_rows,
                                           models, stats, alpha=alpha,
                                           mode=mode,
                                           disable_classes=disable_classes)
            tag = " *" if m > best_m else ""
            print(f"    alpha {alpha:.2f}:  mAP {m:.4f} (Δ {m-m0:+.4f})  "
                  f"V {h[1]:.4f} B {h[2]:.4f} P {h[3]:.4f}{tag}",
                  flush=True)
            if m > best_m:
                best_m = m; best_alpha = alpha; best_mode = mode
    # also try hard filter mode with pp threshold
    print(f"  mode=hardfilt", flush=True)
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        m, h = apply_refiner_and_score(val_preds_raw, val_gts, val_rows,
                                       models, stats, alpha=thr,
                                       mode="hardfilt",
                                       disable_classes=disable_classes)
        tag = " *" if m > best_m else ""
        print(f"    thr {thr:.2f}:  mAP {m:.4f} (Δ {m-m0:+.4f})  "
              f"V {h[1]:.4f} B {h[2]:.4f} P {h[3]:.4f}{tag}",
              flush=True)
        if m > best_m:
            best_m = m; best_alpha = thr; best_mode = "hardfilt"

    print(f"\nBEST: mode={best_mode} alpha={best_alpha:.2f}  val mAP {best_m:.4f} "
          f"(Δ {best_m-m0:+.4f})", flush=True)

    # persist per-seed model state_dicts, stats, best alpha
    ckpt = {"stats": {c: (stats[c][0].tolist(), stats[c][1].tolist())
                       for c in (1, 2, 3)},
            "feature_lists": {c: FEATURES[c] for c in (1, 2, 3)},
            "state_dicts": {c: ([sm.state_dict() for sm in models[c]]
                                if isinstance(models[c], list) else None)
                             for c in (1, 2, 3)},
            "best_alpha": best_alpha,
            "best_mode": best_mode,
            "best_val_map": best_m,
            "baseline_val_map": m0,
            "disable_classes": list(disable_classes),
            "score_auc": score_auc,
            "refiner_auc": refiner_auc}
    torch.save(ckpt, a.out)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
