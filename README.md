# RangeView V2X — Mixed Signals ICCV 2025

Range-view fusion 3D object detector for the [Mixed Signals V2X ICCV 2025 challenge](https://mixed-signals.iccv2025.org).
Detects Vehicle / Bike / Pedestrian from five LiDAR agents (2 static RSU + 3 vehicles), all in the top-RSU ego frame.

**Scoring metric:** `mAP = mean(Vehicle AP@0.5, Bike AP@0.5, Pedestrian AP@0.3)` (frame-wise PR, no global sort — see `src/rangeview/eval/evaluate.py`).

Best test-set mAP achieved: **0.8524** (V=0.8646, B=0.8198, P=0.8727) via per-class hybrid.

---

## Setup

```bash
uv sync
```

Requires Python 3.12+.  All dependencies are pinned in `uv.lock`.

## Repository layout

```
.
├── pyproject.toml           # uv-managed, package = "rangeview"
├── uv.lock
├── README.md
├── src/rangeview/           # main package
│   ├── compat.py            # MPS / CUDA compat shim
│   ├── data/                # dataset, augmentation, geometry, cache builders
│   │   ├── dataset.py
│   │   ├── augment.py
│   │   ├── geometry.py
│   │   └── caches/          # entry points: raw → cache
│   ├── models/              # network + primitives
│   │   ├── net.py           # RangeViewFusionNet
│   │   ├── decode.py        # box decoding + rotated NMS
│   │   ├── loss.py
│   │   ├── norm.py          # GroupNorm subclass (autocast-safe)
│   │   ├── ema.py
│   │   ├── agent_fusion.py
│   │   └── meta_kernel.py
│   ├── training/train.py
│   ├── inference/
│   │   ├── infer.py         # forward → cached .pt of predictions
│   │   ├── features.py      # per-box features for the refiner
│   │   └── tta.py           # weighted box fusion (WBF)
│   ├── refiner/
│   │   ├── refiner.py       # MLP false-positive filter (train + predict)
│   │   ├── ensemble.py      # multi-checkpoint refiner loader
│   │   └── util.py
│   ├── submission/
│   │   ├── build.py         # refiner + RSU filter + zip
│   │   ├── filter.py        # Bike ground / dim filter
│   │   ├── variant.py       # cross-model WBF + P expansion (main variant)
│   │   └── hybrid.py        # per-class swap ensembler
│   └── eval/
│       ├── evaluate.py      # official scorer (drop-in for the training loop)
│       ├── score_ckpt.py    # standalone CLI: score a checkpoint on val
│       └── score_exact.py   # imports official scoring_program/scoring.py
├── scripts/                 # thin shell wrappers around `uv run python -m ...`
│   ├── train.sh
│   ├── infer.sh
│   ├── evaluate.sh
│   └── build_submission.sh
├── artifacts/               # trained checkpoints + caches + refiners
│   ├── checkpoints/         # main models
│   ├── checkpoints_v2/      # continued-training checkpoint
│   ├── cache_train/         # RangeViewDataset cache — training
│   ├── cache_test_full/     # RangeViewDataset cache — test
│   ├── ground_map.npz       # static ground surface (built once)
│   ├── sensor_origins*.npz  # per-frame sensor positions
│   ├── refiner_ckpts.pt     # Bike refiner
│   ├── refiner_ckpts_2x.pt  # Vehicle refiner
│   └── test_preds_*.pt      # cached forward passes at 1024 / 512 azimuth
├── data_raw/                # raw challenge data (external, not tracked)
├── devkit_sources/          # challenge devkit
├── scoring_program/         # Codabench-provided official scorer
└── experiments/             # everything discarded — see experiments_done.md
```

---

## Training

Trains a range-view fusion network (GroupNorm throughout — BN destroys inference here, see notes).

```bash
scripts/train.sh \
  --epochs 14 --optimizer adamw --lr 7e-4 --lr-decay 0.8 --warmup-epochs 1 \
  --no-vfe --no-iou-head --no-residuals
```

This is the recipe behind the **0.8424** single-model test score.  Val is held out on scenes 31 + 34.

## Inference

Forward a trained checkpoint over a cache to a `.pt` of raw predictions (boxes, scores, labels per frame):

```bash
# Test cache uses UUID filenames instead of seqN_idxM, so pass --val-scenes -1
# to disable scene filtering.  Y-flip TTA is on by default (add --no-yflip to skip).
scripts/infer.sh \
  --ckpt  artifacts/checkpoints/rangeview_epoch7.pth \
  --cache artifacts/cache_test_full \
  --val-scenes -1 \
  --out   artifacts/test_preds_ep7.pt

# Extract per-box features that the refiner reads
uv run python -m rangeview.inference.features \
  --pred-cache  artifacts/test_preds_ep7.pt \
  --frame-cache artifacts/cache_test_full \
  --out         artifacts/test_features_ep7.pt
```

Repeat `scripts/infer.sh` for each checkpoint whose predictions you want to fuse.  For the winning variant three checkpoints were used (`v2`, `ep8`, `ep8cp`) at both 1024 and 512 azimuth resolution.

## Evaluation

Runs the OFFICIAL frame-wise scorer.  Do not trust any evaluator that sorts detections globally — see the notes on the two broken evaluators in `experiments/experiments_done.md`.

```bash
scripts/evaluate.sh \
  --ckpt  artifacts/checkpoints/rangeview_epoch7.pth \
  --cache artifacts/cache_train \
  --val-scenes 31 34
```

Under the hood this runs `python -m rangeview.eval.score_ckpt`, which loads the
checkpoint and forwards through `rangeview.eval.evaluate.evaluate_official` — the
same scorer the training loop uses for per-epoch checkpoint selection.

## Submission

**End-to-end winning variant** (cross-model WBF + mild Ped expansion + Ped>Bike NMS + refiners + RSU filter + Bike ground-dim filter):

```bash
scripts/build_submission.sh \
  --preds-1024 artifacts/test_preds_v2.pt \
               artifacts/test_preds_ep8.pt \
               artifacts/test_preds_ep8cp.pt \
  --preds-512  artifacts/test_preds_v2_512.pt \
               artifacts/test_preds_ep8_512.pt \
               artifacts/test_preds_ep8cp_512.pt \
  --test-cache artifacts/cache_test_full \
  --v-refiner  artifacts/refiner_ckpts_2x.pt \
  --b-refiner  artifacts/refiner_ckpts.pt \
  --out-dir    submission_out \
  --work-dir   artifacts
```

**Per-class hybrid** (deterministic — swap Vehicle from one zip, Bike from another, Pedestrian from a third; the scoring metric is per-class independent so each class's measured AP is preserved exactly):

```bash
uv run python -m rangeview.submission.hybrid \
  --v-source ~/Downloads/best_vehicle_submission.zip \
  --b-source ~/Downloads/best_bike_submission.zip \
  --p-source ~/Downloads/best_pedestrian_submission.zip \
  --out      submission_hybrid.zip
```

---

## Key design notes

- **BatchNorm destroys inference**: one shared encoder runs per-agent at batch 2.  BN normalises each agent by its own statistics in training and by one global average at eval.  Same weights, same batch: off loss 0.0011 → 0.0141.  Use GroupNorm.  See `src/rangeview/models/norm.py`.
- **Test data has no odometry**: residual (temporal-difference) channels are impossible on test.  Any checkpoint trained with `--residuals` cannot ship.  Only residual-free models work.
- **The score-metric is per-frame PR**: monotone rescaling of scores (Platt, isotonic, temperature) is a mathematical no-op.  Do not spend time on it.
- **Ped>Bike NMS is a class-priority rule**, not a duplicate-removal step: annotators tag mixed pedestrian+bike scenes as pedestrians, so a Bike prediction overlapping a Pedestrian prediction is almost always a false positive.

For everything ruled out (multi-model ensembling that hurt, TTA that helped only Y-flip, PillarVFE, per-frame count heads, learned FP filters), see `experiments/experiments_done.md`.

## Data flow (winning submission)

```
raw scenes ─► build_train_cache ─► cache_train/  ─┐
                                                  ├─► train.py ─► checkpoints/*.pth
raw scenes ─► build_ground_map   ─► ground_map.npz┘
raw scenes ─► build_sensor_origins ─► sensor_origins.npz

test scenes ─► build_test_cache ─► cache_test_full/ ─┐
                                                     │
checkpoint ─► inference.infer ─► test_preds_*.pt ────┤
                                                     │
                                                     ├─► submission.variant
train pred+features ─► refiner.refiner train ─► refiner_ckpts_*.pt  ─┤    (cross-model WBF +
                                                                     │     mild P expansion +
                                                                     │     Ped>Bike NMS +
                                                                     │     refiner + RSU +
                                                                     │     Bike ground-dim)
                                                                     ▼
                                                              submission.zip
```

## What lives in `experiments/`

- `experiments_done.md` — full session log (renamed from HANDOFF.md), covering the 0.5959 → 0.7903 → 0.8424 → 0.8524 progression + every ruled-out idea
- `FINDINGS.md` — cross-cut notes on val→test transfer and repeatedly-hit traps
- `rangeview_legacy/` — every exploratory Python script from the session
- `rangeview_originals/` — untouched copies of the migrated files
- `legacy_shell_scripts/`, `legacy_results/`, `legacy_logs/`, `legacy_artifacts/` — supporting output from those scripts

The main flow above only keeps what shipped in the winning submission.
