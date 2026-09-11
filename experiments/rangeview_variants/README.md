# rangeview experiment variants

A **verbatim snapshot** of the working rangeview tree carrying three architecture
experiments. Nothing under `src/rangeview/` is modified by this branch.

These files are *not* integrated into the `rangeview` package. They use flat
imports (`from model import ...`), the CUDA-only OpenCOOD IoU kernels, and
`torch_scatter` directly — all three of which `src/rangeview/` deliberately
replaced with `rangeview.compat` shims for MPS/CPU portability. Integrating
means porting the deltas, not copying these files over the package.

Self-contained: the 19 modules here are the full transitive closure, so the tree
runs standalone from this directory.

## The three experiments

### 1. Range-adaptive attention — `range_attention.py`

`RangeAdaptiveAttention`, applied inside `RangeEncoder`. Enable with `--attn`.

The motivation is that a range image's spatial statistics change drastically with
range — nearby surfaces occupy many pixels, distant ones a few — so a fixed
receptive field is wrong somewhere in the image. This conditions encoder
attention on range.

**Result: it did not work.** It lost 0.0037 mAP against the baseline, and lost
more at epoch 7 than at epoch 6 (i.e. it degraded with training).

It is included because the *class-wise* breakdown is not uniformly negative and
may still be informative: it is the only variant that improved Pedestrian
without the z-ceiling change, and it gained +0.0166 on seq34.

### 2. Z-ceiling — `model.py`, `z_ceiling` argument

Clips the BEV pillar column at a fixed height instead of `PC_RANGE[5]`. Enable
with `--z-ceiling -0.8`. Motivation: canopy and foliage returns above the
objects of interest dominate `scatter_max`, which keeps the per-channel maximum
and so lets one high leaf return represent a whole cell.

### 3. Separate high-resolution Pedestrian heatmap — `--hires 3` / `--ped-hires`

Gives selected classes their own head at a finer output stride
(`PED_HIRES_STRIDE`) rather than sharing the stride-8 grid. Motivation:
pedestrians are ~0.9 m wide, close to a single cell at the shared stride.

`--hires` takes class ids (`1`=Vehicle, `2`=Bike, `3`=Pedestrian), so
`--hires 2 3` is bike + pedestrian. `--ped-hires` is the legacy equivalent of
`--hires 3`. This one spans four files — the head in `model.py`, target
generation in `dataset.py`, the loss at the hires stride in `loss.py`, and
decoding in `decode.py` — so it cannot be cherry-picked from `model.py` alone.

> **These two were trained together and never ablated apart.** The checkpoint
> that produced the Pedestrian gain records *both* `z_ceiling=-0.8` and
> `ped_hires=True`. The +0.0283 Pedestrian improvement below cannot be
> attributed to either one individually. Separating them is the obvious next
> ablation and has not been run.

## Measured results

Validation = scenes 31 and 34. `mAP = mean(Vehicle AP@0.5, Bike AP@0.5,
Pedestrian AP@0.3)`, matching the official scorer. "+RSU+ccNMS" is the deployed
post-processing stack.

| config | mAP | Vehicle | Bike | Pedestrian | seq31 | seq34 |
|---|---|---|---|---|---|---|
| baseline ep7 | **0.8468** | 0.8976 | 0.9003 | 0.7424 | 0.8724 | 0.8058 |
| `--attn` ep6 | 0.8431 | 0.8982 | 0.8799 | **0.7513** | 0.8585 | **0.8224** |
| `--attn` ep7 | 0.8334 | 0.8903 | 0.8641 | 0.7459 | 0.8481 | 0.8116 |
| `--z-ceiling -0.8 --ped-hires` ep7 | **0.8517** | 0.8860 | 0.8983 | **0.7707** | 0.8602 | **0.8372** |

Without the RSU filter (raw decode):

| config | mAP | Vehicle | Bike | Pedestrian |
|---|---|---|---|---|
| baseline ep7 | 0.8247 | 0.8584 | 0.8804 | 0.7352 |
| `--attn` ep6 | 0.8145 | 0.8470 | 0.8559 | 0.7406 |
| `--z-ceiling -0.8 --ped-hires` ep7 | 0.8284 | 0.8319 | 0.8880 | 0.7654 |

Summary:

- **z-ceiling + ped-hires is the only net win**: +0.0049 mAP, driven entirely by
  Pedestrian (+0.0283). It costs Vehicle (−0.0116), which is why the submitted
  system took Vehicle/Bike from the baseline and Pedestrian from this checkpoint
  rather than using one model for all three.
- **range attention is a net loss** (−0.0037) but improves Pedestrian (+0.0089)
  and seq34 (+0.0166).
- Both variants move Pedestrian and seq34 in the same direction, which is worth
  noting: seq34 tracked the leaderboard far better than seq31 or the pooled
  average throughout.

## Also included but not requested

`model.py` additionally carries `--fourier` (`fourier_azimuth.py`, azimuth
Fourier features), `--vrm`, `--density`, `--per-agent-intensity`, and background
channels. None of these beat the baseline. `agent_fusion.py` here is a v2 of the
package's version, adding occupancy-conditioned attention (`occ_proj`,
`log_temp`); it loads pre-v2 checkpoints via `_load_from_state_dict`, which
supplies identity values for the missing keys. **The occupancy conditioning was
never A/B'd** — it initialises to exactly zero, so a fresh v2 and a v1 model are
identical at step 0, and no run isolates its effect.

## Dependencies not present in `src/rangeview/`

| module | status |
|---|---|
| `pillar_vfe.py` | included here; excluded from the package on purpose (`use_vfe=False` is the shipped recipe — PillarVFE was a wash) |
| `gt_paste.py`, `rv_targets.py` | included here; not in the package |
| `opencood` | external CUDA IoU/NMS kernels, used by `loss.py` and `decode.py`; the package replaces these with Shapely fallbacks in `rangeview.compat` |
| `torch_scatter` | used directly by `model.py`; the package wraps it as `compat.scatter_max_`, **which takes `(out, src, index, dim)` — a different argument order** |

## Known portability issues

These files are a verbatim snapshot and have not been edited, so they still
carry absolute paths from the machine they ran on.

**Breaks on import** — `score_exact.py:15` is a module-level constant:

```python
_SP = Path("/home/mehul/mixed_signals/scoring_program/scoring.py")
```

Point it at this repo's own copy, which is already present at
`scoring_program/scoring.py`.

**Overridable** — these are argparse defaults, so pass the flag instead of
editing:

| file | flag | default |
|---|---|---|
| `train.py:72`, `evaluate.py:124` | `--cache` | `/home/mehul/mixed_signals/rangeview/cache` |
| `train.py:92` | `--out` | `/home/mehul/mixed_signals/rangeview/checkpoints` |
| `train.py:93` | `--tb-dir` | `/home/mehul/mixed_signals/rangeview/tb` |

## Running

```bash
# baseline
python train.py --epochs 8

# range attention
python train.py --epochs 8 --attn

# z-ceiling + separate pedestrian heatmap (the configuration that won Pedestrian)
python train.py --epochs 8 --z-ceiling -0.8 --hires 3
```

Flags are recorded into the checkpoint and read back at inference, so
`eval_ckpt.py` / `make_submission.py` reconstruct the right architecture without
being told which variant produced it.

## Inference cost

Measured on an RTX 4050 Laptop (6 GB), batch size 1, single checkpoint, I/O
excluded: 7.64 M parameters, 182 ms/frame eager FP32 at 1.6 GB peak, or **95
ms/frame (10.5 FPS) with `torch.compile` + AMP fp16**. The profile is dominated
by elementwise ops (~51% of CUDA time) from the Meta-Kernel's `unfold`, not by
convolutions — it is memory-bandwidth-bound. `channels_last` measured 11%
*slower* and should not be used.
