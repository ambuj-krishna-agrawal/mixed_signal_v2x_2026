# Starting Kit: Mixed Signals V2X Collaborative 3D Detection Challenge

This kit gets you from "I have the dataset" to "I have a valid submission".

## Contents

| File | Purpose |
|:--|:--|
| `SUBMISSION_FORMAT.md` | The exact prediction-file specification. |
| `msig_dataloader.py` | Reference loader: one uniform frame representation for every split. |
| `msig_transforms.py` | Shared transform / IO math the loader is built on. |
| `check_submission.py` | Validates a submission folder/zip before you upload it. |
| `sample_submission/` | A tiny, valid example submission you can copy and edit. |
| `validation_frames.txt` | Frame ids you must predict in the Development phase. |
| `test_frames.txt` | Frame ids you must predict in the Final phase. |

> The `validation_frames.txt` list and the corresponding validation point
> clouds correspond to the public validation split carved from the training
> scenes. The Final-phase frame ids are released when the Final phase opens.

## Quick start

1. Get the data. The competition data is hosted on HuggingFace as
   per-segment `.tar` archives at
   [`sberrio/Mixed-Signals-V2X`](https://huggingface.co/datasets/sberrio/Mixed-Signals-V2X).
   Download the segments you need and extract in place (`tar xf`). Each archive
   rebuilds the `PointClouds/ Odometry/ labels/` layout the dataloader reads. See
   the Overview tab for exact commands. The dev-kit and baselines are at
   <https://github.com/acfr/Mixed-Signals-Dataset>.
2. Train / run a detector. Any approach is allowed, single-agent or fused
   across the 5 LiDARs. The dataset repo provides PointPillar baselines
   (no/early/intermediate/late fusion) you can start from.
3. Export predictions in the "top" ego frame, one `*.txt` per frame, named
   by frame id (see `SUBMISSION_FORMAT.md`).
4. Check your output. Pass `--frames` so the checker also verifies you have a
   file for every required frame (a missing frame is scored as zero, not
   skipped):
   ```bash
   python3 check_submission.py path/to/my_predictions/ --frames validation_frames.txt
   ```
5. Zip and submit. Zip the `*.txt` files (no enclosing folder needed) and
   upload on the competition's Submit tab.

## Reproduce your score locally

The scoring program (`scoring_program/scoring.py`, downloadable from the
competition) only needs `numpy`. If you hold out your own labelled frames in the
same `cls cx cy cz dx dy dz yaw` format, you can score yourself:

```bash
mkdir -p input/ref input/res output
cp my_labels/*.txt   input/ref/   # ground truth: 8 columns (no score)
cp my_preds/*.txt    input/res/   # predictions: 9 columns (with score)
python3 scoring.py input output
cat output/scores.json
```

This produces exactly the same mAP and per-class AP as the leaderboard.
