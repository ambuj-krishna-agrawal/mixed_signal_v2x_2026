# Submission Format

A submission is a zip archive of per-frame detection files.

## Files

* One plain-text file per evaluation frame, named `<frame_id>.txt`. Frames are
  matched to ground truth by the filename stem, so the name must equal the
  frame id exactly. The full list is in `validation_frames.txt` (Development) or
  `test_frames.txt` (Final).
  * Development frame ids are `mini_<segment>_<syncIndex>` (e.g.
    `mini_9_118.txt`).
  * Final frame ids are opaque tokens (e.g. `f_a3f9c1b2.txt`). The test
    split is temporally anonymized, so ids carry no ordering or timing. Read them
    from `test_frames.txt`. Each corresponds to one per-frame bundle under
    `test/<frame_id>/` (agent `.pcd` clouds plus `transforms.json`).
* Put the `*.txt` files at the root of the zip. An enclosing folder is
  tolerated (the scorer searches recursively), but no folder is cleanest.
* You must provide a file for every frame id listed for the phase
  (`validation_frames.txt` for Development). A frame with no file, or an empty
  file, is scored as zero detections for that frame (not skipped).

## Lines

Each non-empty, non-comment line is one detection with 9 whitespace-separated
columns:

```
cls  cx  cy  cz  dx  dy  dz  yaw  score
```

| Column | Meaning | Units / range |
|:--|:--|:--|
| `cls` | class id | `1`=Vehicle, `2`=Bike, `3`=Pedestrian |
| `cx cy cz` | box centre | metres, "top" ego frame |
| `dx dy dz` | length, width, height | metres |
| `yaw` | heading (rotation about +z) | radians |
| `score` | detection confidence | float in `[0, 1]` |

* Lines starting with `#` and blank lines are ignored.
* Class ids other than 1/2/3 are ignored.
* Boxes whose centre is outside `x, y ∈ [-51.2, 51.2] m` are dropped before
  scoring.
* `score` ranks detections within a frame (higher = matched first). It does not
  need to be calibrated, but a meaningful ordering improves AP.

## Example (`mini_9_118.txt`)

```
# cls cx cy cz dx dy dz yaw score
1   12.34  -3.21  0.05  4.10  1.92  1.55  1.5708  0.97
1    5.10   8.80  0.00  4.30  1.95  1.60 -0.0100  0.81
2   -7.42   2.10  0.10  1.80  0.60  1.55  3.1100  0.64
3    8.01   2.44  0.10  0.80  0.60  1.70  0.0000  0.55
```

Validate before uploading. Pass `--frames` to also confirm every required
frame is present (a missing frame is scored as zero, not skipped):

```bash
python3 check_submission.py path/to/predictions/ --frames validation_frames.txt
```
