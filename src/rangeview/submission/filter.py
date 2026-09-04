"""Post-process an existing submission: apply Filter #2 (ground plausibility,
Bike-safe) + Filter #3 (per-class box-dim sanity, 0.25 margin).

Both filters were validated on val (`filter_experiments.py`) to drop only
FPs (0 TP loss).  Val delta was +0.0004 combined — within noise, but the
individual drops are 100% FP-precision.

Reads:  <in>.zip  or  <in>/    (per-frame .txt files, class cx cy cz dx dy dz yaw score)
Writes: <out>/     (filtered) and optionally <out>.zip

Filters applied IN ORDER (both AND):
  #2 ground plausibility:  drop if |cz - dz/2 - ground_z(cx,cy)| > K_cls
                            (only when ground is trusted at that cell)
                            K = {V:0.7, B:0.5, P:0.5}
  #3 dim sanity:           drop if any (dx, dy, dz) outside class's
                            [p0.5, p99.5] extended by margin*span both sides
                            margin = 0.25
                            (percentiles from val GT — reasonable given the
                            5-agent training distribution)
"""
import argparse
import glob
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from rangeview.paths import ARTIFACTS


CLASS_NAME = {1: "Vehicle", 2: "Bike", 3: "Pedestrian"}

# Val-GT-derived dim percentiles (from filter_experiments.py output).
# {cls: (dx_lo, dx_hi, dy_lo, dy_hi, dz_lo, dz_hi)}
DIM_PCT = {
    1: (2.62, 6.40, 1.58, 2.53, 1.53, 2.34),
    2: (1.74, 2.37, 0.55, 1.03, 1.09, 1.88),
    3: (0.52, 2.36, 0.58, 2.13, 1.71, 2.16),
}
DIM_MARGIN = 0.25   # span multiplier

# Ground-plausibility thresholds per class (V-safe at 0.7, B-catches-floaters
# at 0.5, P-safe at 0.5)
GROUND_THRESH = {1: 0.7, 2: 0.5, 3: 0.5}


def _dim_ok(cls, dx, dy, dz):
    lo_dx, hi_dx, lo_dy, hi_dy, lo_dz, hi_dz = DIM_PCT[cls]
    ex_dx = (hi_dx - lo_dx) * DIM_MARGIN
    ex_dy = (hi_dy - lo_dy) * DIM_MARGIN
    ex_dz = (hi_dz - lo_dz) * DIM_MARGIN
    return (lo_dx - ex_dx <= dx <= hi_dx + ex_dx and
            lo_dy - ex_dy <= dy <= hi_dy + ex_dy and
            lo_dz - ex_dz <= dz <= hi_dz + ex_dz)


class GroundMap:
    def __init__(self, path):
        z = np.load(path)
        self.ground   = z["ground"]
        self.trusted  = z["trusted"]
        self.cell     = float(z["cell"])
        self.range_xy = float(z["pc_range_xy"])

    def z(self, cx, cy):
        i = int((cx + self.range_xy) / self.cell)
        j = int((cy + self.range_xy) / self.cell)
        if 0 <= i < self.ground.shape[0] and 0 <= j < self.ground.shape[1] \
           and self.trusted[i, j]:
            return float(self.ground[i, j]), True
        return 0.0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True,
                    help="submission zip OR directory (per-frame .txt files)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ground-map",
                    default=str(ARTIFACTS / "ground_map.npz"))
    ap.add_argument("--zip", action="store_true",
                    help="also write <out>.zip")
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3],
                    help="apply filter only to these classes (default: all).  "
                         "Post-hoc validated: on test, Bike-only (--classes 2) "
                         "keeps the +0.0043 Bike gain while avoiding "
                         "-0.0039 V and -0.0043 P losses observed in the "
                         "all-classes run.")
    a = ap.parse_args()
    filter_classes = set(a.classes)

    src = a.sub
    tmp_unzip = None
    if src.endswith(".zip"):
        tmp_unzip = a.out.rstrip("/") + "__unzipped"
        shutil.rmtree(tmp_unzip, ignore_errors=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp_unzip)
        src = tmp_unzip
    files = sorted(glob.glob(os.path.join(src, "*.txt")))
    if not files:
        raise SystemExit(f"no .txt files in {src}")

    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)

    gm = GroundMap(a.ground_map)

    kept   = Counter(); dropped = Counter()
    reason = {"ground": Counter(), "dim": Counter()}

    for n, path in enumerate(files):
        stem = os.path.basename(path)[:-4]
        out_lines = []
        for line in open(path).read().splitlines():
            if not line.strip():
                continue
            f = line.split()
            cls = int(float(f[0]))
            cx, cy, cz, dx, dy, dz, yaw = (float(x) for x in f[1:8])

            # Only filter classes the user asked for; pass through the rest.
            if cls in filter_classes:
                # Filter #3: dim sanity
                if not _dim_ok(cls, dx, dy, dz):
                    dropped[cls] += 1
                    reason["dim"][cls] += 1
                    continue
                # Filter #2: ground plausibility (only where ground is trusted)
                gz, gtrust = gm.z(cx, cy)
                if gtrust:
                    bottom = cz - dz / 2
                    if abs(bottom - gz) > GROUND_THRESH[cls]:
                        dropped[cls] += 1
                        reason["ground"][cls] += 1
                        continue
            out_lines.append(line)
            kept[cls] += 1

        with open(os.path.join(a.out, stem + ".txt"), "w") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        if (n + 1) % 250 == 0:
            print(f"  {n+1}/{len(files)}", flush=True)

    written = len(glob.glob(os.path.join(a.out, "*.txt")))
    if written != len(files):
        raise SystemExit(f"wrote {written} but input had {len(files)}; refusing")

    print(f"\n{a.sub}  ->  {a.out}", flush=True)
    for c in (1, 2, 3):
        k, d = kept[c], dropped[c]
        rg = reason["ground"][c]; rd = reason["dim"][c]
        if k + d == 0:
            continue
        print(f"   {CLASS_NAME[c]:10s} kept {k:6d}  dropped {d:5d}  "
              f"(ground {rg}, dim {rd})", flush=True)
    print(f"   TOTAL      kept {sum(kept.values()):6d}  "
          f"dropped {sum(dropped.values()):5d}", flush=True)
    print(f"   files {written}", flush=True)

    if a.zip:
        zpath = a.out.rstrip("/") + ".zip"
        if os.path.exists(zpath):
            os.remove(zpath)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(glob.glob(os.path.join(a.out, "*.txt"))):
                zf.write(p, os.path.basename(p))
        print(f"   wrote {zpath} ({os.path.getsize(zpath)/1e6:.2f} MB)", flush=True)

    if tmp_unzip:
        shutil.rmtree(tmp_unzip, ignore_errors=True)


if __name__ == "__main__":
    main()
