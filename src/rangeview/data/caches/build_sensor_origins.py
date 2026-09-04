"""Index of each agent's sensor origin, per frame, for both caches.

Needed for the sensor-relative ray geometry: the range channel in build_input is
measured from the EGO origin while the lattice is the agent's own beam x azimuth
grid, so for the four non-`top` agents a range discontinuity no longer coincides
with an object boundary. Fixing that needs the observing sensor's position.

It is recoverable from the cache with no re-download. A sensor's no-returns are
all written at one exact coordinate, so the modal point of an agent's cloud is
that locus. The locus IS the sensor origin in cache coordinates, for every agent, with no
correction. Verified on the raw .pcd files: the RSUs write theirs at their own
map-frame position (top -> (0,0,0) after the transform, dome -> (0.04, 0.01,
-0.26)), and the vehicles write theirs at body (0, 0, +1.90) -- the lidar's
mount height above the body frame, NOT the body origin. Because the devkit
shifts every point by z -= 3.25 before the transform, the sensor ends up at
T @ (0, 0, 1.90 - 3.25), which is exactly where the locus lands.

Cross-checked against transforms.json, which carries the pose independently:
for `laser` its t_z = -3.269 predicts a sensor at ~-4.62 and the locus reads
-4.61. An earlier version of this file added +3.25 back on the theory that the
locus was the body origin; that was a 3.25 m error in every vehicle ray.

Verified: the recovered locus moves smoothly frame to frame at physically
sensible speeds -- laser 0.54-0.60 m per 0.1 s (~21 km/h), 003 0.30-0.36 m
(~12 km/h), 004 0.02-0.03 m (parked). A coincidence would jump around.

Written as ONE index file rather than rewritten into every .npz, which would
mean rereading and recompressing ~46 GB of cache.

    python build_sensor_origins.py --cache cache      --out sensor_origins.npz
    python build_sensor_origins.py --cache cache_test --out sensor_origins_test.npz
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

AGENTS = ("top", "dome", "laser", "003", "004")
SAMPLE = 40000          # modal point is 3-37% of pixels; a sample finds it reliably


def modal_point(xyz, rng):
    """The repeated coordinate an agent writes for its no-returns, or None."""
    if len(xyz) == 0:
        return None
    idx = rng.choice(len(xyz), size=min(SAMPLE, len(xyz)), replace=False)
    s = np.round(xyz[idx], 3)
    uq, cnt = np.unique(s, axis=0, return_counts=True)
    k = cnt.argmax()
    if cnt[k] < 0.01 * len(s):          # no dominant duplicate -> no no-returns
        return None
    return uq[k]


def one(path):
    rng = np.random.default_rng(0)
    z = np.load(path)
    out = np.full((len(AGENTS), 3), np.nan, np.float32)
    for i, a in enumerate(AGENTS):
        if f"{a}_img" not in z.files:
            continue
        im = z[f"{a}_img"].astype(np.float32).reshape(-1, 4)
        m = modal_point(im[:, :3], rng)
        if m is None:
            continue
        out[i] = m
    return Path(path).stem, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    files = sorted(Path(a.cache).glob("*.npz"))
    print(f"{len(files)} frames in {a.cache}", flush=True)
    names, origins = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (nm, o) in enumerate(ex.map(one, [str(f) for f in files], chunksize=16), 1):
            names.append(nm); origins.append(o)
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    origins = np.stack(origins)
    np.savez_compressed(a.out, names=np.array(names), origins=origins,
                        agents=np.array(AGENTS))
    print(f"saved {a.out}")
    for i, ag in enumerate(AGENTS):
        o = origins[:, i]
        ok = np.isfinite(o).all(1)
        if ok.sum():
            print(f"  {ag:<7} found in {100*ok.mean():5.1f}% of frames | "
                  f"mean ({o[ok,0].mean():7.2f},{o[ok,1].mean():7.2f},{o[ok,2].mean():6.2f}) "
                  f"| std {o[ok].std(0).round(2)}")
        else:
            print(f"  {ag:<7} NOT FOUND in any frame")


if __name__ == "__main__":
    main()
