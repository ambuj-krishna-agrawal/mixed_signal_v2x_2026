"""Caches the TEST split as range images, in the same format as build_cache.py.

The test split has a completely different layout from train/val, so
build_infer_cache.py (which drives MixedSignalsExplorer, and needs
PointClouds/mini_N + Odometry + timestamps) cannot be used at all:

    f_<hex8>/{top,dome,laser,003,004}.pcd + transforms.json

Each frame is self-contained and anonymised -- no scene grouping, no ordering.
So the devkit geometry is reproduced here directly. Every rule below was
VERIFIED against the training cache rather than assumed:

  * p_ego = T[agent] @ p_raw, uniformly for all five agents. transforms.json
    holds map->ego for the two RSUs (`top` and `dome` share one matrix, and
    their .pcd files are in the map frame) and body->ego for the vehicles.
    Check: applying it puts top's no-returns at (0.00, 0.00, 0.00) on 37.2% of
    pixels and dome's at (0.04, 0.01, -0.26) on 11.9%. The training cache has
    them at exactly the same coordinates, on 37.4% and 11.7%.

  * vehicles get z += -3.25 BEFORE the transform, as the devkit does. Check,
    frame-internally against `top`'s ground plane over shared 2m BEV cells:
    agent 004 sits +3.449 m too high without it, and +0.211 m (MAD 0.109) with
    it. The training cache's own residual is +0.329 m at MAD 0.109.

  * validity is the no-return mask. In the map-frame RSU clouds a no-return is
    a repeated exact coordinate (the sensor's own position), so it is found as
    the modal point rather than hardcoded. The vehicles are marked fully valid
    because that is what the TRAINING cache contains: build_cache.py tested
    `norm(p) > 0.05` after the devkit had already applied z += -3.25, so a
    vehicle no-return at the body origin landed at (0,0,-3.25), norm 3.25, and
    was kept. That is arguably a bug, but the model was TRAINED that way and
    inference must match it, so it is reproduced deliberately.

  * 003/004 are decimated 2x in azimuth, exactly as the training cache is.

Residual channels are impossible here (no temporal neighbours), so only a
cin=6 no-residuals checkpoint can consume this cache.

    python build_test_cache.py --frames-dir <extracted> --out cache_test
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

AGENTS = ("top", "dome", "laser", "003", "004")
AGENT_SHAPE = {"top": (64, 1024), "dome": (128, 1024), "laser": (128, 1024),
               "003": (128, 2048), "004": (128, 2048)}
AZIMUTH_STRIDE = {"003": 2, "004": 2}
VEHICLES = ("laser", "003", "004")
VEHICLE_DZ = -3.25          # devkit: put vehicle body height near the RSU's
INTENSITY_SCALE = 3500.0    # mixed_signals.py:183 clip(i/3500, 0, 1). NOT optional:
                            # raw .pcd intensity runs to ~5700, training saw [0,1].
NO_RETURN_EPS = 0.05
PCD_HEADER_LINES = 11


def read_pcd(path: Path) -> np.ndarray:
    """(N, 4) float32 xyz+intensity from an ASCII .pcd, header skipped."""
    with open(path, "rb") as f:
        for _ in range(PCD_HEADER_LINES):
            f.readline()
        buf = f.read()
    try:                                    # pandas' C parser is ~5x np.loadtxt
        import pandas as pd
        import io
        a = pd.read_csv(io.BytesIO(buf), sep=r"\s+", header=None,
                        dtype=np.float32).to_numpy()
    except Exception:
        a = np.fromstring(buf, dtype=np.float32, sep=" ").reshape(-1, 4)
    return a[:, :4]


def no_return_mask(xyz: np.ndarray) -> np.ndarray:
    """True where a pixel is a no-return, found as the modal exact coordinate.

    Only treated as a no-return locus if it covers >=1% of the lattice; a real
    surface never repeats one coordinate to 4dp that often.
    """
    uq, inv, cnt = np.unique(np.round(xyz, 4), axis=0,
                             return_inverse=True, return_counts=True)
    k = cnt.argmax()
    if cnt[k] < 0.01 * len(xyz):
        return np.zeros(len(xyz), bool)
    return np.linalg.norm(xyz - uq[k], axis=1) <= NO_RETURN_EPS


def build_frame(args) -> tuple[str, bool, str]:
    frame_dir, out_dir = Path(args[0]), Path(args[1])
    out_path = out_dir / f"{frame_dir.name}.npz"
    if out_path.exists():
        return frame_dir.name, True, "skip"
    try:
        T = json.loads((frame_dir / "transforms.json").read_text())
        payload = {}
        for name in AGENTS:
            pcd = frame_dir / f"{name}.pcd"
            if not pcd.exists():
                continue                                  # agent absent
            pc = read_pcd(pcd)
            H, W = AGENT_SHAPE[name]
            if pc.shape[0] != H * W:
                return frame_dir.name, False, f"{name}: {pc.shape[0]} pts != {H*W}"

            xyz = pc[:, :3].astype(np.float64)
            if name in VEHICLES:
                xyz[:, 2] += VEHICLE_DZ

            M = np.asarray(T[name], dtype=np.float64)
            ego = xyz @ M[:3, :3].T + M[:3, 3]

            # Vehicles are deliberately all-valid, to match the training cache.
            valid = np.ones(len(ego), bool)
            if name not in VEHICLES:
                valid = ~no_return_mask(ego)

            inten = np.clip(pc[:, 3:4].astype(np.float64) / INTENSITY_SCALE, 0.0, 1.0)
            img = np.concatenate([ego, inten], axis=1)
            img = img.reshape(H, W, 4)
            vm = valid.reshape(H, W)
            s = AZIMUTH_STRIDE.get(name, 1)
            if s > 1:
                img, vm = img[:, ::s], vm[:, ::s]
            payload[f"{name}_img"] = img.astype(np.float16)
            payload[f"{name}_valid"] = vm.astype(np.uint8)

        if not payload:
            return frame_dir.name, False, "no agents"
        # empty GT, matching the no-label val cache the dataset already reads
        payload["gt_boxes"] = np.zeros((0, 7), np.float32)
        payload["gt_class"] = np.zeros((0,), np.int64)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.rename(out_path)
        return frame_dir.name, True, "ok"
    except Exception as e:
        return frame_dir.name, False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True,
                    help="directory of extracted f_<hex>/ frame folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = sorted(p for p in Path(a.frames_dir).iterdir()
                    if p.is_dir() and (p / "transforms.json").exists())
    if not frames:
        raise SystemExit(f"no frame folders under {a.frames_dir}")
    print(f"{len(frames)} frames -> {out}", flush=True)

    ok = skip = 0
    fails = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (name, good, msg) in enumerate(
                ex.map(build_frame, [(str(f), str(out)) for f in frames],
                       chunksize=4), 1):
            if not good:
                fails.append((name, msg))
            elif msg == "skip":
                skip += 1
            else:
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(frames)}  ok={ok} skip={skip} fail={len(fails)}",
                      flush=True)
    print(f"done: {ok} written, {skip} already cached, {len(fails)} failed")
    for n, m in fails[:10]:
        print(f"  FAIL {n}: {m}")


if __name__ == "__main__":
    main()
