"""Devkit-free train/val cache builder.

Same output layout as build_cache.py (seq{N}_idx{M}.npz with per-agent images,
validity masks, gt_boxes, gt_class), but built from the starting_kit's
msig_dataloader.py + msig_transforms.py instead of the OpenCOOD devkit.

Two conventions must be reconciled against the training cache the epoch-7
checkpoint was trained on:
  * msig_transforms's agent_to_top() for VEHICLES bakes ZLIFT=+3.25 into the
    translation.  Applied to raw body-frame .pcd values that is *not* what the
    OpenCOOD devkit does — the devkit pre-shifts vehicle z by -3.25 and its own
    transform has no ZLIFT.  These two conventions differ by R @ [0,0,+6.5m]
    in the produced ego points, per empirical check on mini_1_100.
  * The training cache was written by the devkit convention.
  * build_test_cache.py already handles this via `z += VEHICLE_DZ = -3.25`
    BEFORE the transform-from-json, which cancels the ZLIFT baked in.

So this builder mirrors build_test_cache's math (z -= 3.25 for vehicles first,
then apply msig_transforms.T which includes ZLIFT — the two cancel), and
recovers no-return validity in the frame each agent's raw .pcd sits in
(map for RSUs, body for vehicles).

Output frame naming: `seq{N}_idx{M}.npz` where N = mini segment, M = sync index.
That matches the pattern dataset.py's `scene_of()` expects.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Set up import paths for the starting_kit modules.
from rangeview.paths import DEVKIT
sys.path.insert(0, str(DEVKIT / "starting_kit"))
from msig_transforms import (  # noqa: E402
    AGENTS, RSU_AGENTS, VEHICLE_AGENTS, SHIFT, ZLIFT,
    agent_to_top, load_odom, nearest_pose, parse_pcd_name, read_pcd,
)

AGENT_SHAPE = {
    "top":   (64, 1024),
    "dome":  (128, 1024),
    "laser": (128, 1024),
    "003":   (128, 2048),
    "004":   (128, 2048),
}
AZIMUTH_STRIDE = {"003": 2, "004": 2}
INTENSITY_SCALE = 3500.0
NO_RETURN_EPS = 0.05
MIN_PTS_IN_BOX = 5


def _in_box_count(pts_xyz, box):
    """Number of `pts_xyz` (M,3) inside the rotated box [cx cy cz dx dy dz yaw]."""
    cx, cy, cz, dx, dy, dz, yaw = box[:7]
    t = pts_xyz - np.array([cx, cy, cz], np.float32)
    ca, sa = np.cos(-yaw), np.sin(-yaw)
    lx = t[:, 0] * ca - t[:, 1] * sa
    ly = t[:, 0] * sa + t[:, 1] * ca
    return int(((np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2)
                & (np.abs(t[:, 2]) <= dz / 2)).sum())


def _agent_points(root, n, agent, si, odom):
    """(H, W, 4) ego-frame image + (H, W) validity for one agent at (n, si).

    Returns (img, valid) or (None, None) if the .pcd is missing or has an
    unexpected point count (lattice violated -> better to drop than guess).
    """
    import glob
    fs = glob.glob(os.path.join(root, "PointClouds", f"mini_{n}", f"{agent}_{si}_*.pcd"))
    if not fs:
        return None, None
    fp = fs[0]

    pts = read_pcd(fp).astype(np.float64)                 # (N, 4)
    H, W = AGENT_SHAPE[agent]
    if pts.shape[0] != H * W:
        return None, None

    if agent in RSU_AGENTS:
        # RSU .pcd is in the map frame.  agent_to_top(RSU) is a pure +SHIFT.
        T = agent_to_top(agent)
        ego = pts[:, :3] @ T[:3, :3].T + T[:3, 3]         # ~ pts_map + SHIFT
        # No-return locus in ego is at (0,0,0) for top, ~ (0.04,0.01,-0.26) for dome.
        # Anything within NO_RETURN_EPS of the ego origin is a no-return.
        valid = np.linalg.norm(ego, axis=1) > NO_RETURN_EPS
    else:
        # Vehicle .pcd is in body frame.  Devkit convention: shift body z by
        # -3.25 BEFORE applying the transform.  msig_transforms.agent_to_top
        # bakes ZLIFT=+3.25 into T for vehicles, which cancels this shift and
        # leaves a pure pose transform to ego — the exact math the training
        # cache was written under.
        _, _, ts_ns = parse_pcd_name(fp)
        if odom.get(agent) is None:
            return None, None
        T = agent_to_top(agent, nearest_pose(odom[agent], ts_ns))
        p_body = pts[:, :3].copy()
        p_body[:, 2] -= ZLIFT                             # devkit's z -= 3.25
        ego = p_body @ T[:3, :3].T + T[:3, 3]
        # Vehicles are deliberately marked all-valid to match the training
        # cache's quirk (HANDOFF §4): validity was decided AFTER z -= 3.25 in
        # the devkit, so a body-origin no-return survived at norm 3.25.
        valid = np.ones(len(ego), bool)

    inten = np.clip(pts[:, 3:4] / INTENSITY_SCALE, 0.0, 1.0)
    img = np.concatenate([ego, inten], axis=1).reshape(H, W, 4)
    vm = valid.reshape(H, W)
    s = AZIMUTH_STRIDE.get(agent, 1)
    if s > 1:
        img, vm = img[:, ::s], vm[:, ::s]
    return img.astype(np.float16), vm.astype(np.uint8)


def _load_label(root, n, si):
    fp = os.path.join(root, "labels", f"mini_{n}", f"mini_{n}_{si}.txt")
    if not os.path.isfile(fp):
        return None
    return np.loadtxt(fp).reshape(-1, 8)


def build_frame(args):
    n, si, root, out_dir = args
    out_path = Path(out_dir) / f"seq{n}_idx{si}.npz"
    if out_path.exists():
        return (n, si, True, "skip")

    # Load per-vehicle odometry once per segment (cheap after first call).
    odom = _agent_odom_cache.setdefault(n, {})
    if not odom:
        for a in VEHICLE_AGENTS:
            odom[a] = load_odom(root, n, a)

    label = _load_label(root, n, si)
    if label is None or len(label) == 0:
        return (n, si, False, "no_label")

    payload = {}
    for a in AGENTS:
        img, vm = _agent_points(root, n, a, si, odom)
        if img is None:
            continue
        payload[f"{a}_img"] = img
        payload[f"{a}_valid"] = vm

    if not payload:
        return (n, si, False, "no_agents")

    # Concatenate all valid points across agents for the ">= MIN_PTS_IN_BOX" filter.
    pts = np.concatenate([
        payload[f"{a}_img"][..., :3].astype(np.float32).reshape(-1, 3)[
            payload[f"{a}_valid"].reshape(-1).astype(bool)]
        for a in AGENT_SHAPE if f"{a}_img" in payload
    ])
    # Label rows: cls cx cy cz dx dy dz yaw   (top ego frame)
    keep = []
    for bi in range(len(label)):
        cls, cx, cy, cz, dx, dy, dz, yaw = label[bi]
        if _in_box_count(pts, [cx, cy, cz, dx, dy, dz, yaw]) > MIN_PTS_IN_BOX:
            keep.append(bi)
    if not keep:
        return (n, si, False, "no_gt_after_filter")

    gt = label[keep]
    payload["gt_boxes"] = gt[:, 1:8].astype(np.float32)       # x y z dx dy dz yaw
    payload["gt_class"] = gt[:, 0].astype(np.int8)            # 1..3

    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    tmp.rename(out_path)
    return (n, si, True, "ok")


_agent_odom_cache = {}


def build(root: Path, out_dir: Path, workers: int, limit: int | None) -> None:
    """Enumerate every labeled (n, si) pair and cache it."""
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_root = root / "labels"
    tasks = []
    for seg_dir in sorted(labels_root.iterdir()):
        n = int(seg_dir.name.split("_")[1])
        for lf in sorted(seg_dir.iterdir()):
            si = int(lf.stem.split("_")[-1])
            tasks.append((n, si, str(root), str(out_dir)))
            if limit is not None and len(tasks) >= limit:
                break
        if limit is not None and len(tasks) >= limit:
            break

    print(f"{len(tasks)} labeled frames -> {out_dir}", flush=True)

    ok = skip = 0
    fails = []
    # ProcessPool cannot pickle module-globals, so per-worker odometry cache is
    # rebuilt on first call.  For a small worker count that's cheap.
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            n, si, good, msg = build_frame(t)
            if not good: fails.append((n, si, msg))
            elif msg == "skip": skip += 1
            else: ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}  ok={ok} skip={skip} fail={len(fails)}",
                      flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (n, si, good, msg) in enumerate(
                    ex.map(build_frame, tasks, chunksize=4), 1):
                if not good: fails.append((n, si, msg))
                elif msg == "skip": skip += 1
                else: ok += 1
                if i % 100 == 0:
                    print(f"  {i}/{len(tasks)}  ok={ok} skip={skip} fail={len(fails)}",
                          flush=True)
    print(f"done: {ok} written, {skip} skipped, {len(fails)} failed")
    if fails[:10]:
        reasons = {}
        for _, _, m in fails: reasons[m] = reasons.get(m, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing PointClouds/, Odometry/, labels/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(Path(a.root), Path(a.out), a.workers, a.limit)
