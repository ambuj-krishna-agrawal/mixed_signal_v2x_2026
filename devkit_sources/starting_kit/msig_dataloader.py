#!/usr/bin/env python3
"""msig_dataloader: reference dataloader for the Mixed Signals V2X challenge.

One loader, one uniform frame representation, for every split, so a detector
written against it runs unchanged in both competition phases:

    Frame = {
        "frame_id": str,
        "agents": { name: {"points": (N,4) xyzi, "T_to_top": (4,4)} },
        "label":  (M,8) or None,     # train only, if labels/ is present
    }

Two auto-detected backends (chosen by the frame id):
  * Native (train / val): ids look like mini_<segment>_<syncIndex>. Clouds
    are grouped by shared sync index and each agent's transform into the ego
    ("top") frame is computed from odometry, mirroring the ground-truth
    pipeline exactly (shared math in msig_transforms.py).
  * Bundle (test): ids are opaque tokens (e.g. f_a3f9c1b2). Each frame is a
    self-contained folder test/<id>/ of role.pcd clouds plus transforms.json;
    the transforms are precomputed, so nothing needs time-aligning.

fuse(frame) concatenates all agents into the ego ("top") frame, the input a
BEV detector expects.

Pure numpy. Requires the co-located msig_transforms.py (the organizers' shared
transform module, shipped with this kit).

Example
-------
    from msig_dataloader import MixedSignals

    # Development phase (native val split; root has PointClouds/ + Odometry/)
    val = MixedSignals("path/to/val", split="val")
    for fid in open("validation_frames.txt"):
        fid = fid.split("#")[0].strip()
        if not fid:
            continue
        frame = val.frame(fid)
        cloud = val.fuse(frame)          # (M,4) in the top ego frame
        # ... run your detector on cloud, write <fid>.txt ...

    # Final phase (test bundles; root is the test/ folder of <id>/ bundles)
    test = MixedSignals("path/to/test", split="test")
    for fid in open("test_frames.txt"):
        fid = fid.split("#")[0].strip()
        if not fid:
            continue
        frame = test.frame(fid)
        cloud = test.fuse(frame)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msig_transforms import (  # noqa: E402
    VEHICLE_AGENTS, frame_agents, fuse, load_odom, read_pcd, read_transforms_json,
)


def _read_bundle(bundle_dir):
    """Read a test bundle folder -> {agent: {"points", "T_to_top"}}."""
    T = read_transforms_json(os.path.join(bundle_dir, "transforms.json"))
    agents = {}
    for agent, mat in T.items():
        pcd = os.path.join(bundle_dir, f"{agent}.pcd")
        agents[agent] = {"points": read_pcd(pcd), "T_to_top": mat}
    return agents


def _load_label(root, n, si):
    """Optional per-frame GT (train only): root/labels/mini_<n>/mini_<n>_<si>.txt."""
    fp = os.path.join(root, "labels", f"mini_{n}", f"mini_{n}_{si}.txt")
    if not os.path.isfile(fp):
        return None
    return np.loadtxt(fp).reshape(-1, 8)


class MixedSignals:
    """Frame loader over one split; caches odometry per segment (native splits)."""

    def __init__(self, root, split="test"):
        self.root = root
        self.split = split
        self._odom = {}   # segment -> {vehicle_agent: odom}

    def frame(self, frame_id):
        if frame_id.startswith("mini_"):
            return self._native_frame(frame_id)
        return self._bundle_frame(frame_id)

    def _native_frame(self, frame_id):
        _, n, si = frame_id.split("_")
        n, si = int(n), int(si)
        if n not in self._odom:
            self._odom[n] = {a: load_odom(self.root, n, a) for a in VEHICLE_AGENTS}
        agents = frame_agents(self.root, n, si, self._odom[n], load_points=True)
        return {"frame_id": frame_id, "agents": agents,
                "label": _load_label(self.root, n, si)}

    def _bundle_frame(self, frame_id):
        agents = _read_bundle(os.path.join(self.root, frame_id))
        return {"frame_id": frame_id, "agents": agents, "label": None}

    @staticmethod
    def fuse(frame):
        """Concatenate all agents into the top ego frame -> (M,4)."""
        return fuse(frame["agents"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Load one frame and print a summary.")
    ap.add_argument("root", help="split root: native split dir, or test/ folder")
    ap.add_argument("frame_id", help="e.g. mini_9_100  or  f_a3f9c1b2")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = MixedSignals(args.root, split=args.split)
    fr = ds.frame(args.frame_id)
    print(f"frame {fr['frame_id']}: agents = {sorted(fr['agents'])}")
    for a, info in sorted(fr["agents"].items()):
        print(f"  {a:6s} points={info['points'].shape}  T_to_top det="
              f"{np.linalg.det(info['T_to_top'][:3, :3]):.3f}")
    cloud = ds.fuse(fr)
    print(f"fused cloud: {cloud.shape} in the top ego frame")
    if fr["label"] is not None:
        print(f"label boxes: {len(fr['label'])}")
