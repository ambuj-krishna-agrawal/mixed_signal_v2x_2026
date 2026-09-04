#!/usr/bin/env python3
"""Shared transform + I/O primitives for the Mixed Signals V2X competition.

Single source of truth for the agent -> ego ("top") fusion math, imported by:
  * tools/generate_reference_from_v04.py: ground-truth generation
  * tools/anonymize_test.py: bakes per-frame transforms.json
  * starting_kit/msig_dataloader: participant reference loader

Keeping the math in one place guarantees the transforms computed natively for
train/val and the transforms baked into the anonymized test bundles are
identical by construction, not by coincidence.

Pure numpy, numpy-2.x safe (no deprecated np.fromstring); no open3d/pypcd/pandas.

Coordinate conventions (verified against the Mixed Signals devkit):
  * RSU lidars "top", "dome" store their clouds already in the "map" frame; the
    RSU is static, so fusing to the ego "top" frame is the constant pure
    translation +SHIFT with identity rotation, zero temporal information.
  * Vehicle lidars "laser", "003", "004" store clouds in the vehicle body frame;
    fusing to "top" is  p_top = R(quat) @ (p_body + [0,0,ZLIFT]) + pos + SHIFT,
    where (pos, quat) is the odometry pose nearest the cloud's timestamp.
  * Ego frame is "top"; map -> top is the pure translation +SHIFT.
"""
import csv
import glob
import json
import os

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (bit-for-bit the values used by the original GT script).
# --------------------------------------------------------------------------- #
SHIFT = np.array([41.551, 51.878, 1.077])   # map -> top (RSU static; pure translation)
ZLIFT = 3.25                                 # vehicle body z-lift before posing
RANGE = 51.2                                 # +/- range filter (m) in the top frame
MIN_POINTS = 5                               # paper's ">5 points in box" keep rule

RSU_AGENTS = ('top', 'dome')
VEHICLE_AGENTS = ('laser', '003', '004')
AGENTS = ('top', 'dome', 'laser', '003', '004')


# --------------------------------------------------------------------------- #
# SE(3) / rotation helpers.
# --------------------------------------------------------------------------- #
def quat_to_R(w, x, y, z):
    """Rotation matrix from a (w, x, y, z) quaternion."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def se3(R, t):
    """Compose a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T, pts):
    """Apply a 4x4 SE(3) to Nx3 (or Nx4+, extra columns preserved) points."""
    if len(pts) == 0:
        return pts
    xyz = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
    if pts.shape[1] > 3:
        return np.hstack([xyz, pts[:, 3:]])
    return xyz


def agent_to_top(agent, pose=None):
    """4x4 SE(3) mapping an agent's cloud into the "top" ego frame.

    RSU agents (top, dome): `pose` ignored; constant +SHIFT, identity rotation.
    Vehicle agents: `pose` = (pos[3], quat_wxyz[4]) at the cloud's timestamp.
    """
    if agent in RSU_AGENTS:
        return se3(np.eye(3), SHIFT)
    if agent not in VEHICLE_AGENTS:
        raise ValueError(f"unknown agent {agent!r}")
    if pose is None:
        raise ValueError(f"vehicle agent {agent!r} needs a pose")
    pos, quat = pose
    R = quat_to_R(*quat)
    t = R @ np.array([0.0, 0.0, ZLIFT]) + np.asarray(pos) + SHIFT
    return se3(R, t)


# --------------------------------------------------------------------------- #
# Point-cloud + filename + odometry I/O.
# --------------------------------------------------------------------------- #
def read_pcd(fp, with_intensity=True):
    """Read an ASCII `.pcd` (FIELDS x y z intensity) as an (N,4) or (N,3) array."""
    with open(fp, 'rb') as fh:
        raw = fh.read()
    i = raw.find(b'DATA ascii\n')
    if i < 0:
        # Kit clouds are all ASCII; a binary DATA line would otherwise be read
        # as zero points and silently drop the whole agent from the fusion.
        raise ValueError(f"{fp}: not an ASCII .pcd (no 'DATA ascii' line)")
    i += len(b'DATA ascii\n')
    body = raw[i:].decode('ascii', 'ignore')
    # numpy-2.x safe replacement for the deprecated np.fromstring(sep=' ').
    vals = np.array(body.split(), dtype=np.float64)
    if vals.size % 4:
        raise ValueError(f"{fp}: expected 4 fields per point (x y z intensity), "
                         f"got {vals.size} values")
    a = vals.reshape(-1, 4)
    return a if with_intensity else a[:, :3]


def parse_pcd_name(path):
    """`<agent>_<si>_<timestamp>.pcd` -> (agent:str, si:int, ts_ns:int)."""
    base = os.path.basename(path)[:-4]        # strip ".pcd"
    parts = base.split('_')
    ts_ns = int(round(float(parts[-1]) * 1e9))
    si = int(parts[-2])
    agent = '_'.join(parts[:-2])
    return agent, si, ts_ns


def load_odom(ds, n, agent):
    """Load an odometry CSV -> (ts_ns[K], pos[K,3], quat_wxyz[K,4]) sorted by ts.

    Returns None if the file is absent (e.g. static RSU agents have no odometry).
    """
    fp = os.path.join(ds, "Odometry", f"mini_{n}", f"odometry_{agent}.csv")
    if not os.path.isfile(fp):
        return None
    ts, pos, quat = [], [], []
    with open(fp) as fh:
        for row in csv.DictReader(fh):
            ts.append(int(row['field.header.stamp']))
            pos.append([float(row['field.pose.pose.position.' + c]) for c in 'xyz'])
            quat.append([float(row['field.pose.pose.orientation.' + c]) for c in 'wxyz'])
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(pos)[o], np.array(quat)[o]


def nearest_pose(odom_entry, ts_ns):
    """(pos[3], quat_wxyz[4]) from an odom entry at the row nearest ts_ns."""
    t, pos, quat = odom_entry
    k = int(np.argmin(np.abs(t - ts_ns)))
    return pos[k], quat[k]


# --------------------------------------------------------------------------- #
# Frame assembly (native layout) + fusion.
# --------------------------------------------------------------------------- #
def frame_agents(ds, n, si, odom, load_points=False):
    """Assemble every agent present at sync step `si` of segment `mini_<n>`.

    Returns {agent: {"path": pcd, "T_to_top": 4x4[, "points": Nx4]}}.
    `odom` is {vehicle_agent: load_odom(...)} (RSU agents need no odometry).
    Agents whose vehicle odometry is missing are skipped.
    """
    out = {}
    for agent in AGENTS:
        fs = glob.glob(os.path.join(ds, "PointClouds", f"mini_{n}", f"{agent}_{si}_*.pcd"))
        if not fs:
            continue
        fp = fs[0]
        if agent in RSU_AGENTS:
            T = agent_to_top(agent)
        else:
            if odom.get(agent) is None:
                continue
            _, _, ts_ns = parse_pcd_name(fp)
            T = agent_to_top(agent, nearest_pose(odom[agent], ts_ns))
        entry = {"path": fp, "T_to_top": T}
        if load_points:
            entry["points"] = read_pcd(fp)
        out[agent] = entry
    return out


def fuse(agents):
    """Concatenate agents' points into the "top" ego frame -> (M, C) array.

    `agents` = {agent: {"points": NxC, "T_to_top": 4x4}} (as from frame_agents
    with load_points=True, or a dataloader Frame's agents dict).
    """
    clouds = []
    for a in agents.values():
        pts = a.get("points")
        if pts is not None and len(pts):
            clouds.append(transform_points(a["T_to_top"], pts))
    if not clouds:
        return np.zeros((0, 4))
    return np.vstack(clouds)


# --------------------------------------------------------------------------- #
# transforms.json (de)serialization for the anonymized test bundles.
# --------------------------------------------------------------------------- #
def transforms_to_json(transforms):
    """{agent: 4x4 ndarray} -> {agent: 4x4 row-major nested list} for JSON."""
    return {a: np.asarray(T).tolist() for a, T in transforms.items()}


def transforms_from_json(obj):
    """Inverse of transforms_to_json: parsed JSON -> {agent: 4x4 ndarray}."""
    return {a: np.array(m, dtype=np.float64) for a, m in obj.items()}


def write_transforms_json(path, transforms):
    with open(path, 'w') as fh:
        json.dump(transforms_to_json(transforms), fh, indent=1)


def read_transforms_json(path):
    with open(path) as fh:
        return transforms_from_json(json.load(fh))


# --------------------------------------------------------------------------- #
# Box geometry (shared by GT point-count filter + QA checks).
# --------------------------------------------------------------------------- #
def points_in_box_mask(P, b):
    """Boolean mask of points inside an oriented box b=[cx,cy,cz,dx,dy,dz,yaw,...].

    x,y are rotated by -yaw; z is axis-aligned. Matches the devkit's containment.
    """
    q = P[:, :3] - b[:3]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    xr = q[:, 0] * c - q[:, 1] * s
    yr = q[:, 0] * s + q[:, 1] * c
    return (np.abs(xr) <= b[3] / 2) & (np.abs(yr) <= b[4] / 2) & (np.abs(q[:, 2]) <= b[5] / 2)


def keep_by_point_count(boxes, agg_points, min_points=MIN_POINTS):
    """Mask of boxes containing > min_points aggregated lidar points."""
    if len(boxes) == 0 or agg_points is None or len(agg_points) == 0:
        return np.zeros(len(boxes), dtype=bool)
    near = agg_points[(np.abs(agg_points[:, 0]) <= 60) & (np.abs(agg_points[:, 1]) <= 60)]
    return np.array([points_in_box_mask(near, b).sum() > min_points for b in boxes])
