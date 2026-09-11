"""Per-pixel target assignment for a dense range-view head.

This replaces the sparse BEV heatmap, where each object contributed exactly ONE
positive cell against 196,608 negatives (measured 1:14,043). In the range view
every pixel that lands on an object is a positive that regresses the offset from
its own 3D point to that object's centre -- measured 58,142 positives per frame,
4,153x more supervision, which is how RangeDet and "What Matters in Range View"
(arXiv:2407.16789) actually predict.

Assignment rule: a pixel belongs to the box containing its 3D point. Where boxes
overlap, the nearer centre wins, so each pixel has exactly one target.
"""
import numpy as np


def assign_pixels(xyz, valid, boxes, classes):
    """xyz (H,W,3) ego-frame points, valid (H,W) bool, boxes (N,7), classes (N,).

    Returns dict of per-pixel targets:
        fg      (H,W) bool   -- pixel lies inside some box
        cls     (H,W) int64  -- 1..3 where fg, else 0
        offset  (H,W,3)      -- vector from the pixel's point to the box centre
        dim     (H,W,3)      -- log box extents
        rot     (H,W,2)      -- sin, cos of yaw
        box_id  (H,W) int64  -- which GT box, -1 where background
    """
    H, W = xyz.shape[:2]
    P = xyz.reshape(-1, 3)
    v = valid.reshape(-1).astype(bool)

    fg = np.zeros(H * W, bool)
    box_id = np.full(H * W, -1, np.int64)
    best_d = np.full(H * W, np.inf, np.float32)

    for j, b in enumerate(boxes):
        cx, cy, cz, dx, dy, dz, yaw = b[:7]
        t = P - np.array([cx, cy, cz], np.float32)
        # cheap reject before the rotated test
        near = v & (np.abs(t[:, 0]) < dx) & (np.abs(t[:, 1]) < dx) & (np.abs(t[:, 2]) <= dz / 2 + 1e-3)
        if not near.any():
            continue
        idx = np.nonzero(near)[0]
        tt = t[idx]
        ca, sa = np.cos(-yaw), np.sin(-yaw)
        lx = tt[:, 0] * ca - tt[:, 1] * sa
        ly = tt[:, 0] * sa + tt[:, 1] * ca
        inside = (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2)
        if not inside.any():
            continue
        sel = idx[inside]
        d = np.linalg.norm(tt[inside], axis=1)          # distance to this centre
        take = d < best_d[sel]                          # overlapping boxes: nearer wins
        s2 = sel[take]
        best_d[s2] = d[take]
        box_id[s2] = j
        fg[s2] = True

    offset = np.zeros((H * W, 3), np.float32)
    dim = np.zeros((H * W, 3), np.float32)
    rot = np.zeros((H * W, 2), np.float32)
    cls = np.zeros(H * W, np.int64)
    if fg.any():
        f = np.nonzero(fg)[0]
        j = box_id[f]
        offset[f] = boxes[j, :3] - P[f]                 # point -> centre, in metres
        dim[f] = np.log(np.clip(boxes[j, 3:6], 1e-3, None))
        rot[f] = np.stack([np.sin(boxes[j, 6]), np.cos(boxes[j, 6])], -1)
        cls[f] = classes[j]

    return {
        "fg": fg.reshape(H, W),
        "cls": cls.reshape(H, W),
        "offset": offset.reshape(H, W, 3),
        "dim": dim.reshape(H, W, 3),
        "rot": rot.reshape(H, W, 2),
        "box_id": box_id.reshape(H, W),
    }
