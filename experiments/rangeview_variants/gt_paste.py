"""Same-frame GT-paste augmentation for the range view.

Motivation (measured on the training scenes): the classes are NOT imbalanced by
frame -- Vehicle/Bike/Pedestrian appear in 100.0/99.8/97.6 percent of frames --
so frame-level oversampling has nothing to oversample. The imbalance is in
INSTANCE DIVERSITY: 660 distinct Vehicles against 153 Bikes and 210 Pedestrians,
each recurring ~146 times because the annotations are tracked. Repeating frames
would show the model the same 153 bikes more often; pasting puts those instances
at new bearings, ranges and occlusion contexts, which is the standard remedy
(SECOND's gt_sampling, arXiv:1811.06621, carried into PointPillars/CenterPoint).

Why the paste is done this way
------------------------------
The obvious range-view shortcut -- copy the object's pixel patch to a new column,
the way RangeAugment rolls the whole image -- is only valid for a sensor AT the
ego origin, because a column shift is a rotation about ego z. Measured error of
that shortcut, |azimuth change about the agent - ego-z angle|, p90:

    top 0.00 deg | dome 0.77 deg | laser 149.8 | 003 165.0 | 004 145.6

and the far sensors are where the objects actually are: 003+004 contribute 86%
of Bike pixels and 99% of Vehicle pixels. So the patch-copy shortcut is unusable
and every pasted point is REPROJECTED into each agent's own lattice instead.

Reprojection cannot be analytic either: per-column median azimuth is linear at
360/1024 deg per column, but with residuals of 1.6-9.2 deg (004: 39.8), i.e. up
to ~25 columns of misplacement, against a bike only 2-4 columns wide. So the
lattice is inverted EMPIRICALLY per frame and agent -- pixel -> direction is
exact from the cached xyz, and `LatticeIndex` bins that map and reads it
backwards. This makes no assumption about lattice regularity at all.

Placement is a rotation about ego z, which keeps the object at its own distance
from the ego origin and on the same ground circle, and every pasted point is
occlusion-tested against the existing range image, so an object landing behind
a wall is rejected rather than pasted through it.
"""
import numpy as np

from model import PC_RANGE


def points_in_box(P, b):
    """Boolean mask of the points of `P` (N,3) inside the 7-dof box `b`."""
    q = P - b[:3]
    c, s = np.cos(-b[6]), np.sin(-b[6])
    lx = q[:, 0] * c - q[:, 1] * s
    ly = q[:, 0] * s + q[:, 1] * c
    return ((np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2)
            & (np.abs(q[:, 2]) <= b[5] / 2))


class LatticeIndex:
    """Empirical inverse of (row,col) -> ray direction for one agent, one frame.

    Forward is exact: the cache stores xyz per pixel, so subtracting the sensor
    origin gives that pixel's ray. Inverting it by binning (azimuth, elevation)
    costs one pass over the valid pixels and, unlike a fitted spherical model,
    absorbs whatever non-uniformity the real lattice has. Nearer surfaces are
    written last so a bin holding two returns keeps the one in front.
    """
    __slots__ = ("lut", "na", "ne", "el0", "esc", "H", "W")

    def __init__(self, xyz, valid, origin, na=2048, ne=256):
        H, W = valid.shape
        vy, vx = np.nonzero(valid)
        rel = xyz[vy, vx] - origin
        d = np.linalg.norm(rel, axis=1)
        keep = d > 1e-3
        vy, vx, rel, d = vy[keep], vx[keep], rel[keep], d[keep]
        az = np.arctan2(rel[:, 1], rel[:, 0])
        el = np.arcsin(np.clip(rel[:, 2] / d, -1.0, 1.0))
        self.na, self.ne = na, ne
        self.el0 = float(el.min())
        span = float(el.max()) - self.el0
        self.esc = ne / (span if span > 1e-9 else 1e-9)
        self.H, self.W = H, W
        lut = np.full(na * ne, -1, np.int32)
        order = np.argsort(-d)                     # far first; near overwrites
        lut[self._bin(az[order], el[order])] = (vy[order].astype(np.int32) * W
                                                + vx[order].astype(np.int32))
        self.lut = lut

    def _bin(self, az, el):
        ai = np.clip(((az + np.pi) * (self.na / (2 * np.pi))).astype(np.int64),
                     0, self.na - 1)
        ei = np.clip(((el - self.el0) * self.esc).astype(np.int64), 0, self.ne - 1)
        return ai * self.ne + ei

    def project(self, pts, origin):
        """(flat pixel index or -1, range) for each point."""
        rel = pts - origin
        d = np.linalg.norm(rel, axis=1)
        dd = np.maximum(d, 1e-6)
        az = np.arctan2(rel[:, 1], rel[:, 0])
        el = np.arcsin(np.clip(rel[:, 2] / dd, -1.0, 1.0))
        ai = np.clip(((az + np.pi) * (self.na / (2 * np.pi))).astype(np.int64),
                     0, self.na - 1)
        ei = np.clip(((el - self.el0) * self.esc).astype(np.int64), 0, self.ne - 1)
        flat = self.lut[ai * self.ne + ei].copy()
        for da, de in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)):
            miss = np.nonzero(flat < 0)[0]
            if miss.size == 0:
                break
            aa = np.clip(ai[miss] + da, 0, self.na - 1)
            ee = np.clip(ei[miss] + de, 0, self.ne - 1)
            cand = self.lut[aa * self.ne + ee]
            hit = cand >= 0
            flat[miss[hit]] = cand[hit]
        return flat, d


def _rot_box(b, th, pivot):
    """Rotate a box by `th` about the VERTICAL axis through `pivot`.

    The pivot is the origin of the sensor that contributes most of the object's
    points, so that sensor's range to the object -- and therefore the angular
    size and sampling density it would observe -- is preserved exactly. Rotating
    about ego z instead changes the range to every off-centre sensor, which is
    what made the density guard reject most placements. z is untouched either
    way, so height above ground (and any extra box column carrying it) is
    unchanged.
    """
    out = b.copy()
    c, s = np.cos(th), np.sin(th)
    dx, dy = b[0] - pivot[0], b[1] - pivot[1]
    out[0] = pivot[0] + c * dx - s * dy
    out[1] = pivot[1] + s * dx + c * dy
    out[6] = b[6] + th
    return out


class GTPaste:
    """Duplicate objects of the chosen classes at new bearings within the frame.

    Args
      classes      class ids eligible to be pasted (1 Vehicle, 2 Bike, 3 Ped)
      n_max        maximum successful pastes per frame
      max_donors   how many candidate objects to examine per frame (cost guard)
      min_px       an object must own at least this many pixels to be worth copying
      min_free     fraction of its points that must land unoccluded, i.e. the
                   object has to fit in genuinely free space or it is rejected
      range_ratio  bound on new_range/old_range for the SECONDARY sensors. The
                   pivot sensor is exact by construction; the others still see
                   the object move, and reprojection gets that geometry right
                   but cannot invent points, so an object moved much closer than
                   it was scanned would reproject full of holes.
      sep_margin   extra clearance, in metres, between the pasted box and every
                   existing box (circumscribed-circle test)
    """

    def __init__(self, classes=(2,), n_max=3, tries=16, min_px=30,
                 min_free=0.75, range_ratio=(0.6, 1.6), sep_margin=1.0,
                 min_shift_deg=10.0, max_shift_deg=180.0, max_donors=8,
                 minor_share=0.10, ground_map="ground_map.npz", max_dz=1.5,
                 seed=None):
        self.classes = tuple(classes)
        self.n_max = int(n_max)
        self.tries = int(tries)
        self.min_px = int(min_px)
        self.min_free = float(min_free)
        self.rlo, self.rhi = range_ratio
        self.sep_margin = float(sep_margin)
        self.min_shift = np.radians(min_shift_deg)
        self.max_shift = np.radians(max_shift_deg)
        self.max_donors = int(max_donors)
        self.minor_share = float(minor_share)
        self.rng = np.random.default_rng(seed)
        self.max_dz = float(max_dz)
        # Ground snapping. A rotation about a vertical axis preserves z, but the
        # ground does not: measured on pasted boxes without this, 21% of Bikes
        # and 50% of Vehicles ended up more than 0.5 m off the surface, against
        # 0% of the real GT boxes (real clearance is median 0.02 m, p5/p95
        # -0.11/+0.17). A floating object is a wrong training example, so the
        # object is re-seated by the ground difference between the two spots.
        self._g = None
        try:
            gm = np.load(ground_map)
            self._g = gm["ground"].astype(np.float32)
            self._gcell = float(gm["cell"])
            self._gR = float(gm["pc_range_xy"])
            self._gn = self._g.shape[0]
        except (FileNotFoundError, KeyError):
            pass
        self.stats = {"tried": 0, "pasted": 0, "rej_free": 0, "rej_ground": 0,
                      "rej_overlap": 0, "rej_range": 0, "rej_px": 0}

    def _ground(self, x, y):
        ix = int(np.clip((x + self._gR) / self._gcell, 0, self._gn - 1))
        iy = int(np.clip((y + self._gR) / self._gcell, 0, self._gn - 1))
        return float(self._g[ix, iy])

    def __call__(self, images, valid, boxes, cls, origins):
        """images {a:(H,W,C)} float32 ego xyz+..., valid {a:(H,W)} uint8,
        boxes (N,7+) float32, cls (N,), origins {a:(3,)}.
        Returns (images, valid, boxes, cls); inputs are modified in place."""
        agents = [a for a in images
                  if a in origins and np.all(np.isfinite(origins[a]))]
        cand = [i for i, c in enumerate(cls)
                if int(c) in self.classes and len(boxes[i]) >= 7]
        if not agents or not cand:
            return images, valid, boxes, cls

        # Per-agent point set and range map of the CURRENT scene, kept up to
        # date as we paste. Pixels sitting exactly at the sensor origin are
        # no-returns that the cache still marks valid (20-22% of 003/004); they
        # carry no surface and must not occlude anything.
        pts, pyx, rmap, lat = {}, {}, {}, {}
        for a in agents:
            v = valid[a].astype(bool)
            xyz = images[a][..., :3]
            r = np.full(v.shape, np.inf, np.float32)
            yy, xx = np.nonzero(v)
            P = xyz[yy, xx]
            dd = np.linalg.norm(P - origins[a], axis=1)
            real = dd > 1e-3
            r[yy[real], xx[real]] = dd[real]
            pts[a], pyx[a], rmap[a] = P, (yy, xx), r

        def _lat(a):
            """Inverting a lattice costs ~14 ms, so do it only for agents that
            actually see one of the donors -- `laser` contributes 0.0% of the
            object pixels of every class and would never be worth building.
            Built on demand, so a LUT made after an earlier paste also sees the
            pixels that paste wrote, keeping the occlusion test consistent."""
            L = lat.get(a)
            if L is None:
                L = lat[a] = LatticeIndex(images[a][..., :3],
                                          valid[a].astype(bool), origins[a])
            return L

        self.rng.shuffle(cand)
        new_boxes, new_cls = [], []
        for bi in cand[:self.max_donors]:
            if len(new_boxes) >= self.n_max:
                break
            b0 = boxes[bi].astype(np.float64)
            rad0 = 0.5 * float(np.hypot(b0[3], b0[4])) + 0.5 * float(b0[5])
            src, npx = {}, 0
            for a in agents:
                P = pts[a]
                near = (np.abs(P[:, 0] - b0[0]) <= rad0)   # cheap prefilter
                near &= (np.abs(P[:, 1] - b0[1]) <= rad0)
                near &= (np.abs(P[:, 2] - b0[2]) <= rad0)
                if not near.any():
                    continue
                sub = P[near]
                m = points_in_box(sub, b0)
                if m.any():
                    yy, xx = pyx[a]
                    src[a] = (yy[near][m], xx[near][m], sub[m])
                    npx += int(m.sum())
            self.stats["tried"] += 1
            if npx < self.min_px:
                self.stats["rej_px"] += 1
                continue
            # a sensor holding a sliver of the object is not worth a LUT build
            src = {a: v for a, v in src.items() if len(v[2]) >= 0.02 * npx}
            npx = sum(len(v[2]) for v in src.values())
            # pivot on the sensor that sees the object best
            pivot_a = max(src, key=lambda a: len(src[a][2]))
            pivot = origins[pivot_a]
            minor = [a for a in src
                     if a != pivot_a and len(src[a][2]) >= self.minor_share * npx]

            for _ in range(self.tries):
                th = float(self.rng.uniform(self.min_shift, self.max_shift))
                if self.rng.random() < 0.5:
                    th = -th
                nb = _rot_box(b0, th, pivot)
                if not (PC_RANGE[0] < nb[0] < PC_RANGE[3]
                        and PC_RANGE[1] < nb[1] < PC_RANGE[4]):
                    continue
                dz = 0.0
                if self._g is not None:
                    dz = self._ground(nb[0], nb[1]) - self._ground(b0[0], b0[1])
                    if abs(dz) > self.max_dz:
                        # a metre and a half of ground change is a roof or an
                        # embankment, not a place this object belongs
                        self.stats["rej_ground"] += 1
                        continue
                    nb[2] += dz
                rad = 0.5 * float(np.hypot(nb[3], nb[4]))
                ok = True
                for ob in list(boxes) + new_boxes:
                    orad = 0.5 * float(np.hypot(ob[3], ob[4]))
                    if (np.hypot(nb[0] - ob[0], nb[1] - ob[1])
                            < rad + orad + self.sep_margin):
                        ok = False
                        break
                if not ok:
                    self.stats["rej_overlap"] += 1
                    continue
                for a in minor:          # pivot sensor is exact by construction
                    r0 = np.linalg.norm(b0[:3] - origins[a])
                    r1 = np.linalg.norm(nb[:3] - origins[a])
                    if not (self.rlo <= r1 / max(r0, 1e-6) <= self.rhi):
                        ok = False
                        break
                if not ok:
                    self.stats["rej_range"] += 1
                    continue

                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                plan, seen, vis = [], 0, 0
                for a, (yy, xx, P) in src.items():
                    Q = (P - pivot) @ R.T + pivot
                    Q[:, 2] += dz                    # re-seat on the new ground
                    L = _lat(a)
                    flat, d = L.project(Q, origins[a])
                    hit = flat >= 0
                    seen += len(Q)
                    if not hit.any():
                        continue
                    fy, fx = flat[hit] // L.W, flat[hit] % L.W
                    # visible only if IN FRONT of what the sensor already sees
                    free = d[hit] < rmap[a][fy, fx] - 0.10
                    vis += int(free.sum())
                    if free.any():
                        plan.append((a, yy[hit][free], xx[hit][free],
                                     fy[free], fx[free], Q[hit][free], d[hit][free]))
                if seen == 0 or vis / seen < self.min_free:
                    self.stats["rej_free"] += 1
                    continue

                for a, sy, sx, ty, tx, Q, d in plan:
                    order = np.argsort(-d)     # nearest wins a shared pixel
                    sy, sx, ty, tx, Q, d = (sy[order], sx[order], ty[order],
                                            tx[order], Q[order], d[order])
                    images[a][ty, tx] = images[a][sy, sx]   # carry every channel
                    images[a][ty, tx, :3] = Q.astype(images[a].dtype)
                    valid[a][ty, tx] = 1
                    rmap[a][ty, tx] = d.astype(np.float32)
                new_boxes.append(nb.astype(boxes.dtype))
                new_cls.append(int(cls[bi]))
                self.stats["pasted"] += 1
                break

        if new_boxes:
            boxes = np.concatenate([boxes, np.stack(new_boxes)], 0)
            cls = np.concatenate([cls, np.array(new_cls, cls.dtype)], 0)
        return images, valid, boxes, cls
