"""Dataset over the cached range images, plus CenterPoint-style target encoding."""
import numpy as np
import re
import torch
from pathlib import Path
from torch.utils.data import Dataset

from augment import GlobalAugment
from model import (PC_RANGE, BEV_RES, OUT_STRIDE, PED_HIRES_STRIDE, NUM_CLASSES,
                   bev_dims)

AGENTS = ("top", "dome", "laser", "003", "004")

# Residual range images, following LiDAR-MOS (arXiv:2105.02446): the difference
# between the current range image and an earlier one exposes motion directly.
# LiDAR-MOS must first transform the previous scan by odometry and re-project it,
# because their sensor is vehicle-mounted. Ours is free for the two RSU lidars:
# measured lattice drift over 100ms on static background is 0.034m (top) and
# 0.050m (dome), i.e. the scan is phase-locked and cell (r,c) is the same ray
# frame to frame. The three vehicle lidars drift 0.38-0.40m because the sensor
# itself moves, so a lattice subtraction there would mix self-motion into the
# signal; they get a zeroed residual and a flag instead.
STATIC_AGENTS = ("top", "dome")
RESIDUAL_LAGS = (1, 5)          # ~0.1s and ~0.5s at 10Hz

# CenterPoint / OpenPCDet nuScenes configs use MIN_RADIUS: 2
MIN_GAUSSIAN_RADIUS = 2

# SECOND / OpenPCDet direction classifier: the regression loss below is made
# invariant to a 180-degree flip, so the sign has to be recovered separately.
# The pi/4 offset keeps the bin boundary away from the axis-aligned headings
# that dominate a road scene, where it would otherwise be maximally unstable.
DIR_OFFSET = np.pi / 4
DIR_NUM_BINS = 2


def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period


def direction_bin(yaw):
    """GT yaw -> {0, 1}: which half-turn the heading falls in."""
    period = 2 * np.pi / DIR_NUM_BINS
    shifted = limit_period(yaw - DIR_OFFSET, 0.0, 2 * np.pi)
    return int(np.floor(shifted / period)) % DIR_NUM_BINS


def gaussian_radius(h, w, min_overlap=0.7):
    """CornerNet radius: how far a predicted centre may drift and still keep
    >= min_overlap IoU. Uses the SMALLER quadratic root (original CornerNet),
    not CenterNet's (b + sq) / 2 variant.

    Measured, not assumed. Switching to the CenterNet form gave radii of
    13/6/4 cells for vehicle/bike/pedestrian; at our 0.4m output cells that is
    wider than the objects themselves (a bike spans ~1.7 cells). Held-out mAP
    after one epoch fell from 0.184 to 0.021, with Bike AP at exactly 0.000,
    while the TRAINING loss went DOWN (hm 1.111 -> 0.832): the softened region
    covered ~450 cells per object, so (1 - gt)^4 made them nearly free to get
    wrong and "predict low everywhere" became the cheap optimum.

    At our object scale f returns 0.2-0.5 for every class, so the caller's
    MIN_GAUSSIAN_RADIUS (2, per CenterPoint) governs in practice and the radius
    is effectively fixed. Do not swap the root without an A/B on held-out mAP
    measured at a matched, sufficient epoch count -- judging at epoch 1 is what
    produced the wrong conclusion the first time.
    """
    a1, b1 = 1, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / (2 * a1)

    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (-b3 + np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(hm, cx, cy, radius):
    d = 2 * radius + 1
    sigma = d / 6.0
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    H, W = hm.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    if r <= -l or b <= -t:
        return
    np.maximum(hm[cy - t:cy + b, cx - l:cx + r],
               g[radius - t:radius + b, radius - l:radius + r],
               out=hm[cy - t:cy + b, cx - l:cx + r])


def scene_of(path):
    return int(re.match(r"seq(\d+)_", Path(path).name).group(1))


class RangeViewDataset(Dataset):
    def __init__(self, cache_dir, max_objs=100, train=True, rv=False,
                 keep_scenes=None, drop_scenes=None, augment=False,
                 aug_kwargs=None, residuals=True, geometry=False,
                 ground_map='ground_map.npz', origins_index=None,
                 background=False, background_map='background.npz',
                 ped_hires=False, hires=None, gt_paste=None, gt_infill=None,
                 pseudo=None, pseudo_cache=None):
        """keep_scenes / drop_scenes: split by *scene*, never by frame.

        Frames within a 30s sequence are near-duplicates at 10Hz, so a random
        frame split would leak almost-identical frames into validation and
        report a meaninglessly optimistic score.
        """
        self.files = sorted(Path(cache_dir).glob("*.npz"))
        if keep_scenes is not None:
            self.files = [f for f in self.files if scene_of(f) in set(keep_scenes)]
        if drop_scenes is not None:
            self.files = [f for f in self.files if scene_of(f) not in set(drop_scenes)]
        # PSEUDO-LABELS on the unlabelled val scenes. Two tiers: positives are
        # merged into gt like the in-fill sidecar; "ignore" boxes are carried
        # with class c+100 so GlobalAugment transforms them with everything
        # else, then written into the heatmap as 0.999 -- which is neither a
        # positive (focal_loss selects positives with gt.eq(1) exactly) nor a
        # negative ((1-gt)^4 = 1e-12 kills the penalty). Without that band a
        # missed object would be supervised as background and the model would
        # be trained to suppress a detection it was getting right.
        self._pseudo_pos = self._pseudo_ign = None
        if pseudo:
            _p = np.load(pseudo, allow_pickle=True)
            _po = np.concatenate([[0], np.cumsum(_p['pos_counts'])])
            _io = np.concatenate([[0], np.cumsum(_p['ign_counts'])])
            nm_ = [str(x) for x in _p['names']]
            self._pseudo_pos = {n: _p['pos_boxes'][a:b] for n, a, b
                                in zip(nm_, _po[:-1], _po[1:])}
            self._pseudo_ign = {n: _p['ign_boxes'][a:b] for n, a, b
                                in zip(nm_, _io[:-1], _io[1:])}
            extra_files = sorted(Path(pseudo_cache or 'cache_val').glob("*.npz"))
            extra_files = [f for f in extra_files if f.stem in self._pseudo_pos]
            self.files = list(self.files) + extra_files
            print(f"pseudo-labels: {len(_p['pos_boxes'])} positives / "
                  f"{len(_p['ign_boxes'])} ignore over {len(extra_files)} frames "
                  f"(hi={float(_p['hi']):.2f} ign={float(_p['ign']):.2f})", flush=True)
        if not self.files:
            raise FileNotFoundError(f"no cached frames in {cache_dir} after filtering")
        self.max_objs = max_objs
        self.train = train
        # Geometry pack: height above ground (per pixel and per box) plus each
        # agent's sensor origin. Both are ground/pose derived and must be
        # resolved in the UNAUGMENTED frame, so they are computed here and
        # carried through GlobalAugment rather than looked up afterwards.
        self.geometry = geometry
        if geometry:
            gm = np.load(ground_map)
            self._g = gm["ground"].astype(np.float32)
            self._gcell = float(gm["cell"]); self._gR = float(gm["pc_range_xy"])
            self._gn = self._g.shape[0]
            if origins_index is None:
                origins_index = ("sensor_origins_test.npz"
                                 if "test" in str(cache_dir) else "sensor_origins.npz")
            oi = np.load(origins_index)
            self._oagents = [str(x) for x in oi["agents"]]
            self._orig = {str(n): o for n, o in zip(oi["names"], oi["origins"])}
        # Static per-pixel background range for the two fixed RSUs. Independent
        # of `geometry`: this is a range-image lookup (row,col)->range, not the
        # ground map's BEV (x,y)->z, and it captures ALL permanent structure.
        self.background = background
        if background:
            bgz = np.load(background_map)
            self._bg = {a: bgz[f"{a}_bg"].astype(np.float32)
                        for a in [str(x) for x in bgz["agents"]]}
            self._bgv = {a: bgz[f"{a}_bgvalid"].astype(bool)
                         for a in [str(x) for x in bgz["agents"]]}
            self._ibg = {a: bgz[f"{a}_ibg"].astype(np.float32)
                         for a in [str(x) for x in bgz["agents"]]
                         if f"{a}_ibg" in bgz.files}
        # Augment training only. Applying it to validation would make the val
        # loss/mAP measure a moving target and break comparability across epochs.
        self.aug = GlobalAugment(**(aug_kwargs or {})) if augment else None
        # GT-paste. Independent of `geometry`: it needs the sensor origins to
        # invert each agent's lattice, but not the ground map. Train only, for
        # the same reason augmentation is train only.
        self.paste = None
        self._infill = None
        if gt_infill:
            _z = np.load(gt_infill if isinstance(gt_infill, str) else 'gt_infill.npz',
                         allow_pickle=True)
            _off = np.concatenate([[0], np.cumsum(_z['counts'])])
            self._infill = {str(n): _z['boxes'][a:b] for n, a, b
                            in zip(_z['names'], _off[:-1], _off[1:])}
            print(f"label in-fill: {len(_z['boxes'])} boxes over "
                  f"{len(self._infill)} frames", flush=True)
        if gt_paste:
            from gt_paste import GTPaste
            oidx = origins_index or ("sensor_origins_test.npz"
                                     if "test" in str(cache_dir)
                                     else "sensor_origins.npz")
            oi = np.load(oidx)
            self._poagents = [str(x) for x in oi["agents"]]
            self._porig = {str(n): o for n, o in zip(oi["names"], oi["origins"])}
            self.paste = GTPaste(**(gt_paste if isinstance(gt_paste, dict) else {}))
        Hb, Wb = bev_dims()
        self.hm_h, self.hm_w = Hb // OUT_STRIDE, Wb // OUT_STRIDE
        # High-resolution PEDESTRIAN-ONLY heatmap. Measured motivation: 30.8% of
        # pedestrians have another pedestrian within 1.0 m and their recall is
        # 73.7%, against 98.9% at 1.0-1.5 m; for Vehicle and Bike the figure is
        # 0.0%, so neither needs this. At the coarse 0.4 m cell two pedestrians
        # 0.7 m apart are 1.75 cells apart and their Gaussians merge into one
        # blob, so the model is never taught there are two objects. At
        # PED_HIRES_STRIDE = 1 (0.2 m) the same pair is 3.5 cells apart.
        self.hires = tuple(sorted(hires)) if hires is not None else ((3,) if ped_hires else ())
        self.ped_hires = 3 in self.hires
        self.ph_h, self.ph_w = Hb // PED_HIRES_STRIDE, Wb // PED_HIRES_STRIDE
        # dense range-view targets (per pixel) instead of the sparse BEV heatmap
        self.rv = rv
        # Residual (temporal-difference) channels. Only `top` and `dome` can
        # carry them -- the vehicle lidars move, so a lattice subtraction there
        # would mix ego-motion into the signal. Set False to train a model that
        # needs no neighbouring frames at inference: the current leaderboard
        # split has consecutive frames, but the final `test/` split ships in a
        # different format and may not.
        self.residuals = residuals


    def __len__(self):
        return len(self.files)

    def _range_at(self, path):
        """(agent -> range image) for the STATIC agents of a cached frame."""
        out = {}
        if path is None or not path.exists():
            return out
        d = np.load(path)
        for a in STATIC_AGENTS:
            if f"{a}_img" in d.files:
                im = d[f"{a}_img"].astype(np.float32)
                v = d[f"{a}_valid"].astype(bool)
                r = np.linalg.norm(im[..., :3], axis=-1)
                out[a] = (r, v)
        return out

    def _neighbour(self, i, lag):
        """The cache file `lag` frames earlier in the SAME sequence, or None."""
        name = self.files[i].name
        seq, idx = name[3:-4].split("_idx")
        j = int(idx) - lag
        if j < 0:
            return None
        return self.files[i].parent / f"seq{seq}_idx{j}.npz"

    def _ground_at(self, x, y):
        ix = np.clip(((x + self._gR) / self._gcell).astype(np.int64), 0, self._gn - 1)
        iy = np.clip(((y + self._gR) / self._gcell).astype(np.int64), 0, self._gn - 1)
        return self._g[ix, iy]

    def __getitem__(self, i):
        d = np.load(self.files[i])
        present = [a for a in AGENTS if f"{a}_img" in d.files]
        images = {a: torch.from_numpy(d[f"{a}_img"].astype(np.float32)) for a in present}
        valid = {a: torch.from_numpy(d[f"{a}_valid"].astype(np.uint8)) for a in present}

        # append one residual channel per lag; zeros where it cannot be formed.
        # With residuals disabled the images stay 4-channel (xyz+intensity) and
        # model.build_input then emits 6 channels instead of 9 -- no neighbouring
        # frames are read at all, which also speeds up loading.
        if self.residuals:
            cur = {a: (np.linalg.norm(images[a][..., :3].numpy(), axis=-1),
                       valid[a].numpy().astype(bool))
                   for a in present if a in STATIC_AGENTS}
            prevs = [self._range_at(self._neighbour(i, L)) for L in RESIDUAL_LAGS]
            for a in present:
                H, W = images[a].shape[:2]
                res = np.zeros((H, W, len(RESIDUAL_LAGS)), np.float32)
                if a in cur:
                    r_t, v_t = cur[a]
                    for k, prev in enumerate(prevs):
                        if a in prev:
                            r_p, v_p = prev[a]
                            if r_p.shape == r_t.shape:
                                both = v_t & v_p
                                res[..., k] = np.where(both, r_t - r_p, 0.0)
                images[a] = torch.cat([images[a], torch.from_numpy(res)], dim=-1)

        boxes = d["gt_boxes"].astype(np.float32)     # (N,7) x y z dx dy dz yaw
        cls = d["gt_class"].astype(np.int64)         # 1..3
        # LABEL IN-FILL. The released annotation drops a tracked instance for a
        # frame or two while it is plainly still there -- measured 6.36% of
        # lifespan-frames in seq37, identical in the source JSON and the cache,
        # so it is the dataset's own gap and not our extraction (our box count
        # matches the published 124,668 to within one). Training on those frames
        # supervises "background" on a visible object, which is suppression
        # pressure against the recall this metric rewards. `gt_infill.npz`
        # restores only gaps of <=3 frames whose interpolated box still holds
        # >5 points; longer gaps are excluded because instances are linked by
        # size signature and two identical objects would merge into one.
        if self._infill is not None:
            extra = self._infill.get(self.files[i].stem)
            if extra is not None and len(extra):
                boxes = np.concatenate([boxes, extra[:, :7].astype(np.float32)])
                cls = np.concatenate([cls, extra[:, 7].astype(np.int64)])

        n_ign = 0
        if self._pseudo_pos is not None:
            _st = self.files[i].stem
            _pp = self._pseudo_pos.get(_st)
            if _pp is not None and len(_pp):
                boxes = np.concatenate([boxes, _pp[:, :7].astype(np.float32)])
                cls = np.concatenate([cls, _pp[:, 7].astype(np.int64)])
            _ii = self._pseudo_ign.get(_st)
            if _ii is not None and len(_ii):
                boxes = np.concatenate([boxes, _ii[:, :7].astype(np.float32)])
                # +100 so the heatmap loop skips them; augmentation still applies
                cls = np.concatenate([cls, _ii[:, 7].astype(np.int64) + 100])
                n_ign = len(_ii)

        if self.paste is not None:
            o_all = self._porig.get(self.files[i].stem)
            if o_all is not None:
                np_im = {a: t.numpy() for a, t in images.items()}
                np_v = {a: t.numpy() for a, t in valid.items()}
                po = {a: o_all[self._poagents.index(a)].astype(np.float32)
                      for a in present if a in self._poagents}
                np_im, np_v, boxes, cls = self.paste(np_im, np_v, boxes, cls, po)
                images = {a: torch.from_numpy(v) for a, v in np_im.items()}
                valid = {a: torch.from_numpy(v) for a, v in np_v.items()}

        bg = {}
        if self.background:
            for a in present:
                if a not in self._bg:
                    continue                      # vehicles have no fixed background
                im = images[a][..., :3].numpy()
                r = np.linalg.norm(im, axis=-1)
                ok = self._bgv[a] & valid[a].numpy().astype(bool)
                resid = np.where(ok, self._bg[a] - r, 0.0)
                # intensity residual: roadside LiDAR background models use BOTH
                # range and intensity (arXiv:2201.04756). A surface that is
                # closer than the background OR reflects differently is dynamic.
                ires = np.zeros_like(resid)
                if a in self._ibg:
                    ires = np.where(ok, images[a][..., 3].numpy() - self._ibg[a], 0.0)
                bg[a] = np.stack([resid, ires, ok.astype(np.float32)],
                                 -1).astype(np.float32)

        hag, origins = {}, {}
        if self.geometry:
            o_all = self._orig.get(self.files[i].stem)
            for a in present:
                xyz = images[a][..., :3].numpy()
                hag[a] = (xyz[..., 2] - self._ground_at(xyz[..., 0], xyz[..., 1])
                          ).astype(np.float32)
                if o_all is not None and a in self._oagents:
                    origins[a] = o_all[self._oagents.index(a)].astype(np.float32)
                else:
                    origins[a] = np.full(3, np.nan, np.float32)
            if len(boxes):
                bh = boxes[:, 2] - self._ground_at(boxes[:, 0], boxes[:, 1])
                boxes = np.column_stack([boxes, bh.astype(np.float32)])

        if self.aug is not None:
            np_images = {a: t.numpy() for a, t in images.items()}
            np_valid = {a: t.numpy() for a, t in valid.items()}
            np_images, np_valid, boxes, hag, origins, bg = self.aug(
                np_images, np_valid, boxes, hag or None, origins or None, bg or None)
            images = {a: torch.from_numpy(v) for a, v in np_images.items()}
            valid = {a: torch.from_numpy(v) for a, v in np_valid.items()}

        hm = np.zeros((NUM_CLASSES, self.hm_h, self.hm_w), np.float32)
        hm_hi = {c: np.zeros((1, self.ph_h, self.ph_w), np.float32) for c in self.hires}
        offset = np.zeros((self.max_objs, 2), np.float32)
        zs = np.zeros((self.max_objs, 1), np.float32)
        dims = np.zeros((self.max_objs, 3), np.float32)
        rots = np.zeros((self.max_objs, 2), np.float32)
        dirs = np.zeros((self.max_objs,), np.int64)
        clsidx = np.zeros((self.max_objs,), np.int64)   # 0..2, for IoU-aware targets
        inds = np.zeros((self.max_objs,), np.int64)
        mask = np.zeros((self.max_objs,), np.float32)

        cell = BEV_RES * OUT_STRIDE
        n = 0
        for b, c in zip(boxes, cls):
            if n >= self.max_objs or not (1 <= c <= NUM_CLASSES):
                continue
            fx = (b[0] - PC_RANGE[0]) / cell
            fy = (b[1] - PC_RANGE[1]) / cell
            cx, cy = int(fx), int(fy)
            if not (0 <= cx < self.hm_w and 0 <= cy < self.hm_h):
                continue                              # centre outside detection range
            # CenterPoint's published rule: radius = max(f(w,l), MIN_RADIUS),
            # with MIN_RADIUS = 2 (OpenPCDet nuScenes configs). Our objects are
            # small enough that f clamps for every class, so this is effectively
            # a fixed radius of 2 rather than the 1 used before.
            r = max(MIN_GAUSSIAN_RADIUS, int(gaussian_radius(b[4] / cell, b[3] / cell)))
            draw_gaussian(hm[c - 1], cx, cy, r)
            if c in hm_hi:                            # classes with a fine map
                pcell = BEV_RES * PED_HIRES_STRIDE
                px = int((b[0] - PC_RANGE[0]) / pcell)
                py = int((b[1] - PC_RANGE[1]) / pcell)
                if 0 <= px < self.ph_w and 0 <= py < self.ph_h:
                    # radius recomputed at the FINE cell size, same rule
                    pr = max(MIN_GAUSSIAN_RADIUS,
                             int(gaussian_radius(b[4] / pcell, b[3] / pcell)))
                    draw_gaussian(hm_hi[c][0], px, py, pr)
            offset[n] = [fx - cx, fy - cy]
            # ground-relative z: cuts the target's std by ~80% (0.766 -> 0.186
            # over 124,667 boxes), because most of the spread is the road
            # sloping across the scene, which a local patch cannot infer.
            zs[n] = b[7] if self.geometry else b[2]
            dims[n] = np.log(np.clip(b[3:6], 1e-3, None))
            rots[n] = [np.sin(b[6]), np.cos(b[6])]
            dirs[n] = direction_bin(float(b[6]))
            clsidx[n] = int(c) - 1
            inds[n] = cy * self.hm_w + cx
            mask[n] = 1.0
            n += 1

        if n_ign:
            cell_ = BEV_RES * OUT_STRIDE
            for b, c in zip(boxes, cls):
                if c <= 100:
                    continue
                ch = int(c) - 100 - 1
                if not (0 <= ch < NUM_CLASSES):
                    continue
                cx = int((b[0] - PC_RANGE[0]) / cell_)
                cy = int((b[1] - PC_RANGE[1]) / cell_)
                r = max(MIN_GAUSSIAN_RADIUS,
                        int(gaussian_radius(b[4] / cell_, b[3] / cell_)))
                y0, y1 = max(0, cy - r), min(self.hm_h, cy + r + 1)
                x0, x1 = max(0, cx - r), min(self.hm_w, cx + r + 1)
                if y0 >= y1 or x0 >= x1:
                    continue
                sub = hm[ch, y0:y1, x0:x1]
                np.copyto(sub, 0.999, where=(sub < 0.999))
            keep_ = cls <= 100
            boxes, cls = boxes[keep_], cls[keep_]

        rv_t = {}
        if self.rv:
            # Dense per-pixel targets: every pixel whose 3D point lies inside a
            # box regresses that box, so one frame supplies ~58k positives
            # instead of the ~14 the BEV heatmap gave (measured 4,153x).
            from rv_targets import assign_pixels
            for a in images:
                T = assign_pixels(images[a][..., :3].numpy(), valid[a].numpy(),
                                  boxes.astype(np.float32), cls.astype(np.int64))
                dbin = np.zeros(T["cls"].shape, np.int64)
                f = T["fg"]
                if f.any():
                    yaw = np.arctan2(T["rot"][..., 0][f], T["rot"][..., 1][f])
                    # must match direction_bin() above: floor(shifted/period) % 2.
                    # An earlier ">0" test was always true (limit_period returns
                    # [0, 2pi)), so every pixel was labelled bin 1 and the decode
                    # added pi to every heading.
                    shifted = limit_period(yaw - DIR_OFFSET, 0.0, 2 * np.pi)
                    dbin[f] = (np.floor(shifted / (2 * np.pi / DIR_NUM_BINS))
                               .astype(np.int64) % DIR_NUM_BINS)
                rv_t[a] = {
                    "fg": torch.from_numpy(T["fg"].astype(np.float32)),
                    "cls": torch.from_numpy(T["cls"]),
                    "offset": torch.from_numpy(T["offset"]),
                    "dim": torch.from_numpy(T["dim"]),
                    "rot": torch.from_numpy(T["rot"]),
                    "dir": torch.from_numpy(dbin),
                    "box_id": torch.from_numpy(T["box_id"]),
                }

        return {
            "images": images,
            "valid": valid,
            "hag": {a: torch.from_numpy(v) for a, v in hag.items()},
            "bg": {a: torch.from_numpy(np.ascontiguousarray(v)) for a, v in bg.items()},
            "origins": {a: torch.from_numpy(np.asarray(v, np.float32))
                        for a, v in origins.items()},
            "rv": rv_t,
            # raw boxes kept alongside the encoded targets: evaluation needs the
            # real geometry, not the heatmap encoding (and box count varies per
            # frame, so collate keeps these as a list rather than stacking)
            # sliced back to 7 columns: the geometry path appends a height-
            # above-ground column for the z target, and the official scorer
            # reshapes this array to (-1, 7)
            "gt_boxes_raw": torch.from_numpy(np.ascontiguousarray(boxes[:, :7])),
            "gt_class_raw": torch.from_numpy(cls),
            "hm": torch.from_numpy(hm),
            **{f"hm_hi_{c}": torch.from_numpy(v) for c, v in hm_hi.items()},
            **({"hm_ped_hi": torch.from_numpy(hm_hi[3])} if 3 in hm_hi else {}),
            "offset": torch.from_numpy(offset),
            "z": torch.from_numpy(zs),
            "dim": torch.from_numpy(dims),
            "rot": torch.from_numpy(rots),
            "dir": torch.from_numpy(dirs),
            "cls_idx": torch.from_numpy(clsidx),
            "ind": torch.from_numpy(inds),
            "mask": torch.from_numpy(mask),
        }


def collate(batch):
    """Agents present vary per frame (whole scenes lack the vehicles, e.g. mini_1).

    Missing agents are padded with zeros rather than intersected away: a zero
    pixel has range 0, so the model's validity mask drops it from the scatter and
    it contributes nothing. Intersecting would instead discard the vehicles from
    every batch that happened to mix a 3-agent scene with a 5-agent one.
    """
    shapes = {}
    for s in batch:
        for a, img in s["images"].items():
            shapes.setdefault(a, img.shape)

    images, valid = {}, {}
    for a, shape in sorted(shapes.items()):
        images[a] = torch.stack([
            s["images"][a] if a in s["images"] else torch.zeros(shape)
            for s in batch
        ])
        valid[a] = torch.stack([
            s["valid"][a] if a in s["valid"] else torch.zeros(shape[:2], dtype=torch.uint8)
            for s in batch
        ])
    out = {"images": images, "valid": valid}
    if batch[0].get("bg"):
        out["bg"] = {
            a: torch.stack([s["bg"][a] if a in s["bg"]
                            else torch.zeros(shape[0], shape[1], 3)
                            for s in batch])
            for a, shape in sorted(shapes.items())}
    if batch[0].get("hag"):
        out["hag"] = {
            a: torch.stack([s["hag"][a] if a in s["hag"] else torch.zeros(shape[:2])
                            for s in batch])
            for a, shape in sorted(shapes.items())}
        out["origins"] = {
            a: torch.stack([s["origins"][a] if a in s["origins"]
                            else torch.full((3,), float("nan"))
                            for s in batch])
            for a, shape in sorted(shapes.items())}
    if batch[0].get("rv"):
        # pad an absent agent with all-background targets; its validity mask is
        # already zero so nothing there contributes to the loss either way
        rv = {}
        for a, shape in sorted(shapes.items()):
            H, W = shape[:2]
            spec = {"fg": (torch.float32, (H, W)), "cls": (torch.int64, (H, W)),
                    "offset": (torch.float32, (H, W, 3)), "dim": (torch.float32, (H, W, 3)),
                    "rot": (torch.float32, (H, W, 2)), "dir": (torch.int64, (H, W)),
                    "box_id": (torch.int64, (H, W))}
            rv[a] = {k: torch.stack([
                s["rv"][a][k] if a in s["rv"] else
                (torch.full(sh, -1, dtype=dt) if k == "box_id" else torch.zeros(sh, dtype=dt))
                for s in batch]) for k, (dt, sh) in spec.items()}
        out["rv"] = rv
    for k in ("hm", "offset", "z", "dim", "rot", "dir", "cls_idx", "ind", "mask"):
        out[k] = torch.stack([s[k] for s in batch])
    for k in [k for k in batch[0] if k.startswith("hm_hi_") or k == "hm_ped_hi"]:
        out[k] = torch.stack([s[k] for s in batch])
    for k in ("gt_boxes_raw", "gt_class_raw"):
        out[k] = [s[k] for s in batch]      # ragged: one entry per frame
    return out
