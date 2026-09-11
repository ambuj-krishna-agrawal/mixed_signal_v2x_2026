"""Range-image-native augmentation.

Grounded in LaserNet (arXiv:1903.08701), which augments range images by random
horizontal pixel shifts and flips specifically to avoid overfitting small
training sets, and in the standard 3D recipe (LidarAugment, arXiv:2210.13488)
of global rotation + flip.

The previous version here rotated/flipped/translated the ego-frame xyz while
leaving the lattice untouched. That is geometrically inconsistent: in a range
image the column index encodes azimuth, so rotating the coordinates without
rolling the columns breaks the column<->azimuth correspondence the 2D encoder
relies on, and a different correspondence is presented on every sample.

So every transform here acts on the lattice AND the coordinates together:

  * azimuth roll  -- np.roll on columns by k, paired with a rotation of
                     theta = 2*pi*k/W about the ego z axis. Verified sound:
                     the lattices wrap continuously (dome col0<->colW-1 gap
                     equals its adjacent-column gap; the moving vehicles show a
                     small seam from motion distortion, one column in 1024).
                     All sensors are 1024 columns after the build-time stride,
                     so one k serves all of them.
  * horizontal flip -- reverse column order, paired with negating y. Reversing
                     azimuth order maps phi -> -phi, which is exactly what
                     negating y does.
  * scale         -- angles are preserved by scaling, so the lattice is
                     untouched and this needs no counterpart.

Global translation is deliberately NOT included: shifting the cloud changes
every ray direction, so no roll of the lattice can restore consistency.
"""
import numpy as np


class RangeAugment:
    def __init__(self, roll=True, flip=True, max_roll_deg=45.0,
                 scale_range=(0.95, 1.05), seed=None,
                 agent_drop=0.0, beam_drop=0.0, pixel_drop=0.0,
                 keep_agents=("top",)):
        """agent_drop / beam_drop / pixel_drop: sensor-degradation augmentation.

        These target GENERALISATION, which is the measured remaining gap: this
        model scores ~0.79 on held-out scenes and 0.72 on the leaderboard, while
        a peer reports ~0.02 between the two. Every mechanical cause was ruled
        out (see FINDINGS.md), leaving robustness to sensor variation.

          agent_drop  probability of zeroing ONE whole non-ego agent's validity.
                      Standard in V2X collaborative perception (simulates a lost
                      link). Fusion is worth a lot here -- vehicle recall is
                      88.2% with all five agents vs 46.1% with `top` alone -- so
                      a model that has only ever seen five healthy sensors can be
                      brittle when one degrades.
          beam_drop   fraction of range-image ROWS zeroed. Native to the range
                      view (LaserNet, SalsaNext) and aimed at the real
                      heterogeneity: `top` is 64-beam, the others are 128.
          pixel_drop  fraction of individual valid pixels zeroed; simulates
                      sparser returns and partial occlusion.

        `keep_agents` is never dropped -- `top` defines the ego frame and the
        attention fusion reads out its slot.

        All three act on the VALIDITY MASK only, so the lattice geometry is
        untouched and the column<->azimuth correspondence is preserved.
        """
        # OpenPCDet's CenterPoint uses global rotation of +-22.5 deg (nuScenes)
        # or +-45 deg (Waymo), never a uniform full turn. A full turn also
        # destroys the strong prior that traffic aligns with a handful of road
        # directions at this fixed intersection, which is exactly the cue the
        # heading branch needs.
        self.max_roll_deg = max_roll_deg
        self.roll = roll
        self.flip = flip
        self.scale_range = scale_range
        self.agent_drop = agent_drop
        self.beam_drop = beam_drop
        self.pixel_drop = pixel_drop
        self.keep_agents = set(keep_agents)
        self.rng = np.random.default_rng(seed)

    def __call__(self, images, valid, boxes, hag=None, origins=None, bg=None):
        """images: {agent: (H,W,4)} ego-frame xyz+intensity
        valid:  {agent: (H,W)} uint8
        boxes:  (N,7) x y z dx dy dz yaw, ego frame, optionally (N,8) whose
                column 7 is the box centre's height above the mapped ground
        hag:    {agent: (H,W)} per-pixel height above ground, or None
        bg:     {agent: (H,W,2)} static-background residual + validity, or None.
                The residual is a RANGE difference, so it scales with the cloud;
                the validity flag does not.
        origins:{agent: (3,)} that agent's sensor position in ego, or None
        Returns new dicts / arrays; inputs are not modified.

        Ground-derived quantities (`hag`, boxes column 7) are computed against a
        map in the UNAUGMENTED frame, so they cannot be re-looked-up after a
        roll or flip. They are carried through instead: a rotation about ego z
        and a y-mirror both leave height above ground unchanged, and only the
        scale touches it. Sensor origins are ordinary points and transform
        exactly like the cloud.
        """
        boxes = boxes.copy()
        widths = {im.shape[1] for im in images.values()}
        # one shared column shift is only a single global rotation if every
        # sensor has the same width; fall back to no roll otherwise
        can_roll = self.roll and len(widths) == 1

        W = next(iter(widths)) if can_roll else None
        if can_roll:
            kmax = max(1, int(round(self.max_roll_deg / 360.0 * W)))
            k = int(self.rng.integers(-kmax, kmax + 1)) % W
        else:
            k = 0
        theta = 2.0 * np.pi * k / W if can_roll else 0.0
        do_flip = self.flip and self.rng.random() < 0.5
        scale = float(self.rng.uniform(*self.scale_range))

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)

        out_img, out_val = {}, {}
        out_hag, out_org, out_bg = {}, {}, {}
        for a, im in images.items():
            v = valid[a]
            im = im.copy()
            if do_flip:
                im = im[:, ::-1].copy()
                v = v[:, ::-1].copy()
                im[..., 1] *= -1.0                      # mirror y to match
            if k:
                im = np.roll(im, k, axis=1)
                v = np.roll(v, k, axis=1)
            xyz = im[..., :3]
            if k:
                xyz = xyz @ R.T                          # rotation matching the roll
            xyz = xyz * scale
            xyz[~v.astype(bool)] = 0.0                   # keep invalid pixels inert
            im[..., :3] = xyz
            out_img[a], out_val[a] = im, v
            if hag is not None and a in hag:
                h = hag[a]
                if do_flip:
                    h = h[:, ::-1].copy()
                if k:
                    h = np.roll(h, k, axis=1)
                out_hag[a] = (h * scale).astype(np.float32)
            if bg is not None and a in bg:
                g = bg[a]
                if do_flip:
                    g = g[:, ::-1].copy()
                if k:
                    g = np.roll(g, k, axis=1)
                g = g.astype(np.float32).copy()
                # bg channels: 0 range residual, 1 intensity residual, 2 valid
                g[..., 0] *= scale          # a range difference scales with the cloud
                g[..., 0] *= (g[..., 2] > 0)
                g[..., 1] *= (g[..., 2] > 0)   # intensity residual does NOT scale
                out_bg[a] = g
            if origins is not None and a in origins:
                o = origins[a].astype(np.float32).copy()
                if do_flip:
                    o[1] *= -1.0
                if k:
                    o = R @ o
                out_org[a] = o * scale

        if len(boxes):
            ctr = boxes[:, :3].astype(np.float32).copy()
            yaw = boxes[:, 6].astype(np.float32).copy()
            if do_flip:
                ctr[:, 1] *= -1.0
                yaw = -yaw
            if k:
                ctr = ctr @ R.T
                yaw = yaw + theta
            ctr = ctr * scale
            boxes[:, :3] = ctr
            boxes[:, 3:6] *= scale                       # extents scale, do not rotate
            boxes[:, 6] = np.arctan2(np.sin(yaw), np.cos(yaw))
            if boxes.shape[1] > 7:
                # column 7 is height above ground: invariant to roll and flip,
                # scales with the cloud
                boxes[:, 7] *= scale

        # --- sensor-degradation augmentation, on the validity mask only ---
        if self.agent_drop > 0 and self.rng.random() < self.agent_drop:
            cand = [a for a in out_val if a not in self.keep_agents]
            if cand:
                victim = cand[int(self.rng.integers(len(cand)))]
                out_val[victim] = np.zeros_like(out_val[victim])
        if self.beam_drop > 0:
            for a, v in out_val.items():
                if not v.any():
                    continue
                rows = self.rng.random(v.shape[0]) < self.beam_drop
                if rows.any():
                    v = v.copy(); v[rows] = 0; out_val[a] = v
        if self.pixel_drop > 0:
            for a, v in out_val.items():
                if not v.any():
                    continue
                m = self.rng.random(v.shape) < self.pixel_drop
                if m.any():
                    v = v.copy(); v[m] = 0; out_val[a] = v

        return out_img, out_val, boxes, out_hag, out_org, out_bg


# Backwards-compatible alias: dataset.py imports GlobalAugment.
GlobalAugment = RangeAugment
