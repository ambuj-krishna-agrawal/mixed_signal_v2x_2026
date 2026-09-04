"""
RangeViewFusionNet: per-sensor range-view encoding, fused by scattering into a
shared BEV grid, with a CenterPoint-style detection head.

Shape of the idea:

    5 range images (native lattice, free)
        -> shallow full-res 2D encoder, weights shared across sensors
    per-pixel features, each with a known ego-frame (x, y, z)
        -> scatter-max into ONE shared BEV grid
    fused BEV                      <- fusion IS the scatter; no attention needed
        -> small BEV backbone + centre head
    3D boxes in the `top` ego frame

Two properties this buys us:

* Variable agent count is free. Agents absent from a frame (the two vehicles
  are missing in 21.4% of frames, and the devkit also drops agents beyond a 70m
  comm range) simply contribute nothing to the scatter. There is no record_len
  bookkeeping and no pairwise transform matrices.
* No occlusion loss. Each sensor keeps its own lattice, so a return that one
  sensor sees behind an obstruction is never overwritten by another sensor's
  surface -- which is exactly what re-projecting a *fused* cloud to a single
  origin would have thrown away.

The encoder deliberately does NOT downsample. Every pixel keeps a feature and a
known 3D position, so the lift stays a dense, exact scatter; context is added
later by the BEV backbone instead.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from rangeview.compat import scatter_max_

from rangeview.models.norm import make_norm

from rangeview.models.agent_fusion import AgentAttentionFusion
from rangeview.models.meta_kernel import MetaKernel

# Detection range, matched to the PointPillars baseline so the two are comparable.
PC_RANGE = (-51.2, -51.2, -6.0, 51.2, 51.2, 1.0)
BEV_RES = 0.2                      # -> 512 x 512 grid
OUT_STRIDE = 2                     # head predicts at 256 x 256 (0.4 m / cell)
NUM_CLASSES = 3                    # 1=Vehicle, 2=Bike, 3=Pedestrian

# Per-channel input statistics measured over the cache (valid pixels only).
# RangeNet++/SalsaNext -- the backbones LiDAR-MOS builds on -- standardise range
# image inputs this way, and it matters: the previous ad-hoc divisors (/50 for
# xyz, /8 for log intensity) delivered z at std 0.044 and intensity at std 0.008
# against x/y at 0.16, so height arrived 3.7x weaker and intensity 20x weaker
# than position. BatchNorm cannot repair that -- it normalises output channels,
# so the relative input weighting is already baked into the first convolution.
INPUT_MEAN = (5.775, -6.728, -2.030, 0.081, 31.511)     # x, y, z, log1p(int), range
INPUT_STD = (37.086, 23.590, 2.924, 0.140, 32.096)

# Two residual range-image channels (LiDAR-MOS), populated for the static RSU
# lidars only. Residuals are already metric differences centred on zero, so they
# need scaling but not mean-subtraction.
NUM_RESIDUALS = 2
RESIDUAL_STD = 0.30
IN_CHANNELS = 6 + NUM_RESIDUALS
# Geometry pack appended by build_input when enabled: sensor-relative range and
# unit ray (4), height above the mapped ground (1), surface normal (3).
GEOM_CHANNELS = 8
HAG_STD = 2.0                      # height above ground is ~[-0.5, 3] m


def bev_dims():
    W = int(round((PC_RANGE[3] - PC_RANGE[0]) / BEV_RES))
    H = int(round((PC_RANGE[4] - PC_RANGE[1]) / BEV_RES))
    return H, W


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, norm=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
            make_norm(cout, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RangeEncoder(nn.Module):
    """Shallow, full-resolution. Shared across sensors -- convs don't care that
    `top` is 64 rows and the others are 128.

    The second stage is a RangeDet Meta-Kernel rather than a plain conv: in a
    range image the 3D offset to a pixel's neighbours varies by orders of
    magnitude with distance, so a fixed kernel cannot express the local
    geometry. Meta-Kernel derives its weights from the neighbours' relative
    Cartesian coordinates instead. Placed early, as in the paper.
    """

    def __init__(self, cin=IN_CHANNELS + 1, cout=32, use_meta=True):
        super().__init__()
        self.use_meta = use_meta
        self.stem = ConvBlock(cin, 32)
        self.meta = MetaKernel(32, 32) if use_meta else ConvBlock(32, 32)
        self.head = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, cout))

    def forward(self, x, xyz=None, valid=None):
        h = self.stem(x)
        h = self.meta(h, xyz, valid) if self.use_meta else self.meta(h)
        return self.head(h)


class BEVBackbone(nn.Module):
    """Multi-scale BEV backbone, matching the structure of the OpenCOOD baseline
    that scores 0.77 on this dataset (`msig_pillar_v2xvit.yaml`).

    The previous version was 0.475M parameters in two single-scale blocks. The
    reference `base_bev_backbone` is layer_nums [3,5,8] over filters
    [64,128,256] with upsample_strides [1,2,4] concatenated to 384 channels --
    7.46M, i.e. 15.7x larger, and its backbone alone is 7.7x this entire
    network. Measured error profile said we were not recall- or ranking-limited
    but capacity-limited: 0.85 on scenes seen-adjacent, 0.62 on genuinely new
    ones, which is what too little capacity to learn session-invariant features
    looks like.

    Two properties of the reference that the old version lacked:
      * THREE scales, upsampled to a common stride and concatenated, so one head
        sees both vehicle-scale context and pedestrian-scale detail.
      * real depth (3/5/8 convs per block) rather than 2.

    Defaults match the reference exactly: layer_nums [3,5,8], filters
    [64,128,256]. Measured at batch 2 in bf16: 7.51M params, 4.58 GB of 5.64,
    1.09 it/s -- barely slower than a shallower variant, because runtime is
    dominated by the per-agent range encoder, not this backbone.
    """

    def __init__(self, cin=32, cout=128, layers=(3, 5, 8), width=(64, 128, 256),
                 up=128, ms_fusion=True):
        super().__init__()
        w1, w2, w3 = width
        n1, n2, n3 = layers

        def block(ci, co, n):
            m = [ConvBlock(ci, co, s=2)]
            m += [ConvBlock(co, co) for _ in range(n)]
            return nn.Sequential(*m)

        self.down1 = block(cin, w1, n1)      # stride 2
        self.down2 = block(w1, w2, n2)       # stride 4
        self.down3 = block(w2, w3, n3)       # stride 8
        # bring every scale back to stride 2 and concatenate (upsample_strides
        # [1,2,4] in the reference config)
        self.up1 = ConvBlock(w1, up, k=1)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(w2, up, 2, 2, bias=False),
            make_norm(up), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(w3, up, 4, 4, bias=False),
            make_norm(up), nn.ReLU(inplace=True))
        self.fuse = ConvBlock(up * 3, cout)

        # MULTI-SCALE CROSS-AGENT FUSION (CoGoal3D's 3D-AGF idea, arXiv:2607.19036,
        # ablated there at +2.20 AP@0.5). Their fusion runs at every backbone
        # scale; ours previously ran only at stride 2, so every fusion decision
        # was made at one spatial resolution -- fine for a pedestrian, too local
        # for a vehicle.
        #
        # Running the real down2/down3 per agent would cost only ~285 MB extra
        # (measured), but they hold 5.83M of 7.51M params, so the COMPUTE would
        # go up ~5x (1.0 -> ~0.35 it/s). Instead a THIN per-agent pyramid carries
        # each agent to stride 4 and 8 cheaply, fusion happens there, and the
        # fused result is injected into the trunk. The heavy blocks still run
        # once, at batch B.
        self.ms_fusion = ms_fusion
        if ms_fusion:
            t2, t3 = w2 // 4, w3 // 8            # thin per-agent channels
            self.thin2 = ConvBlock(w1, t2, s=2)  # stride 2 -> 4, per agent
            self.thin3 = ConvBlock(t2, t3, s=2)  # stride 4 -> 8, per agent
            self.fuse2 = AgentAttentionFusion(t2)
            self.fuse3 = AgentAttentionFusion(t3)
            self.inject2 = ConvBlock(w2 + t2, w2, k=1)
            self.inject3 = ConvBlock(w3 + t3, w3, k=1)
        self.agent_fusion = AgentAttentionFusion(w1)

    def forward(self, x, agent_mask=None, n_agents=1, agent_names=None):
        """x is (B*N, C, H, W) when fusing across agents, else (B, C, H, W).

        Fusion stays after down1, at stride 2. The reference fuses after the
        whole backbone, but running three blocks per agent would not fit 6 GB;
        fusing early keeps blocks 2 and 3 at batch B instead of B*N.
        """
        d1 = self.down1(x)                       # stride 2, still per agent
        ms2 = ms3 = None
        if agent_mask is not None:
            BN, C, H, W = d1.shape
            B = BN // n_agents
            if self.ms_fusion:
                # thin per-agent pyramid, fused at stride 4 and 8 BEFORE the
                # heavy trunk collapses the agent axis
                t2 = self.thin2(d1)
                t3 = self.thin3(t2)
                c2, h2, w2_ = t2.shape[1:]
                c3, h3, w3_ = t3.shape[1:]
                ms2 = self.fuse2(t2.view(B, n_agents, c2, h2, w2_), agent_mask,
                                 agent_names=agent_names)
                ms3 = self.fuse3(t3.view(B, n_agents, c3, h3, w3_), agent_mask,
                                 agent_names=agent_names)
            d1 = self.agent_fusion(d1.view(B, n_agents, C, H, W), agent_mask,
                                   agent_names=agent_names)
        d2 = self.down2(d1)                      # stride 4, batch B
        if ms2 is not None:
            ms2 = ms2[..., :d2.shape[-2], :d2.shape[-1]]
            d2 = self.inject2(torch.cat([d2, ms2], dim=1))
        d3 = self.down3(d2)                      # stride 8, batch B
        if ms3 is not None:
            ms3 = ms3[..., :d3.shape[-2], :d3.shape[-1]]
            d3 = self.inject3(torch.cat([d3, ms3], dim=1))
        u = [self.up1(d1), self.up2(d2), self.up3(d3)]
        u = [t[..., :d1.shape[-2], :d1.shape[-1]] for t in u]
        return self.fuse(torch.cat(u, dim=1))


class CenterHead(nn.Module):
    """Anchor-free. Avoids the 3-class anchor matching the devkit does awkwardly,
    and makes target assignment a simple per-centre lookup."""

    def __init__(self, cin=128, use_iou=True):
        super().__init__()

        def branch(cout, bias=None):
            layers = [ConvBlock(cin, 64), nn.Conv2d(64, cout, 1)]
            # Small final weights so the bias prior actually governs the initial
            # logits. With default init the conv's own variance swamps the prior
            # and the focal loss starts ~8x higher than it should.
            nn.init.normal_(layers[-1].weight, std=0.01)
            nn.init.constant_(layers[-1].bias, 0.0 if bias is None else bias)
            return nn.Sequential(*layers)

        self.hm = branch(NUM_CLASSES, bias=-2.19)   # focal-loss prior
        self.offset = branch(2)                     # sub-cell dx, dy
        self.z = branch(1)
        self.dim = branch(3)                        # log l, w, h
        self.rot = branch(2)                        # sin, cos (axis only)
        # SECOND-style direction classifier. The rotation regression below is
        # deliberately invariant to a 180-degree flip, so the sign is recovered
        # here instead. Measured motivation: without it the (sin,cos) vector
        # collapsed to norm 0.608 and 15.6% of headings were flipped, because
        # L1 was averaging two opposite, equally-valid answers.
        self.dir = branch(2)                        # 2 direction bins
        # IoU-aware confidence (AFDetV2 arXiv:2112.09205, CIA-SSD
        # arXiv:2012.03015). Predicts the BEV IoU this cell's decoded box will
        # have with its ground truth; at inference the score is rectified as
        #     f = score^(1-alpha) * iou_pred^alpha
        # Measured headroom with ORACLE IoU on held-out scenes, re-tuning the
        # per-class threshold at each alpha: mAP 0.7868 -> 0.8953 (alpha 0.5)
        # -> 0.9159 (alpha 1.0). This is the only lever of the right magnitude
        # left; classification score alone cannot tell a well-localised box from
        # a badly-localised one, and the devkit's frame-ordered AP punishes the
        # bad ones immediately.
        self.iou = branch(1) if use_iou else None

    def forward(self, x):
        return {
            "hm": self.hm(x), "offset": self.offset(x), "z": self.z(x),
            "dim": self.dim(x), "rot": self.rot(x), "dir": self.dir(x),
            **({"iou": self.iou(x)} if self.iou is not None else {}),
        }


class RangeViewFusionNet(nn.Module):
    def __init__(self, feat=32, use_meta=True, grad_ckpt=True, pillar_feat=32,
                 use_vfe=True, use_residuals=True, use_iou_head=True,
                 ms_fusion=True, use_geometry=False):
        super().__init__()
        # The encoder runs once per sensor (up to 5x) and its Meta-Kernel unfold
        # dominates memory. Checkpointing drops its intermediate activations and
        # recomputes them in backward, which is what makes per-agent BEV grids +
        # attention fusion fit in 6GB. Costs roughly 30% throughput.
        self.grad_ckpt = grad_ckpt
        # build_input emits 6 channels without residuals (x, y, z, log-intensity,
        # range, validity) and 9 with them (+2 residual lags +1 availability
        # flag). The encoder must be sized to match, so a residual-free model is
        # a genuinely different architecture -- not the same network with two
        # inputs zeroed, which is why zeroing them at inference is NOT a valid
        # measurement of what they contribute.
        self.use_residuals = use_residuals
        self.use_geometry = use_geometry
        cin = (IN_CHANNELS + 1) if use_residuals else 6
        if use_geometry:
            cin += GEOM_CHANNELS
        self.encoder = RangeEncoder(cin=cin, cout=feat, use_meta=use_meta)
        # Per-agent conditioning on the SHARED encoder's output. This is exactly
        # where the BatchNorm defect lived: one encoder run separately per agent,
        # with BN silently normalising each by its own batch statistics in
        # training and by one global average at inference (measured 13x loss
        # gap). GroupNorm removed the mismatch but is agent-agnostic; this gives
        # the network an explicit, learned handle on which sensor it is looking
        # at. Identity at initialisation ((1 + 0) * f + 0).
        from rangeview.models.agent_fusion import AGENT_ORDER
        self.agent_order = AGENT_ORDER
        self.agent_gamma = nn.Parameter(torch.zeros(len(AGENT_ORDER), feat))
        self.agent_beta = nn.Parameter(torch.zeros(len(AGENT_ORDER), feat))
        # use_vfe=False is the shipped recipe (PillarVFE was a wash and is not
        # in this tree). Import only when requested so --no-vfe still constructs.
        if use_vfe:
            try:
                from pillar_vfe import PillarVFE
            except ImportError as e:
                raise ImportError(
                    "PillarVFE is not included in this repository. Use --no-vfe "
                    "(the documented winning recipe). The VFE path was measured "
                    "as a wash and is not required for the 0.84+ models."
                ) from e
            self.vfe = PillarVFE(cin=feat, cout=pillar_feat)
        else:
            self.vfe = None
        self.bev_ch = pillar_feat if use_vfe else feat
        self.backbone = BEVBackbone(cin=self.bev_ch, cout=128,
                                    ms_fusion=ms_fusion)
        self.head = CenterHead(128, use_iou=use_iou_head)
        self.feat = feat

    @staticmethod
    def build_input(img, valid, hag=None, origin=None):
        """img: (B, H, W, 4) ego-frame xyz + intensity, valid: (B, H, W) uint8
        -> (B, 6, H, W) as [x, y, z, log intensity, radial range, validity].

        `valid` comes from the cache, where it was computed in body frame; it
        cannot be re-derived here, because a no-return sits at its own sensor's
        origin in ego coordinates, not at range zero.

        With `hag` and `origin` supplied, eight geometry channels are appended:
        [r_sensor, ux, uy, uz, hag, nx, ny, nz]. The plain `range` channel above
        is measured from the EGO origin while the lattice is this agent's own
        beam x azimuth grid, so for every agent except `top` a range edge stops
        meaning an object edge; r_sensor and the unit ray restore that. See
        geometry.py.
        """
        xyz = img[..., :3]
        inten = img[..., 3:4]
        r = torch.linalg.norm(xyz, dim=-1, keepdim=True)
        v = valid.unsqueeze(-1).float()

        raw = torch.cat([xyz, torch.log1p(inten.clamp(min=0)), r], dim=-1)
        mean = torch.tensor(INPUT_MEAN, device=raw.device, dtype=raw.dtype)
        std = torch.tensor(INPUT_STD, device=raw.device, dtype=raw.dtype)
        feats = ((raw - mean) / std) * v            # zero out invalid pixels

        parts = [feats, v]
        if img.shape[-1] >= 4 + NUM_RESIDUALS:
            res = img[..., 4:4 + NUM_RESIDUALS] / RESIDUAL_STD
            parts.append(res * v)
            # flag telling the network which sensors actually carry a residual;
            # the moving vehicle lidars get zeros, not a measured zero difference
            parts.append((res.abs().amax(-1, keepdim=True) > 0).float().expand_as(res)[..., :1])

        if hag is not None and origin is not None:
            from rangeview.data.geometry import ray_geometry, surface_normals
            r_s, u = ray_geometry(xyz, origin)
            nrm = surface_normals(xyz, valid, origin)
            r_s = ((r_s - mean[4]) / std[4]).unsqueeze(-1) * v
            parts += [r_s, u * v, (hag.unsqueeze(-1) / HAG_STD) * v, nrm * v]

        return torch.cat(parts, dim=-1).permute(0, 3, 1, 2).contiguous(), v.squeeze(-1)

    def forward(self, sample):
        """sample['images']: dict agent -> (B, H, W, C), ego frame, agents vary.

        Each sensor is encoded on its own range-image lattice and scattered into
        its OWN BEV grid; the grids are then fused by attention inside the
        backbone. Previously every agent scattered into one shared grid via
        scatter_max, which treats a 64-beam RSU return and a 128-beam vehicle
        return as interchangeable and simply keeps whichever activates higher.
        The dataset paper reports "performance varies across agents with
        different sensor configurations", so which agent to trust is spatially
        dependent -- an RSU covers the whole intersection, a vehicle only its own
        neighbourhood. Attention can learn that; a max cannot.
        """
        Hb, Wb = bev_dims()
        B = next(iter(sample["images"].values())).shape[0]
        device = next(self.parameters()).device

        # `top` first: the fusion reads out the ego slot, and top is always present
        names = sorted(sample["images"].keys(), key=lambda a: (a != "top", a))
        grids, present = [], []

        for agent in names:
            img = sample["images"][agent]
            x, valid = self.build_input(
                img, sample["valid"][agent],
                hag=sample.get("hag", {}).get(agent) if self.use_geometry else None,
                origin=sample.get("origins", {}).get(agent) if self.use_geometry else None)
            xyz_ch = img[..., :3].permute(0, 3, 1, 2).contiguous()
            if self.grad_ckpt and self.training:
                f = checkpoint(self.encoder, x, xyz_ch, valid, use_reentrant=False)
            else:
                f = self.encoder(x, xyz_ch, valid)
            if agent in self.agent_order:
                ai = self.agent_order.index(agent)
                f = f * (1.0 + self.agent_gamma[ai].view(1, -1, 1, 1)) \
                    + self.agent_beta[ai].view(1, -1, 1, 1)
            C = f.shape[1]
            # fp32 for the scatter: under AMP the encoder emits Half, which
            # scatter_max refuses against an fp32 buffer, and a max-reduction
            # over many contributors is steadier in fp32 anyway.
            f = f.permute(0, 2, 3, 1).reshape(-1, C).float()

            xyz = img[..., :3].reshape(B, -1, 3)
            col = ((xyz[..., 0] - PC_RANGE[0]) / BEV_RES).long()
            row = ((xyz[..., 1] - PC_RANGE[1]) / BEV_RES).long()
            inside = (
                (col >= 0) & (col < Wb) & (row >= 0) & (row < Hb)
                & (xyz[..., 2] >= PC_RANGE[2]) & (xyz[..., 2] <= PC_RANGE[5])
                & valid.reshape(B, -1).bool()
            )
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(col)
            idx = (b_idx * Hb * Wb + row * Wb + col)[inside]

            # PointPillars-style pillar encoding before the reduction. Plain
            # scatter_max discards the within-cell vertical structure: measured,
            # 75.2% of occupied cells hold >1 point and 22.0% of those span more
            # than a metre in z, so a car roof and the road beneath it collapse
            # to one per-channel maximum. Appending the pillar geometry lets the
            # max express which return it kept. See pillar_vfe.py.
            sel = inside.reshape(-1)
            pts = xyz.reshape(-1, 3)[sel]
            cell_xy = torch.stack([
                (col.reshape(-1)[sel].float() + 0.5) * BEV_RES + PC_RANGE[0],
                (row.reshape(-1)[sel].float() + 0.5) * BEV_RES + PC_RANGE[1],
            ], dim=1)
            if self.vfe is not None:
                pf = self.vfe(f[sel], pts.float(), idx, cell_xy) if idx.numel() \
                    else f[sel].new_zeros((0, self.bev_ch))
            else:
                pf = f[sel]
            flat = torch.full((B * Hb * Wb, self.bev_ch), -1e4, device=device)
            if idx.numel():
                scatter_max_(flat, pf.float(), idx, dim=0)
            grids.append(flat.clamp(min=0).view(B, Hb, Wb, self.bev_ch).permute(0, 3, 1, 2))
            # an agent padded with zeros by collate contributes no valid pixels
            present.append(valid.reshape(B, -1).any(dim=1))

        N = len(grids)
        bev = torch.stack(grids, dim=1).reshape(B * N, self.bev_ch, Hb, Wb).contiguous()
        agent_mask = torch.stack(present, dim=1)                      # (B, N)
        return self.head(self.backbone(bev, agent_mask=agent_mask, n_agents=N,
                                       agent_names=names))
