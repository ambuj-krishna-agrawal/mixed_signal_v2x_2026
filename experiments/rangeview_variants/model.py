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
from torch_scatter import scatter_max

from norm import make_norm

from agent_fusion import AgentAttentionFusion
from meta_kernel import MetaKernel

# Detection range, matched to the PointPillars baseline so the two are comparable.
PC_RANGE = (-51.2, -51.2, -6.0, 51.2, 51.2, 1.0)
BEV_RES = 0.2                      # -> 512 x 512 grid
OUT_STRIDE = 2                     # head predicts at 256 x 256 (0.4 m / cell)
PED_HIRES_STRIDE = 1               # pedestrian heatmap at 512 x 512 (0.2 m / cell)

# Height ceiling for the BEV SCATTER only (the range encoder still sees every
# point, so vertical context survives in the per-pixel features).
#
# Ground sits at z = -4.18 (median GT box bottom). PC_RANGE[5] = +1.0 is 5.2 m
# above it -- roughly 3 m of pure headroom containing canopy, wires and facades.
# scatter_max keeps the per-channel MAXIMUM of a cell, so one leaf can represent
# a cell that also contains a car. Measured: 36% of Vehicles, 53% of Bikes and
# 48% of Pedestrians have points >0.3 m above their box top in the same BEV
# footprint.
#
# -0.8 is 3.38 m above ground: it drops ~15% of all points and 0.00% of points
# inside any GT box (object tops p99: Vehicle -0.91, Bike -1.02, Ped -1.21).
# Exactly 3.0 m (z=-1.18) would clip 0.92% of object points off tall vehicles.
BEV_Z_CEILING = -0.8
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

# Per-agent statistics for log1p(intensity). The five sensors report intensity
# on very different scales -- measured over 157M valid pixels of the training
# cache -- so one global (mean, std) leaves the channel meaning something
# different per sensor. Under the global 0.081/0.140 the standardised channel
# came out at mean -0.39 / std 0.32 for `top` but mean 1.08 / std 1.33 for
# `laser`, a 4x spread, while 003/004 sat at a constant -0.58 carrying no
# information. The per-agent FiLM cannot fix this: it modulates the encoder's
# OUTPUT, so the first convolution still sees the badly-scaled input.
#
# 003 and 004 report intensity of exactly 0.000 everywhere, so their std is set
# to 1.0, which maps the channel to a clean constant 0 instead of -0.58.
INTENSITY_STATS = {
    "top":   (0.0258, 0.0449),
    "dome":  (0.0918, 0.0859),
    "laser": (0.2317, 0.1868),
    "003":   (0.0000, 1.0000),
    "004":   (0.0000, 1.0000),
}

# Two residual range-image channels (LiDAR-MOS), populated for the static RSU
# lidars only. Residuals are already metric differences centred on zero, so they
# need scaling but not mean-subtraction.
NUM_RESIDUALS = 2
RESIDUAL_STD = 0.30
IN_CHANNELS = 6 + NUM_RESIDUALS
# Geometry pack appended by build_input when enabled: sensor-relative range and
# unit ray (4), height above the mapped ground (1), surface normal (3).
GEOM_CHANNELS = 8
# Static-background subtraction: signed residual (background_range - range) and
# a validity flag. The flag is not optional -- a zero residual means either
# "exactly at background distance" (informative) or "no background here"
# (uninformative), and `top` genuinely lacks a background at ~24% of its pixels.
# background feature modes. The RANGE residual measured NEGATIVE on Bike
# (0.8896 -> 0.8670 held-out) because 71% of bikes are static and get absorbed
# into the range background -- only 15.7% of bike boxes register any foreground
# pixel. The INTENSITY residual is a different signal: a surface can reflect
# differently without being closer, so a parked bike is not invisible to it.
BG_MODES = {"both": 3, "intensity": 2, "range": 2}
BG_CHANNELS = 3
BG_STD = 3.0
IBG_STD = 0.15        # intensity residual scale (intensity itself is in [0,1])
# Vision Restoration Module (RangePerception, NeurIPS 2023). A range image's
# columns span azimuth 0..2pi, so its left and right margins are ADJACENT in the
# world; zero-padded convolutions there see a border that does not exist, and an
# object lying on the seam is split with nothing on one side ("Vision
# Corruption"). VRM builds an extended azimuth space [-delta, 2pi+delta] by
# wrapping strips onto both margins, encodes, then removes the duplicated region
# -- the paper's Redundancy Pruner.
#
# Measured on our data: 23.7% / 22.8% / 19.9% of vehicles seen by laser / 003 /
# 004 straddle the seam. `top` and `dome` are unaffected (0.0%) -- their seam
# falls where nothing is, which is why this never surfaced in ego-centric checks.
#
# In RangePerception the extension also duplicates DETECTIONS, because their RPN
# runs on the range image; the Pruner removes those. Our head is in BEV, which
# has no seam, so pruning here is a crop of the feature map before the scatter.
#
# delta must exceed the encoder's azimuth receptive-field radius, which is 4
# columns (stem 3x3, MetaKernel k=3, two 3x3). 16 columns = 5.6 deg at 1024/360.
VRM_DELTA = 16
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

    def __init__(self, cin=IN_CHANNELS + 1, cout=32, use_meta=True, attn=False,
                 fourier=False):
        super().__init__()
        self.use_meta = use_meta
        self.stem = ConvBlock(cin, 32)
        self.meta = MetaKernel(32, 32) if use_meta else ConvBlock(32, 32)
        # Range-Adaptive Azimuth Attention: window width scales as 1/r so every
        # pixel attends over a constant PHYSICAL extent. Zero-init residual gate,
        # so at initialisation the encoder is bit-identical to the non-attention
        # one. See range_attention.py.
        self.attn = None
        if attn:
            from range_attention import RangeAdaptiveAttention
            self.attn = RangeAdaptiveAttention(32)
        # Fourier Azimuth Enhancer (FlowFM FreEnc, CVPR'26): enhance the azimuth
        # magnitude spectrum, keep phase. Global azimuth receptive field at
        # O(W log W); the axis is genuinely periodic so the FFT is exact here.
        self.fourier = None
        if fourier:
            from fourier_azimuth import FourierAzimuthEnhancer
            self.fourier = FourierAzimuthEnhancer(32)
        self.head = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, cout))

    def forward(self, x, xyz=None, valid=None):
        h = self.stem(x)
        h = self.meta(h, xyz, valid) if self.use_meta else self.meta(h)
        if self.attn is not None and xyz is not None:
            h = self.attn(h, xyz, valid.float() if valid is not None
                          else torch.ones(h.shape[0], *h.shape[2:], device=h.device))
        if self.fourier is not None:
            h = self.fourier(h)
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

    def forward(self, x, agent_mask=None, n_agents=1, agent_names=None,
                agent_occ=None):
        """x is (B*N, C, H, W) when fusing across agents, else (B, C, H, W).

        Fusion stays after down1, at stride 2. The reference fuses after the
        whole backbone, but running three blocks per agent would not fit 6 GB;
        fusing early keeps blocks 2 and 3 at batch B instead of B*N.

        agent_occ (B, N, 2, Hb, Wb) is the per-cell [occupied, log count] map at
        the FULL BEV resolution; each fusion scale max-pools it to its own size.
        Max rather than mean: a cell at stride 8 should count as observed if any
        of the 64 fine cells under it was, which is the question the attention
        is being asked.
        """
        def _occ_at(t):
            if agent_occ is None:
                return None
            B_, N_, C_, H_, W_ = agent_occ.shape
            o = F.adaptive_max_pool2d(agent_occ.reshape(B_ * N_, C_, H_, W_),
                                      tuple(t.shape[-2:]))
            return o.view(B_, N_, C_, *t.shape[-2:])
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
                v2 = t2.view(B, n_agents, c2, h2, w2_)
                v3 = t3.view(B, n_agents, c3, h3, w3_)
                ms2 = self.fuse2(v2, agent_mask, agent_names=agent_names,
                                 occ=_occ_at(v2))
                ms3 = self.fuse3(v3, agent_mask, agent_names=agent_names,
                                 occ=_occ_at(v3))
            d1v = d1.view(B, n_agents, C, H, W)
            d1 = self.agent_fusion(d1v, agent_mask, agent_names=agent_names,
                                   occ=_occ_at(d1v))
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
                 ms_fusion=True, use_geometry=False, use_background=False,
                 vrm=0, attn=False, fourier=False, bg_mode='both',
                 per_agent_intensity=False, ped_hires=False, density=False,
                 z_ceiling=None, hires=None):
        super().__init__()
        # The encoder runs once per sensor (up to 5x) and its Meta-Kernel unfold
        # dominates memory. Checkpointing drops its intermediate activations and
        # recomputes them in backward, which is what makes per-agent BEV grids +
        # attention fusion fit in 6GB. Costs roughly 30% throughput.
        self.grad_ckpt = grad_ckpt
        if hires is None:
            hires = (3,) if ped_hires else ()
        # build_input emits 6 channels without residuals (x, y, z, log-intensity,
        # range, validity) and 9 with them (+2 residual lags +1 availability
        # flag). The encoder must be sized to match, so a residual-free model is
        # a genuinely different architecture -- not the same network with two
        # inputs zeroed, which is why zeroing them at inference is NOT a valid
        # measurement of what they contribute.
        self.use_residuals = use_residuals
        self.use_geometry = use_geometry
        self.use_background = use_background
        self.bg_mode = bg_mode
        # OFF by default: turning it on changes the input distribution, so a
        # checkpoint trained without it would be silently mis-calibrated on
        # channel 3. Recorded in the checkpoint and read back at inference.
        self.per_agent_intensity = bool(per_agent_intensity)
        self.vrm = int(vrm)          # azimuth restoration width, 0 = off
        self.use_attn = bool(attn)
        self.use_fourier = bool(fourier)
        cin = (IN_CHANNELS + 1) if use_residuals else 6
        if use_geometry:
            cin += GEOM_CHANNELS
        if use_background:
            cin += BG_MODES[bg_mode]
        self.encoder = RangeEncoder(cin=cin, cout=feat, use_meta=use_meta, attn=attn,
                                    fourier=fourier)
        # Per-agent conditioning on the SHARED encoder's output. This is exactly
        # where the BatchNorm defect lived: one encoder run separately per agent,
        # with BN silently normalising each by its own batch statistics in
        # training and by one global average at inference (measured 13x loss
        # gap). GroupNorm removed the mismatch but is agent-agnostic; this gives
        # the network an explicit, learned handle on which sensor it is looking
        # at. Identity at initialisation ((1 + 0) * f + 0).
        from agent_fusion import AGENT_ORDER
        self.agent_order = AGENT_ORDER
        self.agent_gamma = nn.Parameter(torch.zeros(len(AGENT_ORDER), feat))
        self.agent_beta = nn.Parameter(torch.zeros(len(AGENT_ORDER), feat))
        # use_vfe=False reproduces the pre-PillarVFE architecture, so
        # checkpoints trained before it can still be loaded and A/B'd.
        from pillar_vfe import PillarVFE
        self.vfe = PillarVFE(cin=feat, cout=pillar_feat) if use_vfe else None
        # POINT-DENSITY CHANNEL.
        #
        # scatter_max keeps the per-channel maximum of every point landing in a
        # BEV cell and discards HOW MANY points there were, so a cell holding
        # 1,173 returns and one holding 96 are indistinguishable. Measured on
        # held-out 31/34: true vehicles carry a median 1,173 points per box,
        # vehicle false positives a median 96 -- a 12x separation the encoder
        # currently cannot see. 91.1% of vehicle FPs sit on no object at all but
        # are NOT empty (0.0% have <5 points), so this is a density cue, not a
        # free-space one. One extra channel, log1p-compressed.
        self.density = density
        # None keeps PC_RANGE[5] (the old behaviour, bit-identical)
        self.z_ceiling = z_ceiling
        self.bev_ch = pillar_feat if use_vfe else feat
        self.backbone = BEVBackbone(cin=self.bev_ch + int(density), cout=128,
                                    ms_fusion=ms_fusion)
        self.head = CenterHead(128, use_iou=use_iou_head)

        # HIGH-RESOLUTION PER-CLASS HEATMAPS (optional).
        #
        # Only the heatmap goes to 0.2 m; offset/z/dim/rot stay at 0.4 m and are
        # gathered from the coarse cell containing the fine peak. The failure
        # this fixes is target/peak MERGING, not regression precision.
        #
        # Why per class rather than for everything. At 0.4 m/cell the Gaussian
        # target is clamped by MIN_GAUSSIAN_RADIUS=2 to a 5-cell blob, and the
        # blob/object width ratio is:
        #     Vehicle 12.1 x 5.5 cells -> 0.9x   (well matched, needs nothing)
        #     Bike     5.0 x 1.8 cells -> 2.7x   (worst over-spread)
        #     Ped      2.3 x 2.1 cells -> 2.3x
        # Pedestrian additionally has 30.8% of instances with a neighbour under
        # 1.0 m (Vehicle and Bike: 0.0%), so its peaks merge outright -- that is
        # why ped_hires lifted the proposal ceiling 87.8% -> 92.5% and gave
        # +0.0462 Pedestrian on the leaderboard. Bike is NOT crowded and its
        # recall is already 98.1% (ceiling 99.2%), so a fine Bike map is aimed at
        # LOCALISATION (our Bike AP@0.7 is 0.3488 vs the leader's 0.3937), not
        # recall. Vehicle is deliberately excluded: 0.9x needs no help and a
        # third branch buys nothing.
        #
        # Measured cost: one branch 4799 -> 5191 MB, two -> ~5481 MB of 5771.
        self.hires = tuple(sorted(hires or ()))
        self.hi_hm = nn.ModuleDict()
        for c in self.hires:
            w = 64
            br = nn.Sequential(
                nn.Conv2d(128, w, 1, bias=False), make_norm(w), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(w, w, 2, 2, bias=False), make_norm(w),
                nn.ReLU(inplace=True), nn.Conv2d(w, 1, 1))
            nn.init.normal_(br[-1].weight, std=0.01)
            nn.init.constant_(br[-1].bias, -2.19)          # focal prior
            self.hi_hm[str(c)] = br
        # backward compat: ped_hires=True is exactly hires=(3,)
        self.ped_hires = 3 in self.hires
        self.feat = feat

    @staticmethod
    def build_input(img, valid, hag=None, origin=None, bg=None, bg_mode="both",
                    agent=None):
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
        if agent in INTENSITY_STATS:                # per-agent intensity scaling
            im_, is_ = INTENSITY_STATS[agent]
            feats[..., 3] = ((raw[..., 3] - im_) / is_) * v[..., 0]

        parts = [feats, v]
        if img.shape[-1] >= 4 + NUM_RESIDUALS:
            res = img[..., 4:4 + NUM_RESIDUALS] / RESIDUAL_STD
            parts.append(res * v)
            # flag telling the network which sensors actually carry a residual;
            # the moving vehicle lidars get zeros, not a measured zero difference
            parts.append((res.abs().amax(-1, keepdim=True) > 0).float().expand_as(res)[..., :1])

        if bg is not None:
            # channel 0 residual (normalised), channel 1 validity flag, already
            # zeroed where invalid by the dataset
            if bg_mode == "both":
                cols = [bg[..., 0] / BG_STD, bg[..., 1] / IBG_STD, bg[..., 2]]
            elif bg_mode == "intensity":
                cols = [bg[..., 1] / IBG_STD, bg[..., 2]]
            else:                                    # "range"
                cols = [bg[..., 0] / BG_STD, bg[..., 2]]
            parts.append(torch.stack(cols, dim=-1))

        if hag is not None and origin is not None:
            from geometry import ray_geometry, surface_normals
            r_s, u = ray_geometry(xyz, origin)
            nrm = surface_normals(xyz, valid, origin)
            r_s = ((r_s - mean[4]) / std[4]).unsqueeze(-1) * v
            parts += [r_s, u * v, (hag.unsqueeze(-1) / HAG_STD) * v, nrm * v]

        return torch.cat(parts, dim=-1).permute(0, 3, 1, 2).contiguous(), v.squeeze(-1)

    def forward(self, sample, return_bev=False):
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

        # `top` first: v1 fusion reads out the ego slot, and top is always
        # present. v2 averages the present agents' query slots instead, so the
        # order no longer privileges one sensor -- but keep it, since the
        # per-agent FiLM tables are indexed by NAME and several callers assume it.
        names = sorted(sample["images"].keys(), key=lambda a: (a != "top", a))
        grids, present, occs = [], [], []

        for agent in names:
            img = sample["images"][agent]
            x, valid = self.build_input(
                img, sample["valid"][agent],
                hag=sample.get("hag", {}).get(agent) if self.use_geometry else None,
                origin=sample.get("origins", {}).get(agent) if self.use_geometry else None,
                bg=sample.get("bg", {}).get(agent) if self.use_background else None,
                bg_mode=self.bg_mode,
                agent=agent if self.per_agent_intensity else None)
            xyz_ch = img[..., :3].permute(0, 3, 1, 2).contiguous()
            d = self.vrm
            if d:
                # extended azimuth space [-delta, 2pi+delta]
                x = torch.cat([x[..., -d:], x, x[..., :d]], dim=-1)
                xyz_ch = torch.cat([xyz_ch[..., -d:], xyz_ch, xyz_ch[..., :d]], dim=-1)
                valid = torch.cat([valid[..., -d:], valid, valid[..., :d]], dim=-1)
            if self.grad_ckpt and self.training:
                f = checkpoint(self.encoder, x, xyz_ch, valid, use_reentrant=False)
            else:
                f = self.encoder(x, xyz_ch, valid)
            if d:
                f = f[..., d:-d]            # Redundancy Pruner
                valid = valid[..., d:-d]
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
                & (xyz[..., 2] >= PC_RANGE[2])
                & (xyz[..., 2] <= (PC_RANGE[5] if self.z_ceiling is None
                                   else self.z_ceiling))
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
                scatter_max(pf.float(), idx, dim=0, out=flat)
            g_ = flat.clamp(min=0).view(B, Hb, Wb, self.bev_ch).permute(0, 3, 1, 2)
            # Per-CELL occupancy, computed unconditionally: the v2 fusion needs
            # to know which agent actually observed a cell, and `agent_mask` is
            # (B, N) frame-level presence, which cannot say that. One extra
            # scatter_add over the index the scatter_max already built.
            # log1p then /8: log1p(1173)=7.07 for a real vehicle cell-sum,
            # log1p(96)=4.57 for a ghost, so the useful range lands near [0,1]
            cnt = torch.zeros(B * Hb * Wb, device=device, dtype=torch.float32)
            if idx.numel():
                cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            logcnt = (torch.log1p(cnt) / 8.0).view(B, 1, Hb, Wb)
            occs.append(torch.cat([(cnt.view(B, 1, Hb, Wb) > 0).float(), logcnt], 1))
            if self.density:
                g_ = torch.cat([g_, logcnt], dim=1)
            grids.append(g_)
            # an agent padded with zeros by collate contributes no valid pixels
            present.append(valid.reshape(B, -1).any(dim=1))

        N = len(grids)
        bev = torch.stack(grids, dim=1).reshape(
            B * N, self.bev_ch + int(self.density), Hb, Wb).contiguous()
        agent_mask = torch.stack(present, dim=1)                      # (B, N)
        agent_occ = torch.stack(occs, dim=1)              # (B, N, 2, Hb, Wb)
        feat = self.backbone(bev, agent_mask=agent_mask, n_agents=N,
                             agent_names=names, agent_occ=agent_occ)
        # `return_bev` exposes the fused BEV feature map (B, cout, Hb/2, Wb/2)
        # for the stage-2 RoI re-scoring head. Default False so every existing
        # caller is unaffected.
        out = self.head(feat)
        for c in self.hires:
            # (B, 1, 512, 512) class-c heatmap logits at 0.2 m/cell.
            # Checkpointed: each branch upsamples to 512x512 and holds ~240 MB of
            # activations. Two of them overflowed a 6 GB card on dense frames
            # even though a light 2-frame batch measured 5421/5771. The branch is
            # 7 shallow layers, so recompute is cheap relative to the encoder.
            br = self.hi_hm[str(c)]
            out[f"hm_hi_{c}"] = (checkpoint(br, feat, use_reentrant=False)
                                 if (self.grad_ckpt and self.training)
                                 else br(feat))
        if self.ped_hires:                      # legacy key, still consumed
            out["hm_ped_hi"] = out["hm_hi_3"]
        return (out, feat) if return_bev else out


def fold_gate_(sd):
    """Backward compat for checkpoints written before the residual gate was removed.

    `RangeAdaptiveAttention` and `FourierAzimuthEnhancer` used to return
    `x + gamma * proj(out)` with a learned scalar `gamma` initialised to zero
    (SAGAN-style). The gate was removed on 2026-08-30 because it spends the
    early epochs of a short run ramping off zero.

    `proj` is a BIAS-FREE 1x1 conv, so `gamma * proj(out)` is exactly
    `Conv2d(weight=gamma*W)(out)`. Folding the scalar into the projection weight
    is therefore an exact algebraic identity, not an approximation: an old
    checkpoint loads into the current model and computes the same function.

    Mutates `sd` in place and returns it. Leaves `agent_gamma` (a different,
    still-live parameter) untouched.
    """
    # ped_hm.* -> hi_hm.3.*  (the single pedestrian branch became a per-class
    # ModuleDict on 2026-09-01). Without this the checkpoint that scored
    # Pedestrian 0.8416 on the leaderboard would fail to load.
    for k in [k for k in sd if k.startswith("ped_hm.")]:
        sd["hi_hm.3." + k[len("ped_hm."):]] = sd.pop(k)
    for key in [k for k in sd if k.endswith((".attn.gamma", ".fourier.gamma"))]:
        proj = key.rsplit(".", 1)[0] + ".proj.weight"
        if proj in sd:
            sd[proj] = sd[proj] * sd[key].reshape(-1)[0].to(sd[proj].dtype)
        sd.pop(key)
    return sd
