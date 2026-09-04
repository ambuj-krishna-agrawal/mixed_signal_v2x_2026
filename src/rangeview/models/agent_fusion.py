"""Per-agent BEV features fused by attention, following the V2X-ViT / AttFusion
pattern used by OpenCOOD.

Why this is compatible with our range-view encoder: V2X-ViT never touches the
point representation. In the devkit's own pipeline it runs as

    pillar_vfe -> scatter -> backbone -> spatial_features_2d -> V2XTransformer

i.e. it consumes a per-agent BEV feature map and is agnostic to how that map was
produced. We produce ours by encoding each sensor's range image and scattering
it, which yields the same tensor.

What it replaces: a single shared BEV grid filled by scatter_max over every
agent at once. That treats a 64-beam RSU return and a 128-beam vehicle return as
interchangeable and simply keeps whichever activates higher. The dataset paper
notes "performance varies across agents with different sensor configurations",
so which agent to trust is spatially dependent -- an RSU sees the whole
intersection reliably, a vehicle only its own neighbourhood. Attention can learn
that; a max cannot.
"""
import torch
import torch.nn as nn

from rangeview.models.norm import make_norm


# Fixed sensor set, in the order model.py sorts them (`top` first, then
# alphabetical). Per-agent parameters are indexed by NAME via this table, never
# by slot position, so an absent agent cannot silently shift them.
AGENT_ORDER = ("top", "003", "004", "dome", "laser")


class AgentAttentionFusion(nn.Module):
    """Scaled dot-product attention across the agent axis, applied per BEV cell.

    Input  (B, N, C, H, W) with a validity mask (B, N) for absent agents.
    Output (B, C, H, W).

    HETEROGENEOUS AGENT CONDITIONING (V2X-ViT's `use_hetero`, its HMSA block):
    the five sensors are genuinely different instruments -- `top` is an OS1-64
    (45 deg vertical FOV, 100 m), `dome` an OS Dome-128 (180 deg, 45 m), and the
    three vehicle lidars are two further types that record no intensity at all.
    V2X-ViT gives each agent TYPE its own attention projections rather than
    treating the agent axis as homogeneous.

    We learned the hard way that this matters: BatchNorm was implicitly doing
    per-agent normalisation during training (each agent forward normalised by its
    own batch statistics) and collapsing at inference, where one global running
    average served all five -- worth a measured 13x on the loss. GroupNorm fixed
    the train/eval mismatch but is deliberately agent-agnostic. This embedding
    puts the agent identity back, explicitly and deterministically, instead of
    letting it leak in through normalisation statistics.

    A per-agent affine (FiLM) is used rather than full per-agent q/k/v weights:
    it captures the same "this sensor's features mean something different"
    signal at 2*C parameters per agent instead of 3*C*C, and it is initialised to
    the identity so it cannot hurt at step 0.
    """

    def __init__(self, channels, hetero=True):
        super().__init__()
        self.hetero = hetero
        if hetero:
            # gamma starts at 0 and is used as (1 + gamma), so this is exactly
            # the identity transform at initialisation
            self.agent_gamma = nn.Parameter(torch.zeros(len(AGENT_ORDER), channels))
            self.agent_beta = nn.Parameter(torch.zeros(len(AGENT_ORDER), channels))
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            make_norm(channels), nn.ReLU(inplace=True),
        )

    def forward(self, feats, agent_mask, agent_names=None):
        """feats (B,N,C,H,W), agent_mask (B,N) -> (B,C,H,W).

        agent_names: the N sensor names in slot order. Required for the
        heterogeneous conditioning; without it the fusion stays homogeneous.
        """
        B, N, C, H, W = feats.shape
        flat = feats.reshape(B * N, C, H, W)

        if self.hetero and agent_names is not None:
            idx = torch.tensor(
                [AGENT_ORDER.index(a) if a in AGENT_ORDER else 0 for a in agent_names],
                device=feats.device)
            g = (1.0 + self.agent_gamma[idx]).view(1, N, C, 1, 1)
            b = self.agent_beta[idx].view(1, N, C, 1, 1)
            flat = (feats * g + b).reshape(B * N, C, H, W)
        q = self.q(flat).view(B, N, C, H * W)
        k = self.k(flat).view(B, N, C, H * W)
        v = self.v(flat).view(B, N, C, H * W)

        # cross-agent attention: at each BEV cell the N agents form a length-N
        # sequence and attend to one another. The N x N matrix is tiny (N<=5).
        q = q.permute(0, 3, 1, 2).reshape(B * H * W, N, C)
        k = k.permute(0, 3, 1, 2).reshape(B * H * W, N, C)
        v = v.permute(0, 3, 1, 2).reshape(B * H * W, N, C)

        att = torch.bmm(q, k.transpose(1, 2)) * self.scale
        m = agent_mask.bool().unsqueeze(1).expand(B, H * W, N).reshape(B * H * W, 1, N)
        att = att.masked_fill(~m, float("-inf")).softmax(dim=-1)
        att = torch.nan_to_num(att, nan=0.0)       # cell seen by no agent

        fused = torch.bmm(att, v)[:, 0]            # read out the ego (top) slot
        fused = fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return self.out(fused)
