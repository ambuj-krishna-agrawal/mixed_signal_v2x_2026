"""Normalisation choice for the range-view detector.

BatchNorm was measured to be the single largest defect in this pipeline. One
shared encoder is run SEPARATELY per agent with batch size 2, so during training
BatchNorm normalises each agent by that agent's own batch statistics -- in effect
per-agent normalisation. At inference every agent instead receives one global
running average that fits none of them, because the five sensors have genuinely
different range distributions (`top` is 64 rows on a mast; `004` is a vehicle
roof metres from other traffic).

Measured on an overfit batch, identical weights, train vs eval mode:
    off loss   0.0011 -> 0.0141   (13x)
    top   3D centre err  0.390 m -> 0.921 m,  per-pixel box IoU 0.685 -> 0.233
    dome                 0.240 m -> 0.688 m,                    0.687 -> 0.321
    004                  0.037 m -> 0.235 m,                    0.966 -> 0.783
and it does NOT wash out with training -- at epoch 7 (31k steps) on held-out
data the off loss was still 0.0429 (eval) vs 0.0282 (batch stats).

GroupNorm (Wu & He, arXiv:1803.08494) normalises within each sample, so training
and inference are identical and the batch size is irrelevant. That makes
legitimate at inference exactly the behaviour that already works in train mode.
"""
import torch.nn as nn

NORM = "group"


class GroupNorm(nn.GroupNorm):
    """GroupNorm that returns the autocast dtype instead of fp32.

    torch.autocast keeps group_norm in fp32 (BatchNorm gets a fused kernel that
    returns bf16). With 43 norm layers at full range resolution that doubled
    activation memory and pushed a 4.5 GB model out of a 5.64 GB card. The
    statistics are still computed in fp32 -- only the output is cast back, so
    numerical behaviour is unchanged and the downstream activations halve.
    """

    def forward(self, x):
        return super().forward(x).to(x.dtype)          # "group" | "batch"


def make_norm(channels, kind=None):
    kind = kind or NORM
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    # Wu & He use 32 groups; fall back to the largest divisor <= 32 so narrow
    # layers (our heads go down to 2-3 channels) stay valid.
    g = min(32, channels)
    while g > 1 and channels % g:
        g -= 1
    return GroupNorm(g, channels)
