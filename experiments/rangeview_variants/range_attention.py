"""Range-Adaptive Azimuth Attention (RA3) for range images.

Standard image self-attention transfers badly to range images, and there is
measured evidence: "To the Point" (Chai et al., CVPR'21) reports self-attention
BELOW plain convolution on Waymo (57.9 vs 63.4 AP). Three reasons:

  1. pixel adjacency != 3D adjacency. Two neighbouring columns can straddle a
     depth discontinuity and be tens of metres apart in space, so the locality
     prior that makes attention work on RGB is simply false here.
  2. scale varies with range. A vehicle spans ~66 azimuth columns at 10 m and
     ~16 at 40 m, so any FIXED window is wrong at one end of the range.
  3. positional encoding lives in (row, col) rather than in 3D geometry.

RA3 addresses each:

  (a) RANGE-ADAPTIVE DILATION. The arc subtended by w columns at range r is
      w * (2*pi/W) * r. To make the outermost tap land a fixed physical
      distance L/2 away regardless of depth:
              d(r) = L * W / (2*pi * r * 2 * T)
      with T taps either side. Every pixel then attends over the same real
      -world extent. Prior work conditions the WEIGHTS on range (RangePerception
      RAK; "To the Point" RQ-Conv2D); conditioning the RECEPTIVE FIELD is the
      part we have not found published, and it targets a measured defect: our
      encoder's azimuth receptive field is 9 columns against objects spanning
      33-66.

  (b) GEOMETRY GATING. Logits are penalised by squared 3D distance, so pixels
      across an occlusion boundary are suppressed even when adjacent on the
      lattice.

  (c) 3D RELATIVE POSITION BIAS from (dx, dy, dz) rather than (drow, dcol).
      This part is NOT novel -- it is MetaKernel's idea (RangeDet), which we
      already use for convolution, carried over to attention.

Implemented with per-pixel gathers rather than unfold: unfold at full lattice
resolution would materialise C*taps*H*W and does not fit in 6 GB.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeAdaptiveAttention(nn.Module):
    def __init__(self, ch, taps=3, phys_extent=4.0, heads=4,
                 max_dilation=16, sigma=2.0):
        super().__init__()
        self.T = taps                    # taps each side; total 2T+1
        self.L = phys_extent             # metres the window should span
        self.h = heads
        self.max_d = max_dilation
        self.sigma = sigma
        self.q = nn.Conv2d(ch, ch, 1, bias=False)
        self.k = nn.Conv2d(ch, ch, 1, bias=False)
        self.v = nn.Conv2d(ch, ch, 1, bias=False)
        self.pos = nn.Sequential(nn.Conv2d(3, 16, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, heads, 1))
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x, xyz, valid):
        """x (B,C,H,W); xyz (B,3,H,W) ego metres; valid (B,H,W)."""
        B, C, H, W = x.shape
        dev = x.device
        r = xyz.norm(dim=1).clamp(min=1.0)                       # (B,H,W)
        # (a) per-pixel dilation so the outer tap sits ~L/2 away in metres
        d = (self.L * W / (2 * math.pi * r * 2 * self.T)).round()
        d = d.clamp(1, self.max_d).long()                        # (B,H,W)

        q = self.q(x); k = self.k(x); v = self.v(x)
        hd = C // self.h
        base = torch.arange(W, device=dev).view(1, 1, W)
        logits, vals, geos = [], [], []
        for t in range(-self.T, self.T + 1):
            idx = (base + t * d) % W                             # wrap azimuth
            gi = idx.unsqueeze(1).expand(B, C, H, W)
            kt = torch.gather(k, 3, gi)
            vt = torch.gather(v, 3, gi)
            pt = torch.gather(xyz, 3, idx.unsqueeze(1).expand(B, 3, H, W))
            vt_ok = torch.gather(valid.unsqueeze(1).float(), 3,
                                 idx.unsqueeze(1)).squeeze(1)     # (B,H,W)
            dp = pt - xyz                                        # (B,3,H,W)
            # (c) 3D relative position bias, per head
            bias = self.pos(dp)                                  # (B,heads,H,W)
            # (b) geometry gate: squared 3D distance
            gate = -(dp.pow(2).sum(1, keepdim=True)) / (2 * self.sigma ** 2)
            att = (q.view(B, self.h, hd, H, W) * kt.view(B, self.h, hd, H, W)
                   ).sum(2) / math.sqrt(hd)                      # (B,heads,H,W)
            att = att + bias + gate
            att = att.masked_fill(vt_ok.unsqueeze(1) < 0.5, float("-inf"))
            logits.append(att); vals.append(vt.view(B, self.h, hd, H, W))
        A = torch.softmax(torch.stack(logits, 0), 0)             # (taps,B,heads,H,W)
        A = torch.nan_to_num(A)                                  # all-invalid pixels
        out = (A.unsqueeze(3) * torch.stack(vals, 0)).sum(0)     # (B,heads,hd,H,W)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)
