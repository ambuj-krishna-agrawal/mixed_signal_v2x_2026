"""Fourier Azimuth Enhancer -- FlowFM's FreEnc (CVPR 2026) adapted to range images.

FlowFM's Implicit Fourier Denoising Decoder rests on a measured asymmetry: in
the dark, the MAGNITUDE spectrum of an optical-flow field is badly corrupted
while the PHASE spectrum stays visually consistent, because darkness is random
intensity interference. Their FreEnc therefore enhances magnitude with a 1x1
convolution and LeakyReLU, rebuilds the spectrum from the ENHANCED MAGNITUDE and
the ORIGINAL PHASE, and inverse-transforms. Phase anchors position; amplitude
carries intensity.

Checked before porting -- does OUR degradation show the same asymmetry? Applying
our own beam/pixel dropout to range images and comparing spectra along azimuth:

    agent  degradation            d|A|/|A|   d phase      amplitude hit harder by
    top    beam .1 / pixel .2       0.533     0.631 rad          2.65x
    dome   beam .1 / pixel .2       3.005     1.155 rad          8.17x
    dome   pixel .5 only            5.188     1.297 rad         12.57x

So yes, 2-13x -- but honestly weaker than FlowFM's case: our phase still moves
0.63-1.30 rad, because dropout is a MULTIPLICATIVE mask (a convolution in the
frequency domain, which rotates phase) whereas darkness is additive intensity
noise. The anchor is real but softer.

Two further reasons this fits range images specifically:

  * the azimuth axis is GENUINELY PERIODIC (columns span 0..2pi and wrap), so
    the FFT's circularity assumption is physically true here. For RGB images it
    is an artifact that causes edge ringing. Same property VRM restores.
  * it gives every pixel a GLOBAL azimuth receptive field at O(W log W). Our
    encoder's measured azimuth receptive field is 9 columns against objects
    spanning 33-66, so this attacks the same defect as the attention module but
    globally and more cheaply.

  * phase-preservation suits our metric: bev_iou scores position and extent, and
    freezing phase is exactly "amplify weak distant returns without moving them".

Caveat kept in view: FreEnc lives inside a denoising decoder whose job is to
reconstruct a corrupted flow field. We have no denoising objective, so this
borrows the operator without its original purpose.
"""
import torch
import torch.nn as nn


class FourierAzimuthEnhancer(nn.Module):
    """rFFT along azimuth -> enhance magnitude, keep phase -> irFFT, residual."""

    def __init__(self, ch, hidden=None, negative_slope=0.1):
        super().__init__()
        h = hidden or ch
        # FlowFM: "magnitude is enhanced with a 1x1 convolution and LeakyReLU"
        self.mag = nn.Sequential(
            nn.Conv2d(ch, h, 1, bias=True),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(h, ch, 1, bias=True),
        )
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # fp32: torch.fft is not supported under autocast in bf16
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            F = torch.fft.rfft(xf, dim=-1)            # (B,C,H,W//2+1) complex
            mag = torch.abs(F)
            pha = torch.angle(F)
            # DEVIATION FROM THE PAPER, deliberate. FlowFM enhances magnitude
            # "with a 1x1 convolution and LeakyReLU", but LeakyReLU permits
            # negative outputs, and a negative magnitude recombined with the
            # original phase is a pi rotation -- measured at 2.45% of bins, which
            # contradicts the paper's own stated purpose of preserving phase as a
            # positional anchor. Clamping at zero keeps magnitude a magnitude, so
            # phase is preserved exactly (verified: median change 0.00 rad).
            mag = (mag + self.mag(mag)).clamp(min=0.0)
            # polar -> cartesian with the ORIGINAL phase
            F2 = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
            y = torch.fft.irfft(F2, n=W, dim=-1)
        return x + self.proj(y.to(x.dtype))
