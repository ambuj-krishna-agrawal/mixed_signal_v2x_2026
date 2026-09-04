"""RangeDet's Meta-Kernel (ICCV 2021, arXiv:2103.10039).

The problem it solves, in the paper's words: "the inconsistency between the 2D
range image coordinates used in feature extraction and the 3D Cartesian
coordinates used in output". A plain 2D conv weights each of its nine
neighbours by a fixed learned kernel, but in a range image the *3D* offset to
those neighbours varies enormously -- neighbouring pixels are centimetres apart
on a nearby surface and metres apart on a distant one. One fixed kernel cannot
represent geometry whose scale changes across the image.

Meta-Kernel instead generates the weights from geometry: for a 3x3 grid it
takes the relative Cartesian coordinates of the nine neighbours with respect to
the centre, passes them through a shared MLP to produce nine weight vectors,
multiplies each neighbour's features by its weight, concatenates, and mixes the
result with a 1x1 convolution.

This is the fix for the encoder weakness in our current model, which feeds
absolute xyz as input channels and then relies on ordinary convolutions.
"""
import torch
import torch.nn as nn

from rangeview.models.norm import make_norm
import torch.nn.functional as F


class MetaKernel(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, coord_hidden=32):
        super().__init__()
        self.k = k
        self.n = k * k
        self.in_ch = in_ch

        # shared MLP over the 3 relative Cartesian coords -> one weight vector
        # per neighbour, applied identically at every pixel and neighbour
        self.coord_mlp = nn.Sequential(
            nn.Conv2d(3, coord_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(coord_hidden, in_ch, 1, bias=True),
        )
        # 1x1 mixing over the concatenated, geometry-weighted neighbours
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch * self.n, out_ch, 1, bias=False),
            make_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat, xyz, valid):
        """feat: (B,C,H,W)  xyz: (B,3,H,W) ego frame  valid: (B,H,W)"""
        B, C, H, W = feat.shape
        p = self.k // 2

        # neighbours: (B, C*n, H*W) -> (B, C, n, H, W)
        f_nb = F.unfold(feat, self.k, padding=p).view(B, C, self.n, H, W)
        c_nb = F.unfold(xyz, self.k, padding=p).view(B, 3, self.n, H, W)
        v_nb = F.unfold(valid.unsqueeze(1).float(), self.k, padding=p).view(B, 1, self.n, H, W)

        # relative coordinates w.r.t. the centre pixel
        rel = c_nb - xyz.unsqueeze(2)
        # an invalid neighbour has meaningless coordinates: zero it out so it
        # neither contributes features nor drives the weight MLP
        rel = rel * v_nb

        # shared MLP: fold the neighbour axis into the batch so weights are tied
        w = self.coord_mlp(rel.permute(0, 2, 1, 3, 4).reshape(B * self.n, 3, H, W))
        w = w.view(B, self.n, C, H, W).permute(0, 2, 1, 3, 4)

        out = (f_nb * w * v_nb).reshape(B, C * self.n, H, W)
        return self.mix(out)
