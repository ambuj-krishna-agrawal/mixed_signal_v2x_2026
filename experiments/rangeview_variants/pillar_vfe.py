"""PointPillars-style pillar feature encoding before the BEV scatter.

Why: the previous pipeline flattened the range-view encoder's per-pixel features
and reduced each BEV cell with `scatter_max`. Measured on this dataset, that
throws away real structure -- 75.2% of occupied 0.2 m cells contain more than one
point, and 22.0% of multi-point cells span more than a metre in z. A car roof and
the road under it, or a pedestrian's head and feet, collapse to one per-channel
maximum.

PointPillars (arXiv:1812.05784) and the OpenCOOD baseline that scores 0.77 on
this dataset (`msig_pillar_v2xvit.yaml`, `pillar_vfe: num_filters [64]`) instead
put the within-pillar geometry INSIDE the per-point features before pooling, so
the max can express "the tallest return in this cell" rather than an arbitrary
channel-wise mixture.

We keep the range-view encoder -- it supplies neighbourhood context a per-point
MLP cannot -- and append the pillar geometry PointPillars uses:

    x_c, y_c, z_c : offset from the mean of the points in this pillar
    x_p, y_p      : offset from the pillar's geometric centre
    z             : absolute height

so the pooled feature knows the vertical distribution it came from.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter_mean



class PillarVFE(nn.Module):
    """Per-point MLP over [encoder features | pillar geometry] -> pooled channels."""

    N_GEOM = 6          # x_c, y_c, z_c, x_p, y_p, z

    def __init__(self, cin, cout=32):
        # The reference uses 64 pillar channels; we use 32 because the per-agent
        # BEV grids are stacked as (B, N, C, 512, 512) and 64 channels pushed
        # that to 671 MB and OOMed a 6 GB card. 32 keeps the geometry-aware
        # pooling at the previous footprint (5.45 GB of 6.14).
        super().__init__()
        self.cout = cout
        # LayerNorm, NOT GroupNorm. make_norm() returns GroupNorm(32, 32) --
        # one channel per group -- and applied to a per-point (P, C, 1, 1)
        # tensor each group then holds exactly ONE element, so the normaliser
        # computes (x - x) / sqrt(0 + eps) = 0 and the whole VFE collapses to a
        # constant. Measured: variance across points 1.45e-10, i.e. every point
        # emitted the identical feature and the BEV grid carried no information.
        # It showed up in training as `rot` frozen at 0.153, `hm` stalling near
        # 4.8 instead of ~1.1, and mAP 0.000.
        # LayerNorm normalises across the channel axis WITHIN each point, which
        # is the standard choice for a PointNet-style per-point MLP and is
        # batch-size independent like GroupNorm.
        self.mlp = nn.Sequential(
            nn.Linear(cin + self.N_GEOM, cout, bias=False),
            nn.LayerNorm(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat, xyz, flat_idx, cell_xy):
        """feat (P, C) per-point encoder features for the VALID points only
        xyz  (P, 3) ego-frame coordinates
        flat_idx (P,) pillar id per point (batch*H*W + row*W + col)
        cell_xy (P, 2) the geometric centre of each point's pillar
        -> (P, cout) features ready to scatter_max
        """
        # offset from the pillar's point-cloud centroid (PointPillars x_c,y_c,z_c)
        mean = scatter_mean(xyz, flat_idx, dim=0, dim_size=int(flat_idx.max()) + 1)
        centroid = mean.index_select(0, flat_idx)
        geom = torch.cat([xyz - centroid,                 # x_c, y_c, z_c
                          xyz[:, :2] - cell_xy,           # x_p, y_p
                          xyz[:, 2:3]], dim=1)            # z
        x = torch.cat([feat, geom.to(feat.dtype)], dim=1)
        return self.mlp(x)
