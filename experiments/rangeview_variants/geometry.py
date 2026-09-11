"""Geometric input features: sensor-relative rays, height above ground, normals.

All three are computed on the fly in build_input from tensors already in the
batch, so neither cache needs rebuilding.

1. SENSOR-RELATIVE RAY GEOMETRY (4 channels: r_sensor, ux, uy, uz)
   model.build_input's `range` channel is ||xyz|| in the EGO frame while the
   lattice is the agent's own beam x azimuth grid. For `top` those coincide;
   for the other four they do not, so a range discontinuity between adjacent
   pixels stops coinciding with an object boundary -- which is the whole
   inductive bias range view rests on. These channels restore it: true range
   from the observing sensor, plus the unit ray direction.

   Sensor origins come from build_sensor_origins.py. See that file for how they
   are recovered and verified (all three vehicles sit at a constant height above
   ground, MAD 2-4 cm over ~900 frames at unrelated positions).

2. HEIGHT ABOVE GROUND (1 channel)
   z minus the static ground surface from build_ground_map.py. Absolute z is a
   poor proxy: the ground varies with std 1.19 m across the scene, and a local
   patch cannot infer the global surface. Pedestrians and bikes are defined by
   height above the road they stand on.

3. SURFACE NORMALS (3 channels)
   Cross product of lattice-neighbour differences, oriented toward the sensor.
   Separates vertical surfaces (pedestrian, cyclist, pole) from horizontal
   (road, car roof). Computed on the native lattice, so it costs two shifts and
   a cross product.
"""
import numpy as np
import torch

AGENT_INDEX = {"top": 0, "dome": 1, "laser": 2, "003": 3, "004": 4}


class GroundMap:
    """Static ego-frame ground surface, as a lookup on the BEV grid."""

    def __init__(self, path="ground_map.npz", device=None):
        z = np.load(path)
        self.cell = float(z["cell"])
        self.R = float(z["pc_range_xy"])
        g = torch.from_numpy(z["ground"].astype(np.float32))
        self.n = g.shape[0]
        self.ground = g.to(device) if device is not None else g

    def to(self, device):
        self.ground = self.ground.to(device)
        return self

    def at(self, xy):
        """xy: (..., 2) ego metres -> (...) ground z."""
        ix = ((xy[..., 0] + self.R) / self.cell).long().clamp_(0, self.n - 1)
        iy = ((xy[..., 1] + self.R) / self.cell).long().clamp_(0, self.n - 1)
        return self.ground[ix, iy]


def height_above_ground(xyz, gmap):
    """xyz: (B, H, W, 3) -> (B, H, W) height above the mapped surface."""
    return xyz[..., 2] - gmap.at(xyz[..., :2])


def ray_geometry(xyz, origin, eps=1e-6):
    """xyz: (B, H, W, 3); origin: (B, 3) -> (r_sensor (B,H,W), unit (B,H,W,3)).

    A frame whose origin was not recovered arrives as NaN; those fall back to
    the ego origin, which is exactly the current behaviour, so a missing origin
    degrades to the old feature rather than poisoning the batch.
    """
    o = torch.nan_to_num(origin, nan=0.0).view(-1, 1, 1, 3)
    d = xyz - o
    r = d.norm(dim=-1)
    return r, d / (r.unsqueeze(-1) + eps)


def surface_normals(xyz, valid, origin=None, eps=1e-6):
    """xyz: (B, H, W, 3), valid: (B, H, W) -> (B, H, W, 3) unit normals.

    Central differences along the beam axis and the azimuth axis, then a cross
    product. Zeroed where either neighbour is invalid, since a normal spanning
    a no-return is meaningless. Oriented toward the sensor so that a surface
    facing the lidar has a consistent sign.
    """
    du = torch.zeros_like(xyz)
    dv = torch.zeros_like(xyz)
    du[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dv[:, :, 1:-1] = xyz[:, :, 2:] - xyz[:, :, :-2]

    n = torch.cross(du, dv, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + eps)

    v = valid.bool()
    ok = torch.zeros_like(v)
    ok[:, 1:-1, 1:-1] = (v[:, 2:, 1:-1] & v[:, :-2, 1:-1]
                         & v[:, 1:-1, 2:] & v[:, 1:-1, :-2] & v[:, 1:-1, 1:-1])
    n = n * ok.unsqueeze(-1)

    if origin is not None:                      # face the sensor
        o = torch.nan_to_num(origin, nan=0.0).view(-1, 1, 1, 3)
        flip = ((o - xyz) * n).sum(-1, keepdim=True) < 0
        n = torch.where(flip, -n, n)
    return n


def batch_to_device(b, device):
    """Move the per-agent inputs a forward pass needs onto `device`.

    Hand-building this dict as {"images":..., "valid":...} silently drops `hag`,
    `origins` and `bg`, which makes build_input emit the wrong channel count into
    a wider stem. Every evaluation and submission path must go through this.
    """
    out = {"images": {k: v.to(device) for k, v in b["images"].items()},
           "valid": {k: v.to(device) for k, v in b["valid"].items()}}
    for key in ("hag", "origins", "bg"):
        if key in b and isinstance(b[key], dict) and b[key]:
            out[key] = {a: v.to(device) for a, v in b[key].items()}
    return out
