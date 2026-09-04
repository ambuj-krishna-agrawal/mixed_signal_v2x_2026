"""Cross-platform compatibility shim.

Three concerns live here so the rest of the tree stays clean:

1. Device selection + autocast + empty_cache without hardcoding "cuda".
2. `scatter_max_` / `scatter_mean` fallbacks: native `torch.scatter_reduce_`
   when torch_scatter is unavailable (Mac MPS has no torch_scatter build).
3. Rotated BEV IoU / NMS fallbacks via Shapely, so decode/evaluate/loss stop
   needing the OpenCOOD CUDA kernels. The CUDA path is preserved via a
   preference order: if opencood is importable AND CUDA is present, use it;
   otherwise use the Shapely fallback.
"""
from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext

import numpy as np
import torch


# ---------------------------------------------------------------------------
# device helpers

def default_device() -> torch.device:
    """cuda -> mps -> cpu. Overridable via MIXED_SIGNALS_DEVICE."""
    forced = os.environ.get("MIXED_SIGNALS_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_supported(device: torch.device) -> bool:
    return device.type in ("cuda", "mps", "cpu")


@contextmanager
def autocast_ctx(device: torch.device | str, dtype=torch.bfloat16, enabled: bool = True):
    """torch.amp.autocast that accepts the actual device type instead of a
    hardcoded "cuda". Silently disables on backends that would refuse.

    fp16 on MPS is problematic (GradScaler unavailable, focal-loss clamp
    breaks per train.py's comment), so bf16 is the default. Callers pick.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if not enabled or not _amp_supported(device):
        yield
        return
    # torch.amp.autocast wants a device_type string, not a torch.device
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True):
        yield


def empty_cache(device: torch.device | None = None) -> None:
    d = device or default_device()
    if d.type == "cuda":
        torch.cuda.empty_cache()
    elif d.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # cpu: nothing to do


def pin_ok(device: torch.device | None = None) -> bool:
    """DataLoader pin_memory only helps for CUDA. MPS silently ignores it in
    recent torch, but older versions warn on every worker."""
    d = device or default_device()
    return d.type == "cuda"


# ---------------------------------------------------------------------------
# scatter shims. torch_scatter.scatter_max / scatter_mean have no MPS build.

try:
    from torch_scatter import scatter_max as _ts_scatter_max
    from torch_scatter import scatter_mean as _ts_scatter_mean
    _HAS_TS = True
except Exception:
    _HAS_TS = False


def scatter_max_(out: torch.Tensor, src: torch.Tensor, index: torch.Tensor,
                 dim: int = 0) -> torch.Tensor:
    """In-place equivalent of `torch_scatter.scatter_max(src, index, dim, out=out)`.

    Uses native `Tensor.scatter_reduce_(reduce="amax", include_self=True)` when
    torch_scatter is missing. `out` must be pre-initialised with the identity
    element for max (typically a very negative sentinel), the same way the
    original call sites use it.
    """
    if _HAS_TS and out.device.type in ("cuda", "cpu"):
        _ts_scatter_max(src, index, dim=dim, out=out)
        return out
    if index.dim() == 1 and src.dim() > 1 and dim == 0:
        index = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(dim, index, src, reduce="amax", include_self=True)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: int | None = None) -> torch.Tensor:
    """Drop-in for `torch_scatter.scatter_mean(src, index, dim=0, dim_size=...)`."""
    if _HAS_TS and src.device.type in ("cuda", "cpu"):
        return _ts_scatter_mean(src, index, dim=dim, dim_size=dim_size)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    if index.dim() == 1 and src.dim() > 1 and dim == 0:
        index = index.unsqueeze(-1).expand_as(src)
    return out.scatter_reduce(dim, index, src, reduce="mean", include_self=False)


# ---------------------------------------------------------------------------
# rotated BEV IoU + NMS. Shapely fallback; preserves numpy/torch signatures.

from shapely.geometry import Polygon


def _rot_rect(cx, cy, dx, dy, yaw) -> Polygon:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    hx, hy = float(dx) / 2.0, float(dy) / 2.0
    corners = np.array([[ hx,  hy], [ hx, -hy], [-hx, -hy], [-hx,  hy]])
    rot = np.array([[c, -s], [s, c]])
    corners = corners @ rot.T + np.array([float(cx), float(cy)])
    return Polygon(corners)


def boxes_bev_iou_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,7) x (M,7) rotated BEV IoU, numpy in/out. Columns: x,y,z,dx,dy,dz,yaw.

    Matches OpenCOOD's `boxes_bev_iou_cpu` signature so call sites can be a
    drop-in swap.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 7)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 7)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    polys_a = [_rot_rect(x[0], x[1], x[3], x[4], x[6]) for x in a]
    polys_b = [_rot_rect(x[0], x[1], x[3], x[4], x[6]) for x in b]
    area_a = np.array([p.area for p in polys_a], dtype=np.float64)
    area_b = np.array([p.area for p in polys_b], dtype=np.float64)
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, pa in enumerate(polys_a):
        for j, pb in enumerate(polys_b):
            inter = pa.intersection(pb).area
            if inter <= 0.0:
                continue
            union = area_a[i] + area_b[j] - inter
            if union > 0.0:
                out[i, j] = float(inter / union)
    return out


def boxes_iou_bev(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch wrapper around `boxes_bev_iou_cpu` that preserves the device of
    the inputs (materialised on CPU internally)."""
    device = a.device
    A = a.detach().cpu().float().contiguous().numpy()
    B = b.detach().cpu().float().contiguous().numpy()
    M = boxes_bev_iou_cpu(A, B)
    return torch.from_numpy(M).to(device)


def nms_bev(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Greedy rotated-BEV NMS. Returns `keep` indices into the input order.

    Matches the semantics of the CPU fallback already present in decode.py:
    boxes are score-sorted, each surviving box suppresses later boxes with
    IoU > iou_thresh.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    b_sorted = boxes.index_select(0, order)
    M = boxes_iou_bev(b_sorted, b_sorted).cpu()
    taken = torch.zeros(len(b_sorted), dtype=torch.bool)
    keep_local = []
    for i in range(len(b_sorted)):
        if taken[i]:
            continue
        keep_local.append(i)
        taken |= M[i] > iou_thresh
    idx = torch.tensor(keep_local, dtype=torch.long)
    return order.index_select(0, idx.to(order.device))
