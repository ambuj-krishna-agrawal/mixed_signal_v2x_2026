"""Inference-only throughput benchmark: forward + decode only, no gradient."""
import argparse
import time

import torch
from torch.utils.data import DataLoader

from rangeview.compat import default_device
from rangeview.data.dataset import RangeViewDataset, collate
from rangeview.models.decode import SCORE_THRESH, decode
from rangeview.data.geometry import batch_to_device
from rangeview.models.net import RangeViewFusionNet
import rangeview.models.decode as _dec
import rangeview.models.norm as _norm


def bench(ckpt, cache, val_scenes, bs, workers, warmup=2, steps=15):
    device = default_device()
    sd = torch.load(ckpt, map_location=device)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    _norm.NORM = "batch" if any(k.endswith("running_mean") for k in sd) else "group"
    _dec.configure_from_state_dict(sd)
    _ms = any(k.startswith("backbone.fuse2") for k in sd)
    m = RangeViewFusionNet(use_vfe=False, use_residuals=False,
                           use_geometry=False, use_iou_head=False,
                           ms_fusion=_ms).to(device).eval()
    m.load_state_dict(sd)

    ds = RangeViewDataset(cache, keep_scenes=val_scenes, residuals=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=collate,
                    persistent_workers=(workers > 0))
    it = iter(dl)
    with torch.no_grad():
        for _ in range(warmup):
            try: b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            b = batch_to_device(b, device)
            _ = decode(m({"images": b["images"], "valid": b["valid"]}),
                       score_thresh=SCORE_THRESH)
        if device.type == "mps": torch.mps.synchronize()

        t0 = time.time()
        n = 0
        for _ in range(steps):
            try: b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            b = batch_to_device(b, device)
            _ = decode(m({"images": b["images"], "valid": b["valid"]}),
                       score_thresh=SCORE_THRESH)
            n += bs
        if device.type == "mps": torch.mps.synchronize()
    dt = time.time() - t0
    return {"bs": bs, "steps": steps, "wall_s": dt, "frames_per_s": n / dt,
            "it_per_s": steps / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-scenes", type=int, nargs="+", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    a = ap.parse_args()
    print(f"{'bs':>4} {'steps':>6} {'wall_s':>8} {'it/s':>7} {'frames/s':>10}")
    for bs in a.batches:
        try:
            r = bench(a.ckpt, a.cache, a.val_scenes, bs, workers=4)
            print(f"{r['bs']:>4} {r['steps']:>6} {r['wall_s']:>8.1f} "
                  f"{r['it_per_s']:>7.2f} {r['frames_per_s']:>10.2f}", flush=True)
        except (RuntimeError, MemoryError) as e:
            print(f"{bs:>4}  OOM/error: {str(e)[:120]}", flush=True)
            break


if __name__ == "__main__":
    main()
