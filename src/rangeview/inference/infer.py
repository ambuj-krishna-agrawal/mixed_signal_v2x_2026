"""Run a trained checkpoint over a cache and save decoded predictions to a `.pt`.

The .pt contains a dict with:
    preds:      list of {"boxes", "scores", "labels"} per frame (torch tensors, CPU)
    gts:        list of (gt_boxes_np, gt_class_np) per frame
    scene_ids:  list of scene ids per frame
    file_names: list of the cache filenames the predictions came from

Downstream consumers (`rangeview.submission.variant`, `rangeview.inference.features`)
read this format.

Y-flip test-time augmentation is on by default (a validated +0.005 test gain, see
`experiments/experiments_done.md` §11.3).  Disable with `--no-yflip`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rangeview.compat import default_device
from rangeview.data.dataset import RangeViewDataset, collate
from rangeview.data.geometry import batch_to_device
from rangeview.inference import tta
from rangeview.models import decode as _dec
from rangeview.models import norm as _norm
from rangeview.models.decode import SCORE_THRESH, decode
from rangeview.models.net import RangeViewFusionNet


def load_checkpoint(ckpt: str | Path, device):
    """Load a checkpoint's state dict onto `device`, auto-detecting norm and ms-fusion."""
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    _norm.NORM = "batch" if any(k.endswith("running_mean") for k in sd) else "group"
    _dec.configure_from_state_dict(sd)
    ms_fusion = any(k.startswith("backbone.fuse2") for k in sd)
    model = RangeViewFusionNet(use_vfe=False, use_residuals=False, use_geometry=False,
                               use_iou_head=False, ms_fusion=ms_fusion).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def infer_dataset(model, loader, device, thr: float, yflip: bool):
    """Forward pass over `loader`.  If `yflip` is True, run identity + y-flip and
    merge via NMS at IoU 0.2 (an in-training-distribution TTA — see experiments_done §11.3).
    """
    def _forward_pass(tf_batch, uf_boxes):
        out = []
        for batch in loader:
            batch = batch_to_device(batch, device)
            b = tf_batch(batch) if tf_batch is not None else batch
            dets = decode(model({"images": b["images"], "valid": b["valid"]}),
                          score_thresh=thr)
            for det in dets:
                if uf_boxes is not None and det["boxes"].numel():
                    det = {"boxes":  uf_boxes(det["boxes"]),
                           "scores": det["scores"],
                           "labels": det["labels"]}
                out.append(det)
        return out

    identity = _forward_pass(None, None)
    if not yflip:
        return identity
    tf, uf = tta.TRANSFORMS["y_flip"]
    flipped = _forward_pass(tf, uf)
    return [tta.merge_multipass([a, b], mode="nms", iou_thresh=0.2)
            for a, b in zip(identity, flipped)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="Model checkpoint (.pth)")
    ap.add_argument("--cache", required=True, help="Path to RangeViewDataset cache directory")
    ap.add_argument("--out", required=True, help="Destination .pt path")
    ap.add_argument("--val-scenes", type=int, nargs="*", default=[],
                    help="Restrict to these scene ids. -1 means 'all frames'")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--thr", type=float, default=SCORE_THRESH,
                    help="Per-class score threshold before decode (default: SCORE_THRESH map)")
    ap.add_argument("--no-yflip", action="store_true",
                    help="Skip Y-flip TTA (default is to run identity + Y-flip merge)")
    a = ap.parse_args()

    device = default_device()

    if a.val_scenes == [-1]:
        # Test cache uses UUID filenames, not seqN_idxM — bypass scene_of by
        # forcing scene 0 for every frame.
        import rangeview.data.dataset as _ds
        _ds.scene_of = lambda p: 0
        keep = [0]
    else:
        keep = a.val_scenes or None

    ds = RangeViewDataset(a.cache, residuals=False, keep_scenes=keep)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, collate_fn=collate)
    print(f"loaded {len(ds)} frames from {a.cache}", flush=True)

    model = load_checkpoint(a.ckpt, device)
    print(f"running inference (yflip={not a.no_yflip})…", flush=True)
    preds = infer_dataset(model, dl, device, a.thr, yflip=not a.no_yflip)
    preds_cpu = [{k: v.detach().cpu() for k, v in d.items()} for d in preds]

    gts = []
    for batch in dl:
        for i in range(len(batch["gt_boxes_raw"])):
            gts.append((batch["gt_boxes_raw"][i].numpy(),
                        batch["gt_class_raw"][i].numpy()))

    file_names = [f.name for f in ds.files]
    scene_ids  = [0 if keep == [0] else _scene_id_of(f) for f in ds.files]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"preds":      preds_cpu,
                "gts":        gts,
                "scene_ids":  scene_ids,
                "file_names": file_names},
               out_path)
    print(f"saved -> {out_path}", flush=True)


def _scene_id_of(path):
    """Extract scene id from a `seqN_idxM.npz` filename; 0 if not that shape."""
    import re
    m = re.match(r"seq(\d+)_", Path(path).name)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    main()
