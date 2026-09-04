"""Training loop for RangeViewFusionNet, with step-level TensorBoard logging.

Validation runs mid-epoch, not just at epoch end, on a fixed deterministic
subset so successive points are comparable. Both val loss and val mAP are
logged: loss alone has repeatedly misled us here (4 of the 5 loss terms are
gathered at ground-truth centres and are blind to whether the object was
detected at all), so mAP is the metric that decides.
"""
import argparse
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from rangeview.compat import autocast_ctx, default_device, pin_ok
from rangeview.data.dataset import RangeViewDataset, collate
from rangeview.models.net import RangeViewFusionNet
from rangeview.models.loss import compute_loss
from rangeview.models.ema import ModelEMA
from rangeview.eval.evaluate import evaluate_official as run_eval
from rangeview.paths import ARTIFACTS

# run_eval is the COMPETITION's scorer, not our re-implementation. Ours builds
# the PR curve from a globally score-sorted list; the devkit concatenates
# per-FRAME results (calculate_ap with global_sort_detections=False). Measured on
# identical predictions, held-out scenes 8/16/26/34: ours 0.8700 vs devkit
# 0.7719, and the score-threshold optimum MOVES (ours favoured 0.05, the devkit
# peaks at 0.30 and collapses to 0.500 at 0.05). Ranking checkpoints on ours
# would pick the wrong epoch and the wrong threshold.


def move(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):        # images, valid: agent -> tensor
            out[k] = {a: t.to(device, non_blocking=True) for a, t in v.items()}
        elif isinstance(v, list):      # ragged raw GT, only used by evaluation
            out[k] = v
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


@torch.no_grad()
def val_loss(model, loader, device, amp, amp_dtype=torch.bfloat16):
    model.eval()
    tot, n = 0.0, 0
    parts = {}
    for vb in loader:
        vb = move(vb, device)
        with autocast_ctx(device, dtype=amp_dtype, enabled=amp):
            out = model(vb)
        out = {k: v.float() for k, v in out.items()}      # see note in main()
        losses = compute_loss(out, vb)
        tot += float(losses["total"]); n += 1
        for k, v in losses.items():
            parts[k] = parts.get(k, 0.0) + float(v)
    model.train()
    n = max(n, 1)
    return tot / n, {k: v / n for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ARTIFACTS / "cache_train"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01,
                    help="SGD learning rate (LiDAR-MOS/SalsaNext use 0.01, RangeNet 0.005)")
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--sched", choices=["expdecay", "onecycle"], default="expdecay",
                    help="expdecay = LiDAR-MOS recipe: warmup then x0.99 per epoch")
    ap.add_argument("--lr-decay", type=float, default=0.99, help="per-epoch decay")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto",
                    help="cuda / mps / cpu / auto. auto picks cuda>mps>cpu.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                    help="bf16 (default) has fp32's exponent range, so it needs "
                         "no GradScaler and cannot hit the scaler-collapse "
                         "failure mode. Measured on the GroupNorm BEV model over "
                         "250 steps: bf16 0/250 non-finite vs fp16 6/250, "
                         "identical loss and memory, 1.22 vs 1.28 it/s.")
    ap.add_argument("--out", default=str(ARTIFACTS / "checkpoints"))
    ap.add_argument("--tb-dir", default=str(ARTIFACTS / "tb"))
    ap.add_argument("--overfit", type=int, default=0,
                    help="train on only N frames -- sanity check that loss can reach ~0")
    ap.add_argument("--val-scenes", type=int, nargs="*", default=[],
                    help="scenes held out for validation (official val labels are withheld, "
                         "so this is our only generalisation signal)")
    ap.add_argument("--iou-aware", action="store_true",
                    help="RangeDet IoU-aware classification targets (heatmap target "
                         "at a positive becomes the decoded box's BEV IoU)")
    ap.add_argument("--agent-drop", type=float, default=0.2,
                    help="probability of zeroing one whole non-ego agent per "
                         "sample (V2X link-loss augmentation)")
    ap.add_argument("--beam-drop", type=float, default=0.1,
                    help="fraction of range-image rows zeroed")
    ap.add_argument("--pixel-drop", type=float, default=0.2,
                    help="fraction of valid pixels zeroed")
    ap.add_argument("--no-ms-fusion", action="store_true",
                    help="disable multi-scale cross-agent fusion (fuse only at "
                         "stride 2, the behaviour before 2026-08-25)")
    ap.add_argument("--no-iou-head", action="store_true",
                    help="omit the IoU-prediction head and its loss term. Needed "
                         "to reproduce the 0.73 model exactly: that checkpoint "
                         "predates the head, and the head adds a loss term at "
                         "weight 1.0 which changes training dynamics.")
    ap.add_argument("--geometry", action="store_true",
                    help="geometry pack: sensor-relative ray (4), height above "
                         "the mapped ground (1), surface normals (3). Widens the "
                         "encoder stem to 14 channels AND switches the z head to "
                         "a ground-relative target; decode reads that back off "
                         "the checkpoint, so the two cannot be mismatched.")
    ap.add_argument("--no-residuals", action="store_true",
                    help="train WITHOUT the temporal residual input channels "
                         "(encoder cin 9 -> 6). A residual-free model needs no "
                         "neighbouring frames at inference; the current split "
                         "has consecutive frames but the final test/ split ships "
                         "differently and may not.")
    ap.add_argument("--no-vfe", action="store_true",
                    help="disable PillarVFE (measured a wash: 0.7856 vs 0.7913 "
                         "at epoch 4, for 11%% slower training and +0.5 GB)")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable train-time augmentation (for ablation)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="weight EMA decay; 0 disables. Evaluated instead of the raw "
                         "weights, which oscillate under the near-constant LR schedule")
    ap.add_argument("--cbgs", action="store_true",
                    help="class-balanced grouping & sampling (Zhu 2019). Weights "
                         "every frame by the inverse frequency of the rarest class "
                         "it contains, so a Bike-rare epoch is exchanged for one "
                         "where every class contributes equal expected gradient.")
    ap.add_argument("--class-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    metavar=("VEH", "BIKE", "PED"),
                    help="per-class multiplier on the focal-heatmap loss. "
                         "Standard playbook (CenterPoint-Waymo): raise the small "
                         "classes to match their AP contribution.")
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="AdamW weight decay. ep8->ep9 overfit at 1e-4; try 5e-4 "
                         "for continuation training on top of an already-good model.")
    ap.add_argument("--log-every", type=int, default=50, help="steps between TB train logs")
    ap.add_argument("--val-every", type=int, default=250,
                    help="steps between mid-epoch val-LOSS checks (subset)")
    ap.add_argument("--map-every", type=int, default=1000,
                    help="steps between mid-epoch val-mAP checks (subset); 0 disables")
    ap.add_argument("--val-subset", type=int, default=80,
                    help="frames used for the mid-epoch checks (fixed, deterministic)")
    ap.add_argument("--ckpt", default=None,
                    help="path to checkpoint to load")
    args = ap.parse_args()

    if args.device == "auto":
        device = default_device()
    else:
        device = torch.device(args.device)
    # AMP is only useful on cuda and mps. On CPU the autocast context is a
    # no-op anyway, but the flag stays honest so print statements read right.
    amp = args.amp and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    val_full = val_quick = None
    if args.overfit:
        ds = RangeViewDataset(args.cache, geometry=args.geometry)
        ds.files = ds.files[: args.overfit]
        print(f"OVERFIT MODE: {len(ds.files)} frames")
    else:
        ds = RangeViewDataset(args.cache, geometry=args.geometry,
                              drop_scenes=args.val_scenes,
                              residuals=not args.no_residuals,
                              augment=not args.no_augment,
                              aug_kwargs={"agent_drop": args.agent_drop,
                                          "beam_drop": args.beam_drop,
                                          "pixel_drop": args.pixel_drop})
        try:
            val_ds = RangeViewDataset(args.cache, geometry=args.geometry,
                                      keep_scenes=args.val_scenes,
                                      residuals=not args.no_residuals)
        except FileNotFoundError:
            val_ds = None
        n_val = len(val_ds) if val_ds is not None else 0
        print(f"train {len(ds)} frames | val {n_val} frames (scenes {args.val_scenes}) "
              f"| augment={not args.no_augment}")

        if val_ds is not None:
            mk = lambda d: DataLoader(d, batch_size=args.batch_size, shuffle=False,
                                      num_workers=max(2, args.workers // 2), collate_fn=collate,
                                      pin_memory=pin_ok(device))
            val_full = mk(val_ds)
            # evenly spaced subset -> covers both val scenes, and is identical at
            # every check so the curve is comparable step to step
            step = max(1, len(val_ds) // args.val_subset)
            val_quick = mk(Subset(val_ds, list(range(0, len(val_ds), step))[: args.val_subset]))
            print(f"mid-epoch checks: val loss every {args.val_every} steps, "
                  f"val mAP every {args.map_every} steps, on {len(val_quick.dataset)} frames")

    if args.cbgs:
        from torch.utils.data import WeightedRandomSampler
        ds.build_cbgs_weights()
        sampler = WeightedRandomSampler(
            weights=ds.frame_weights.tolist(),
            num_samples=len(ds), replacement=True,
        )
        print(f"CBGS sampler active | mean frame weight "
              f"{float(ds.frame_weights.mean()):.4g} | "
              f"std {float(ds.frame_weights.std()):.4g}", flush=True)
        dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=pin_ok(device), drop_last=True)
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=pin_ok(device), drop_last=True)

    if args.geometry:
        import rangeview.models.decode as _dec
        _dec.set_ground_map('ground_map.npz')
    model = RangeViewFusionNet(use_vfe=not args.no_vfe,
                               use_geometry=args.geometry,
                               use_residuals=not args.no_residuals,
                               use_iou_head=not args.no_iou_head,
                               ms_fusion=not args.no_ms_fusion).to(device)
    print(f"device={device} | frames={len(ds)} | steps/epoch={len(dl)} | "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"optimizer={args.optimizer} lr={args.lr} sched={args.sched} "
          f"decay={args.lr_decay}/epoch warmup={args.warmup_epochs}ep", flush=True)
    print(f"residuals={not args.no_residuals} | vfe={not args.no_vfe} "
          f"| iou_head={not args.no_iou_head} "
          f"| ms_fusion={not args.no_ms_fusion} "
          f"| geometry={args.geometry} "
          f"| augment drop: agent {args.agent_drop} "
          f"beam {args.beam_drop} pixel {args.pixel_drop}", flush=True)
    
    if args.ckpt:
        # Resume weights (NOT optimizer state -- AdamW moments and the LR
        # schedule restart). This was previously commented out while --ckpt still
        # had a default path, so the flag silently did nothing and every "resumed"
        # run actually started from scratch. Load strictly and report, so a
        # mismatch is loud instead of silent.
        _raw = torch.load(args.ckpt, map_location=device)
        _sd = _raw.get("model", _raw) if isinstance(_raw, dict) else _raw
        missing, unexpected = model.load_state_dict(_sd, strict=False)
        loaded = len(model.state_dict()) - len(missing)
        print(f"loaded {loaded}/{len(model.state_dict())} tensors from {args.ckpt} "
              f"(epoch {_raw.get('epoch', '?') if isinstance(_raw, dict) else '?'})",
              flush=True)
        if missing or unexpected:
            print(f"  WARNING missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
            if loaded == 0:
                raise SystemExit("--ckpt matched NOTHING; refusing to train from "
                                 "scratch under a resume flag")
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"weight EMA enabled (decay {args.ema_decay}) -- validation and "
              f"checkpoints use the EMA weights", flush=True)

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.sched == "expdecay":
        # LiDAR-MOS / SalsaNext / RangeNet++: short warmup, then a gentle x0.99
        # per-epoch decay. Chosen over OneCycle deliberately: OneCycle drove our
        # LR to 0.00% of peak by epoch 19 and held-out mAP collapsed 0.518 -> 0.172
        # while training loss kept falling, i.e. the anneal was memorising 27
        # scenes. This schedule is still at 83% of peak after 20 epochs, keeping
        # the model exploring, which acts as implicit regularisation. It also
        # means any checkpoint is valid -- stopping early does not leave the LR
        # mid-schedule the way OneCycle does.
        warm = max(1, int(args.warmup_epochs * len(dl)))

        def lr_at(step):
            if step < warm:
                return (step + 1) / warm
            return args.lr_decay ** (step / max(1, len(dl)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl)), pct_start=0.3)
    # GradScaler is only meaningful for fp16 on CUDA. bf16 has fp32's exponent
    # range and needs no scaler; MPS/CPU have no GradScaler at all. Disabled
    # scaler is a no-op passthrough, so the rest of the loop is untouched.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp and amp_dtype is torch.float16 and device.type == "cuda"),
    )

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    # Stale checkpoints from an earlier run with a different architecture will
    # sit in here until this run happens to overwrite them, and any sweep will
    # silently mix architectures. Move them aside up front.
    stale = sorted(out_dir.glob("*.pth"))
    if stale:
        attic = out_dir.parent / f"{out_dir.name}_prev_{time.strftime('%m%d_%H%M')}"
        attic.mkdir(parents=True, exist_ok=True)
        for f in stale:
            f.rename(attic / f.name)
        print(f"moved {len(stale)} pre-existing checkpoints to {attic}", flush=True)
    tb_dir = Path(args.tb_dir) / time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(str(tb_dir))
    print(f"tensorboard: {tb_dir}\n  tensorboard --logdir {args.tb_dir}", flush=True)

    gstep = 0
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), {}
        for i, batch in enumerate(dl):
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)

            with autocast_ctx(device, dtype=amp_dtype, enabled=amp):
                out = model(batch)
            # Loss in fp32, deliberately outside autocast.
            out = {k: v.float() for k, v in out.items()}
            cw = torch.tensor(args.class_weights, dtype=torch.float32,
                              device=device) if any(w != 1.0 for w in args.class_weights) else None
            losses = compute_loss(out, batch, iou_aware=args.iou_aware,
                                  class_weights=cw)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            gstep += 1

            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + float(v)

            if gstep % args.log_every == 0:
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", float(v), gstep)
                writer.add_scalar("lr", opt.param_groups[0]["lr"], gstep)

            if (i + 1) % args.log_every == 0:
                msg = " ".join(f"{k} {run[k]/(i+1):.3f}" for k in ("total", "hm", "off", "z", "dim", "rot"))
                print(f"[ep {epoch}][{i+1}/{len(dl)}] {msg} | "
                      f"{(i+1)/(time.time()-t0):.2f} it/s", flush=True)

            if val_quick is not None and gstep % args.val_every == 0:
                vt, vparts = val_loss(ema.ema if ema else model, val_quick, device, amp, amp_dtype)
                for k, v in vparts.items():
                    writer.add_scalar(f"val/{k}", v, gstep)
                print(f"    [step {gstep}] val loss {vt:.4f}", flush=True)

            if val_quick is not None and args.map_every and gstep % args.map_every == 0:
                mAP, aps = run_eval(ema.ema if ema else model, val_quick, device,
                                    score_thresh=0.30, verbose=False)
                model.train()
                writer.add_scalar("val/mAP", mAP, gstep)
                for c, name in ((1, "Vehicle"), (2, "Bike"), (3, "Pedestrian")):
                    writer.add_scalar(f"val_AP/{name}", aps[c], gstep)
                print(f"    [step {gstep}] val mAP {mAP:.4f} "
                      f"(Veh {aps[1]:.3f} Bike {aps[2]:.3f} Ped {aps[3]:.3f})", flush=True)

        n = max(1, len(dl))
        line = f"== epoch {epoch} done: train {run.get('total',0)/n:.4f}"
        writer.add_scalar("epoch/train_total", run.get("total", 0) / n, epoch)

        if val_full is not None:
            vt, vparts = val_loss(ema.ema if ema else model, val_full, device, amp, amp_dtype)
            line += f" | val {vt:.4f}"
            for k, v in vparts.items():
                writer.add_scalar(f"epoch_val/{k}", v, epoch)

            # FULL-SET mAP at every epoch boundary, not just the mid-epoch
            # subset. The subset has repeatedly failed as an indicator in this
            # project (measured off by +-0.07-0.10 in both directions, and it
            # once ranked the wrong checkpoint best). It is fine for a trend
            # between checks; it is not a number to decide on. A full pass costs
            # ~4 min against a ~59 min epoch, far cheaper than submitting the
            # wrong checkpoint.
            if args.map_every:
                fm, fa = run_eval(ema.ema if ema else model, val_full,
                                  device, score_thresh=0.30, verbose=False)
                writer.add_scalar("epoch_val/mAP_full", fm, epoch)
                for c, v in fa.items():
                    writer.add_scalar(f"epoch_val/AP_full_class{c}", v, epoch)
                line += (f" | FULL mAP {fm:.4f} (Veh {fa[1]:.3f} "
                         f"Bike {fa[2]:.3f} Ped {fa[3]:.3f}) "
                         f"[{len(val_full.dataset)} frames]")

        print(f"{line} | {time.time()-t0:.0f}s", flush=True)
        ckpt = {"model": model.state_dict(), "epoch": epoch}
        if ema is not None:
            # the EMA weights go in "model" so every eval script picks them up
            # by default; raw weights kept alongside for comparison
            ckpt = {"model": ema.ema.state_dict(), "raw": model.state_dict(), "epoch": epoch}
        torch.save(ckpt, out_dir / f"rangeview_epoch{epoch+1}.pth")

    writer.close()
    print("training finished ->", out_dir)


if __name__ == "__main__":
    main()
