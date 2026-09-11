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

from dataset import RangeViewDataset, collate
from model import RangeViewFusionNet, fold_gate_
from loss import compute_loss
from ema import ModelEMA
from evaluate import evaluate as run_eval_ours
from evaluate_official import evaluate_official as run_eval

# With --ped-hires the fine heatmap emits both members of a crowded pedestrian
# pair, but two 0.94 x 0.85 m boxes 0.4 m apart have BEV IoU 0.40, so a flat NMS
# of 0.2 deletes one again and the mAP curve would show no gain even when the
# change works. Measured: at 0.3 m and 0.5 m separation, NMS 0.2 and 0.4 keep
# one, NMS 0.55 keeps both. Vehicle/Bike stay at 0.2 -- they never crowd (0.0%
# within 2 m), so a looser threshold there could only admit duplicates.
PED_HIRES_NMS = {1: 0.2, 2: 0.2, 3: 0.55}
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
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
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
    ap.add_argument("--cache", default="/home/mehul/mixed_signals/rangeview/cache")
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                    help="bf16 (default) has fp32's exponent range, so it needs "
                         "no GradScaler and cannot hit the scaler-collapse "
                         "failure mode. Measured on the GroupNorm BEV model over "
                         "250 steps: bf16 0/250 non-finite vs fp16 6/250, "
                         "identical loss and memory, 1.22 vs 1.28 it/s.")
    ap.add_argument("--out", default="/home/mehul/mixed_signals/rangeview/checkpoints")
    ap.add_argument("--tb-dir", default="/home/mehul/mixed_signals/rangeview/tb")
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
    ap.add_argument("--fourier", action="store_true",
                    help="Fourier Azimuth Enhancer (FlowFM FreEnc, CVPR'26): FFT "
                         "along the periodic azimuth axis, enhance the magnitude "
                         "spectrum, keep the original phase, inverse FFT. Gives a "
                         "GLOBAL azimuth receptive field at O(W log W). +3,137 "
                         "params, zero-init gate.")
    ap.add_argument("--per-agent-intensity", action="store_true",
                    help="standardise log1p(intensity) with PER-AGENT statistics. "
                         "Under the global constants the channel lands at mean "
                         "-0.39/std 0.32 for `top` but +1.08/1.33 for `laser`, and "
                         "at a constant -0.58 for 003/004 which report no intensity "
                         "at all. Changes the input distribution, so it is recorded "
                         "in the checkpoint and old checkpoints keep the old path.")
    ap.add_argument("--bg-mode", choices=["both","intensity","range"], default="both",
                    help="which static-background residuals to feed. 'range' is "
                         "what measured negative on Bike; 'intensity' keeps only "
                         "the reflectance residual, which a parked bike still "
                         "shows because it reflects differently than the road.")
    ap.add_argument("--z-ceiling", type=float, default=None,
                    help="height ceiling for the BEV scatter, in ego z. "
                         "Ground is -4.18, so -0.8 is 3.4 m above it: drops ~15%% "
                         "of points and 0.00%% of points inside any GT box. "
                         "Stops tree canopy winning scatter_max in cells that "
                         "also contain an object. The range encoder is unaffected.")
    ap.add_argument("--density", action="store_true",
                    help="append a log1p point-count channel to the BEV grid. "
                         "scatter_max discards how many points hit a cell; true "
                         "vehicles carry a median 1173 points per box against 96 "
                         "for vehicle false positives, a 12x cue the encoder "
                         "currently cannot see. +1 channel.")
    ap.add_argument("--hires", type=int, nargs="*", default=None,
                    help="classes to give a 0.2 m heatmap (1=Vehicle 2=Bike "
                         "3=Pedestrian). '--hires 2 3' = bike + pedestrian. "
                         "Overrides --ped-hires, which is equivalent to '--hires 3'.")
    ap.add_argument("--ped-hires", action="store_true",
                    help="add a 0.2 m PEDESTRIAN-ONLY heatmap. 30.8%% of "
                         "pedestrians have a neighbour <1.0 m and recall there "
                         "is 73.7%% vs 98.9%% beyond; at 0.4 m their Gaussians "
                         "merge into one blob. Vehicle/Bike never crowd (0.0%% "
                         "under 2 m) and are untouched. +432 MB, +29k params.")
    ap.add_argument("--attn", action="store_true",
                    help="Range-Adaptive Azimuth Attention in the encoder: the "
                         "attention window scales as 1/r so every pixel attends "
                         "over a constant PHYSICAL extent, gated by 3D distance "
                         "and biased by the 3D relative offset. +4,229 params, "
                         "zero-init residual gate. See range_attention.py.")
    ap.add_argument("--vrm", type=int, default=0, metavar="DELTA",
                    help="Vision Restoration Module (RangePerception): wrap DELTA "
                         "azimuth columns onto both margins before the encoder and "
                         "prune them after. 0 = off, 16 = paper-style restoration. "
                         "Adds no parameters, so it is recorded in the checkpoint.")
    ap.add_argument("--background", action="store_true",
                    help="static-background subtraction for the two fixed RSUs: "
                         "signed residual (background_range - range) plus a "
                         "validity flag, +2 channels. Needs background.npz.")
    ap.add_argument("--no-vfe", action="store_true",
                    help="disable PillarVFE (measured a wash: 0.7856 vs 0.7913 "
                         "at epoch 4, for 11%% slower training and +0.5 GB)")
    ap.add_argument("--gt-paste", nargs="+", type=int, default=None, metavar="CLS",
                    help="GT-paste augmentation for these class ids, e.g. --gt-paste 2 3. "
                         "Duplicates objects at new bearings to raise instance "
                         "diversity (153 distinct Bikes vs 660 Vehicles)")
    ap.add_argument("--gt-paste-n", type=int, default=3,
                    help="max successful pastes per frame")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable train-time augmentation (for ablation)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="weight EMA decay; 0 disables. Evaluated instead of the raw "
                         "weights, which oscillate under the near-constant LR schedule")
    ap.add_argument("--log-every", type=int, default=50, help="steps between TB train logs")
    ap.add_argument("--val-every", type=int, default=250,
                    help="steps between mid-epoch val-LOSS checks (subset)")
    ap.add_argument("--map-every", type=int, default=1000,
                    help="steps between mid-epoch val-mAP checks (subset); 0 disables")
    ap.add_argument("--val-subset", type=int, default=80,
                    help="frames used for the mid-epoch checks (fixed, deterministic)")
    ap.add_argument("--gt-infill", nargs="?", const="gt_infill.npz", default=None,
                    metavar="NPZ",
                    help="restore GT boxes the released annotation drops for <=3 "
                         "frames while the object is still visible (>5 points). "
                         "Applied to TRAINING only; validation keeps the released "
                         "labels so local numbers stay comparable to the leaderboard.")
    ap.add_argument("--pseudo", nargs="?", const="pseudo_labels.npz", default=None,
                    help="pseudo-label sidecar for the unlabelled val scenes; adds those "
                         "frames to training with an ignore band so a missed object is "
                         "not supervised as background")
    ap.add_argument("--pseudo-cache", default="cache_val",
                    help="cache dir holding the pseudo-labelled frames")
    ap.add_argument("--fusion-v1", action="store_true",
                    help="force the OLD agent fusion (top-slot readout, no "
                         "per-cell occupancy, fixed logit scale). New runs "
                         "default to v2; use this only for an A/B.")
    ap.add_argument("--ckpt", default=None,
                    help="path to checkpoint to load")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    val_full = val_quick = None
    if args.overfit:
        ds = RangeViewDataset(args.cache, geometry=args.geometry,
                              background=args.background, ped_hires=args.ped_hires, hires=args.hires)
        ds.files = ds.files[: args.overfit]
        print(f"OVERFIT MODE: {len(ds.files)} frames")
    else:
        ds = RangeViewDataset(args.cache, geometry=args.geometry,
                              background=args.background,
                              drop_scenes=args.val_scenes,
                              residuals=not args.no_residuals,
                              augment=not args.no_augment,
                              aug_kwargs={"agent_drop": args.agent_drop,
                                          "beam_drop": args.beam_drop,
                                          "pixel_drop": args.pixel_drop},
                              gt_paste=({"classes": tuple(args.gt_paste),
                                         "n_max": args.gt_paste_n}
                                        if args.gt_paste else None),
                              gt_infill=args.gt_infill,
                              pseudo=args.pseudo, pseudo_cache=args.pseudo_cache,
                              ped_hires=args.ped_hires, hires=args.hires)
        # NOTE: no gt_infill on the validation set. The leaderboard scores
        # against the released labels, dropout included, so val must too --
        # otherwise the local number measures a target nobody grades.
        val_ds = RangeViewDataset(args.cache, geometry=args.geometry,
                                  background=args.background,
                                  keep_scenes=args.val_scenes,
                                  residuals=not args.no_residuals, ped_hires=args.ped_hires, hires=args.hires)
        print(f"train {len(ds)} frames | val {len(val_ds)} frames (scenes {args.val_scenes}) "
              f"| augment={not args.no_augment}")

        mk = lambda d: DataLoader(d, batch_size=args.batch_size, shuffle=False,
                                  num_workers=max(2, args.workers // 2), collate_fn=collate,
                                  pin_memory=(device.type == "cuda"))
        val_full = mk(val_ds)
        # evenly spaced subset -> covers both val scenes, and is identical at
        # every check so the curve is comparable step to step
        step = max(1, len(val_ds) // args.val_subset)
        val_quick = mk(Subset(val_ds, list(range(0, len(val_ds), step))[: args.val_subset]))
        print(f"mid-epoch checks: val loss every {args.val_every} steps, "
              f"val mAP every {args.map_every} steps, on {len(val_quick.dataset)} frames")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, pin_memory=(device.type == "cuda"), drop_last=True)

    if args.geometry:
        import decode as _dec
        _dec.set_ground_map('ground_map.npz')
    model = RangeViewFusionNet(use_vfe=not args.no_vfe,
                               use_geometry=args.geometry,
                               use_background=args.background,
                               vrm=args.vrm, attn=args.attn,
                               fourier=args.fourier, ped_hires=args.ped_hires, hires=args.hires,
                               density=args.density, z_ceiling=args.z_ceiling,
                               bg_mode=args.bg_mode,
                               per_agent_intensity=args.per_agent_intensity,
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
          f"| geometry={args.geometry} | background={args.background} "
          f"| vrm={args.vrm} | attn={args.attn} | fourier={args.fourier} "
          f"| hires={args.hires if args.hires is not None else ((3,) if args.ped_hires else ())} "
          f"| density={args.density} "
          f"| z_ceiling={args.z_ceiling} "
          f"| per_agent_int={args.per_agent_intensity} | bg_mode={args.bg_mode} "
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
        fold_gate_(_sd)
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

    # AgentAttentionFusion auto-detects v1 from a pre-v2 state dict so that old
    # checkpoints reproduce their exact numbers at inference. That is the wrong
    # default when RESUMING to train: without this, --ckpt <old> would silently
    # pin the run to v1 for its entire duration. Re-assert the mode explicitly.
    from agent_fusion import AgentAttentionFusion
    _fus = [m for m in model.modules() if isinstance(m, AgentAttentionFusion)]
    for m in _fus:
        m.v2 = not args.fusion_v1
    print(f"agent fusion: {'v1 (legacy)' if args.fusion_v1 else 'v2'} "
          f"on {len(_fus)} fusion points"
          + ("" if not args.ckpt else "  [re-asserted after --ckpt resume]"),
          flush=True)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"weight EMA enabled (decay {args.ema_decay}) -- validation and "
              f"checkpoints use the EMA weights", flush=True)

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
    # GradScaler is only meaningful for fp16; bf16 has the range to go without it
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype is torch.float16)

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

            with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
                out = model(batch)
            # Loss in fp32, deliberately outside autocast. In fp16 the focal
            # loss's clamp is silently useless: its upper bound 1-1e-4 = 0.9999
            # is not representable (spacing near 1.0 is 0.000977), so it rounds
            # to exactly 1.0 and log(1-p) becomes -inf the moment the model is
            # confident anywhere. The forward pass stays half precision.
            out = {k: v.float() for k, v in out.items()}
            losses = compute_loss(out, batch, iou_aware=args.iou_aware)

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
                                    score_thresh=0.30, verbose=False,
                                    nms_iou=PED_HIRES_NMS if args.ped_hires else 0.2)
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
                                  device, score_thresh=0.30, verbose=False,
                                  nms_iou=PED_HIRES_NMS if args.ped_hires else 0.2)
                writer.add_scalar("epoch_val/mAP_full", fm, epoch)
                for c, v in fa.items():
                    writer.add_scalar(f"epoch_val/AP_full_class{c}", v, epoch)
                line += (f" | FULL mAP {fm:.4f} (Veh {fa[1]:.3f} "
                         f"Bike {fa[2]:.3f} Ped {fa[3]:.3f}) "
                         f"[{len(val_full.dataset)} frames]")

        print(f"{line} | {time.time()-t0:.0f}s", flush=True)
        ckpt = {"model": model.state_dict(), "epoch": epoch, "vrm": args.vrm, "attn": args.attn, "fourier": args.fourier, "ped_hires": args.ped_hires, "hires": args.hires, "density": args.density, "z_ceiling": args.z_ceiling,
                "bg_mode": args.bg_mode,
                "per_agent_intensity": args.per_agent_intensity,
                "fusion_v2": not args.fusion_v1,
                "gt_infill": bool(args.gt_infill)}
        if ema is not None:
            # the EMA weights go in "model" so every eval script picks them up
            # by default; raw weights kept alongside for comparison
            ckpt = {"model": ema.ema.state_dict(), "raw": model.state_dict(),
                    "epoch": epoch, "vrm": args.vrm, "attn": args.attn, "fourier": args.fourier, "ped_hires": args.ped_hires, "hires": args.hires, "density": args.density, "z_ceiling": args.z_ceiling,
                "bg_mode": args.bg_mode,
                "per_agent_intensity": args.per_agent_intensity,
                "fusion_v2": not args.fusion_v1,
                "gt_infill": bool(args.gt_infill)}
        torch.save(ckpt, out_dir / f"rangeview_epoch{epoch+1}.pth")

    writer.close()
    print("training finished ->", out_dir)


if __name__ == "__main__":
    main()
