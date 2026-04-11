"""Training entry point — design002: 2D Heatmap + Scalar Depth (80×48 upsampled grid).

Head outputs (u_norm, v_norm, z_rel) per joint using native 40×24 heatmap
bilinearly upsampled to 80×48 before softmax. Loss computed in UV+Z space.
MPJPE computed after decoding back to root-relative metres.
"""

from __future__ import annotations

import math
import os
import resource
import sys
import time
from pathlib import Path

# ── Raise the open-file-descriptor limit ─────────────────────────────────────
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

# ── Ensure sibling modules (model.py, transforms.py) are importable ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
from torch.amp import GradScaler
from tqdm import tqdm

from infra import (
    NUM_JOINTS,
    get_splits, build_dataloader,
    Logger, IterLogger,
    save_checkpoint, load_checkpoint,
    pose_loss, mpjpe, pelvis_abs_error, BODY_IDX,
)
from model import SapiensPose3D
from transforms import build_train_transform, build_val_transform
from config import get_config


# ── LR SCHEDULE ───────────────────────────────────────────────────────────────

def get_lr_scale(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Linear warmup then cosine decay."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── DECODE HELPER ─────────────────────────────────────────────────────────────

def decode_joints_heatmap(pred_uvz, K, img_h, img_w, pelvis_abs):
    """Convert (u_norm, v_norm, z_rel) → root-relative (x,y,z) metres."""
    pred_u = pred_uvz[:, :, 0] * img_w  # u_px
    pred_v = pred_uvz[:, :, 1] * img_h  # v_px
    z_rel  = pred_uvz[:, :, 2]           # z_rel metres
    X_pel  = pelvis_abs[:, 0:1]          # (B, 1) pelvis depth
    X_ref  = X_pel + z_rel               # (B, 70) approx world X
    cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
    fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]
    x_rel = -(pred_u - cx) * X_ref / fx   # (B, 70)
    y_rel = -(pred_v - cy) * X_ref / fy
    return torch.stack([x_rel, y_rel, z_rel], dim=-1)  # (B, 70, 3)


# ── TRAIN LOOP ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args,
                    iter_logger=None) -> dict:
    model.train()
    optimizer.zero_grad()
    total = {k: 0.0 for k in ("loss", "pose", "mpjpe_body", "pelvis_err", "weighted")}
    n = 0
    pbar = tqdm(loader, total=args.max_batches or len(loader),
                desc=f"Epoch {epoch} [train]", leave=True, dynamic_ncols=True)

    for i, batch in enumerate(pbar):
        rgb           = batch["rgb"].to(device, non_blocking=True)
        depth         = batch["depth"].to(device, non_blocking=True)
        joints        = batch["joints"].to(device, non_blocking=True)
        gt_pelvis_abs = batch["pelvis_abs"].to(device, non_blocking=True)   # (B, 3)
        gt_pd         = batch["pelvis_depth"].to(device, non_blocking=True) # (B, 1)
        gt_uv         = batch["pelvis_uv"].to(device, non_blocking=True)    # (B, 2)
        K             = batch["intrinsic"].to(device, non_blocking=True)    # (B, 3, 3)
        x = torch.cat([rgb, depth], dim=1)

        # Convert GT joints (B, 70, 3) root-relative metres → normalized UV
        with torch.no_grad():
            X_ref      = gt_pelvis_abs[:, 0:1].unsqueeze(1) + joints[:, :, 2:3]  # (B, 70, 1)
            u_px       = K[:, 0:1, 0:1] * (-joints[:, :, 0:1]) / X_ref + K[:, 0:1, 2:3]
            v_px       = K[:, 1:2, 1:2] * (-joints[:, :, 1:2]) / X_ref + K[:, 1:2, 2:3]
            u_norm     = u_px / args.img_w
            v_norm     = v_px / args.img_h
            gt_uv_joints = torch.cat([u_norm, v_norm], dim=-1)  # (B, 70, 2)

        with torch.amp.autocast("cuda", enabled=args.amp):
            out    = model(x)
            l_xy   = pose_loss(out["joints"][:, BODY_IDX, :2], gt_uv_joints[:, BODY_IDX])
            l_z    = pose_loss(out["joints"][:, BODY_IDX, 2:3], joints[:, BODY_IDX, 2:3])
            l_pose = l_xy + args.lambda_z_joint * l_z
            l_dep  = pose_loss(out["pelvis_depth"], gt_pd)
            l_uv   = pose_loss(out["pelvis_uv"],    gt_uv)
            loss   = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % args.accum_steps == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            pf_decoded = decode_joints_heatmap(out["joints"].float(), K, args.img_h, args.img_w, gt_pelvis_abs)
            body = mpjpe(pf_decoded, joints, BODY_IDX).item()
            pe   = pelvis_abs_error(out["pelvis_depth"].float(), out["pelvis_uv"].float(),
                                    gt_pelvis_abs, K, args.img_h, args.img_w).item()
            total["loss"]       += loss.item() * args.accum_steps
            total["pose"]       += l_pose.item()
            total["mpjpe_body"] += body
            total["pelvis_err"] += pe
            total["weighted"]   += 0.67 * body + 0.33 * pe
            if iter_logger is not None:
                iter_logger.log({"epoch": epoch, "iter": n,
                                 "loss": loss.item() * args.accum_steps,
                                 "loss_pose": l_pose.item(),
                                 "loss_depth": l_dep.item(),
                                 "loss_uv": l_uv.item(),
                                 "mpjpe_body": body, "pelvis_err": pe,
                                 "mpjpe_weighted": 0.67 * body + 0.33 * pe})
        del x, out, l_xy, l_z, l_pose, l_dep, l_uv, loss
        n += 1
        if n == 1:
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            reserv = torch.cuda.memory_reserved(device) / 1024**3
            pbar.write(f"[GPU mem after batch 1] allocated={alloc:.2f}GB  reserved={reserv:.2f}GB")
        pbar.set_postfix(loss=f"{total['loss']/n:.4f}",
                         body=f"{total['mpjpe_body']/n:.1f}mm",
                         pelvis=f"{total['pelvis_err']/n:.1f}mm",
                         w=f"{total['weighted']/n:.1f}mm")
        if args.max_batches > 0 and n >= args.max_batches:
            break

    pbar.close()
    n = max(1, n)
    return {
        "train_loss":        total["loss"] / n,
        "train_loss_pose":   total["pose"] / n,
        "train_mpjpe_body":  total["mpjpe_body"] / n,
        "train_pelvis_err":  total["pelvis_err"] / n,
        "train_mpjpe_weighted": total["weighted"] / n,
    }


@torch.no_grad()
def validate_heatmap(model, loader, device, args) -> dict:
    """Validation with heatmap decode: out['joints'] is (u,v,z) space."""
    model.eval()
    total = {k: 0.0 for k in ("loss", "pose", "body", "pelvis_err", "weighted")}
    n = 0
    pbar = tqdm(loader, total=args.max_batches or len(loader),
                desc="         [val] ", leave=True, dynamic_ncols=True)

    for batch in pbar:
        rgb           = batch["rgb"].to(device, non_blocking=True)
        depth         = batch["depth"].to(device, non_blocking=True)
        joints        = batch["joints"].to(device, non_blocking=True)
        gt_pelvis_abs = batch["pelvis_abs"].to(device, non_blocking=True)
        gt_pd         = batch["pelvis_depth"].to(device, non_blocking=True)
        gt_uv         = batch["pelvis_uv"].to(device, non_blocking=True)
        K             = batch["intrinsic"].to(device, non_blocking=True)
        x = torch.cat([rgb, depth], dim=1)

        with torch.amp.autocast("cuda", enabled=args.amp):
            out = model(x)

        pf_decoded = decode_joints_heatmap(out["joints"].float(), K, args.img_h, args.img_w, gt_pelvis_abs)

        # Loss in UV+Z space
        X_ref  = gt_pelvis_abs[:, 0:1].unsqueeze(1) + joints[:, :, 2:3]
        u_px   = K[:, 0:1, 0:1] * (-joints[:, :, 0:1]) / X_ref + K[:, 0:1, 2:3]
        v_px   = K[:, 1:2, 1:2] * (-joints[:, :, 1:2]) / X_ref + K[:, 1:2, 2:3]
        gt_uv_joints = torch.cat([u_px / args.img_w, v_px / args.img_h], dim=-1)
        l_xy   = pose_loss(out["joints"][:, BODY_IDX, :2], gt_uv_joints[:, BODY_IDX])
        l_z    = pose_loss(out["joints"][:, BODY_IDX, 2:3], joints[:, BODY_IDX, 2:3])
        l_pose = (l_xy + args.lambda_z_joint * l_z).item()

        body   = mpjpe(pf_decoded, joints, BODY_IDX).item()
        pe     = pelvis_abs_error(out["pelvis_depth"].float(), out["pelvis_uv"].float(),
                                  gt_pelvis_abs, K, args.img_h, args.img_w).item()

        total["loss"]       += l_pose
        total["pose"]       += l_pose
        total["body"]       += body
        total["pelvis_err"] += pe
        total["weighted"]   += 0.67 * body + 0.33 * pe
        del x, out, pf_decoded
        n += 1
        pbar.set_postfix(body=f"{total['body']/n:.1f}mm",
                         pelvis=f"{total['pelvis_err']/n:.1f}mm",
                         w=f"{total['weighted']/n:.1f}mm")
        if args.max_batches > 0 and n >= args.max_batches:
            break

    pbar.close()
    n = max(1, n)
    return {
        "val_loss":              total["loss"] / n,
        "val_loss_pose":         total["pose"] / n,
        "val_mpjpe_body":        total["body"] / n,
        "val_pelvis_err":        total["pelvis_err"] / n,
        "val_mpjpe_weighted":    total["weighted"] / n,
    }


def main():
    # ── Determinism ───────────────────────────────────────────────────────────
    import random
    import numpy as np
    _SEED = 2026
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args    = get_config()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nOutput: {out_dir}")

    # Save a snapshot of this script at the start of training
    import shutil as _shutil
    _shutil.copy(__file__, str(out_dir / "train_snapshot.py"))
    print(f"  Saved script snapshot → {out_dir}/train_snapshot.py")

    # Data
    print("\nBuilding data splits ...")
    if args.splits_file:
        import json as _json
        with open(args.splits_file) as _f:
            _sp = _json.load(_f)
        train_seqs = _sp["train"]
        val_seqs   = _sp["val"]
        print(f"  Loaded splits from {args.splits_file}")
    else:
        train_seqs, val_seqs, _ = get_splits(
            overview_path=str(Path(args.data_root) / "data" / "overview.txt"),
            val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
            single_body_only=args.single_body_only, skip_missing_body=True,
            depth_required=True, mp4_required=False,
            frames_root=str(Path(args.data_root) / "data" / "frames"),
        )
    if args.max_train_seqs > 0:
        train_seqs = train_seqs[:args.max_train_seqs]
    if args.max_val_seqs > 0:
        val_seqs = val_seqs[:args.max_val_seqs]
    print(f"  Sequences — train: {len(train_seqs)}, val: {len(val_seqs)}")

    train_tf = build_train_transform(args.img_h, args.img_w)
    val_tf   = build_val_transform(args.img_h, args.img_w)
    train_loader = build_dataloader(train_seqs, args.data_root, train_tf,
                                    batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    val_loader   = build_dataloader(val_seqs,   args.data_root, val_tf,
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
    print(f"  Frames  — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    # Model
    print(f"\nBuilding {args.arch} ...")
    model = SapiensPose3D(
        arch=args.arch, img_size=(args.img_h, args.img_w), num_joints=NUM_JOINTS,
        head_hidden=args.head_hidden, head_num_heads=args.head_num_heads,
        head_num_layers=args.head_num_layers, head_dropout=args.head_dropout,
        drop_path_rate=args.drop_path, num_depth_bins=args.num_depth_bins,
        heatmap_h=args.heatmap_h, heatmap_w=args.heatmap_w,
        upsample_factor=args.upsample_factor,
    ).to(device)
    if args.pretrain and not args.resume:
        model.load_pretrained(args.pretrain)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── LLRD helpers ────────────────────────────────────────────────────────
    NUM_BLOCKS = 24
    GAMMA = args.llrd_gamma
    BASE_LR = args.base_lr_backbone
    UNFREEZE_EPOCH = args.unfreeze_epoch

    def _block_lr(block_idx: int) -> float:
        """LR for ViT block `block_idx` (0=shallowest, 23=deepest)."""
        return BASE_LR * (GAMMA ** (NUM_BLOCKS - 1 - block_idx))

    def _embed_lr() -> float:
        return BASE_LR * (GAMMA ** NUM_BLOCKS)

    # Collect named parameter sets
    depth_pe_params = list(model.backbone.depth_bucket_pe.parameters())

    vit = model.backbone.vit

    def _build_optimizer_frozen():
        """Epochs 0 .. UNFREEZE_EPOCH-1: freeze blocks 0-11 + embed; train 12-23 + depth_pe + head."""
        for p in vit.patch_embed.parameters():
            p.requires_grad = False
        for i in range(12):
            for p in vit.layers[i].parameters():
                p.requires_grad = False
        for i in range(12, NUM_BLOCKS):
            for p in vit.layers[i].parameters():
                p.requires_grad = True
        for p in depth_pe_params:
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True

        groups = []
        for i in range(12, NUM_BLOCKS):
            groups.append({"params": list(vit.layers[i].parameters()), "lr": _block_lr(i)})
        groups.append({"params": depth_pe_params, "lr": args.lr_depth_pe})
        groups.append({"params": list(model.head.parameters()), "lr": args.lr_head})
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    def _build_optimizer_full():
        """Epochs UNFREEZE_EPOCH .. end: all params trainable with LLRD."""
        for p in model.parameters():
            p.requires_grad = True

        groups = []
        embed_params = list(vit.patch_embed.parameters())
        groups.append({"params": embed_params, "lr": _embed_lr()})
        for i in range(NUM_BLOCKS):
            groups.append({"params": list(vit.layers[i].parameters()), "lr": _block_lr(i)})
        groups.append({"params": depth_pe_params, "lr": args.lr_depth_pe})
        groups.append({"params": list(model.head.parameters()), "lr": args.lr_head})
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    optimizer = _build_optimizer_frozen()
    scaler = GradScaler("cuda", enabled=args.amp)

    # Resume
    start_epoch = 0
    best_mpjpe  = float("inf")
    if args.resume:
        start_epoch, best_mpjpe = load_checkpoint(model, optimizer, scaler, args.resume, device)

    logger      = Logger(str(out_dir / "metrics.csv"))
    iter_logger = IterLogger(str(out_dir / "iter_metrics.csv"))

    print(f"\nStarting training for {args.epochs} epochs ...\n")
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]
    patience_counter = 0
    final_val_body = float("nan")

    for epoch in range(start_epoch, args.epochs):
        # Rebuild optimizer at unfreeze epoch
        if epoch == UNFREEZE_EPOCH:
            print(f"  *** Unfreezing all backbone layers at epoch {epoch+1} ***")
            optimizer = _build_optimizer_full()
            for g in optimizer.param_groups:
                g["initial_lr"] = g["lr"]

        scale = get_lr_scale(epoch, args.epochs, args.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale

        # Report LR: deepest block (block 23)
        # Frozen phase: group 0=block12, ..., group 11=block23, group 12=depth_pe, group 13=head
        # Full phase: group 0=embed, groups 1-24=blocks 0-23, group 25=depth_pe, group 26=head
        if epoch < UNFREEZE_EPOCH:
            lr_bb = optimizer.param_groups[11]["lr"]   # block 23
            lr_hd = optimizer.param_groups[13]["lr"]   # head
        else:
            lr_bb = optimizer.param_groups[24]["lr"]   # block 23
            lr_hd = optimizer.param_groups[26]["lr"]   # head
        print(f"Epoch {epoch+1}/{args.epochs}  lr_backbone={lr_bb:.2e}  lr_head={lr_hd:.2e}")

        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args,
                                  iter_logger=iter_logger)

        val_m = None
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            val_m = validate_heatmap(model, val_loader, device, args)
            torch.cuda.empty_cache()
            model.train()

        epoch_time = time.time() - t0
        print(f"  → loss={train_m['train_loss']:.4f}"
              f"  body={train_m['train_mpjpe_body']:.1f}mm"
              f"  pelvis={train_m['train_pelvis_err']:.1f}mm"
              f"  w={train_m['train_mpjpe_weighted']:.1f}mm", end="")
        if val_m:
            print(f"  | val body={val_m['val_mpjpe_body']:.1f}mm"
                  f"  pelvis={val_m['val_pelvis_err']:.1f}mm"
                  f"  w={val_m['val_mpjpe_weighted']:.1f}mm", end="")
        print(f"  ({epoch_time:.0f}s)\n")

        logger.log({"epoch": epoch + 1, "lr_backbone": lr_bb, "lr_head": lr_hd,
                    **train_m, **(val_m or {}), "epoch_time": epoch_time})

        if val_m:
            final_val_body = val_m["val_mpjpe_body"]
            if val_m["val_mpjpe_weighted"] < best_mpjpe:
                best_mpjpe = val_m["val_mpjpe_weighted"]
                patience_counter = 0
                print(f"  *** New best weighted MPJPE = {best_mpjpe:.1f}mm"
                      f"  (body={val_m['val_mpjpe_body']:.1f}  pelvis={val_m['val_pelvis_err']:.1f}) ***\n")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    logger.close()
    iter_logger.close()
    print(f"Training complete. Best val weighted MPJPE = {best_mpjpe:.1f}mm")
    print(f"Checkpoints in {out_dir}/")
    print(final_val_body)


if __name__ == "__main__":
    main()
