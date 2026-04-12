"""Training entry point — design003: Cross-Frame Memory Attention (past frozen, no_grad).

Two 4-channel frames (x_t, x_prev) are fed to the same backbone (shared weights).
Both backbone passes are gradient-checkpointed. The projected features are
concatenated along the token axis → 1920-token memory for the decoder.
Validation uses single-frame mode (x_prev=None).
"""

from __future__ import annotations

import math
import os
import random
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

import cv2
import math as _math
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from infra import (
    NUM_JOINTS,
    get_splits, build_dataloader,
    BedlamFrameDataset,
    Logger, IterLogger,
    save_checkpoint, load_checkpoint,
    validate, pose_loss, mpjpe, pelvis_abs_error, BODY_IDX,
    RGB_MEAN, RGB_STD, DEPTH_MAX_METERS,
    collate_fn,
)
from model import SapiensPose3D
from transforms import build_train_transform, build_val_transform
from config import get_config


# ── TEMPORAL DATASET ─────────────────────────────────────────────────────────

def _crop_frame_with_bbox(rgb_raw, depth_raw, bbox, out_h, out_w):
    """Replicate CropPerson crop logic for an auxiliary frame."""
    H, W = rgb_raw.shape[:2]

    cx_box = (bbox[0] + bbox[2]) / 2.0
    cy_box = (bbox[1] + bbox[3]) / 2.0
    w_box  = max(bbox[2] - bbox[0], 1.0)
    h_box  = max(bbox[3] - bbox[1], 1.0)

    target_aspect = out_w / out_h
    if w_box / h_box < target_aspect:
        w_exp = h_box * target_aspect; h_exp = h_box
    else:
        h_exp = w_box / target_aspect; w_exp = w_box

    x0 = cx_box - w_exp / 2.0; y0 = cy_box - h_exp / 2.0
    x1 = cx_box + w_exp / 2.0; y1 = cy_box + h_exp / 2.0

    pad_left   = max(0, int(_math.ceil(-x0)))
    pad_top    = max(0, int(_math.ceil(-y0)))
    pad_right  = max(0, int(_math.ceil(x1 - W)))
    pad_bottom = max(0, int(_math.ceil(y1 - H)))

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        rgb_raw = cv2.copyMakeBorder(
            rgb_raw, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        if depth_raw is not None:
            depth_raw = cv2.copyMakeBorder(
                depth_raw, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0.0
            )
        x0 += pad_left; y0 += pad_top; x1 += pad_left; y1 += pad_top

    ix0, iy0 = int(round(x0)), int(round(y0))
    ix1, iy1 = int(round(x1)), int(round(y1))

    rgb_crop = cv2.resize(
        rgb_raw[iy0:iy1, ix0:ix1], (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    depth_crop = None
    if depth_raw is not None:
        depth_crop = cv2.resize(
            depth_raw[iy0:iy1, ix0:ix1], (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
    return rgb_crop, depth_crop


def _normalise_rgb(rgb_uint8, mean, std):
    x = (rgb_uint8.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def _normalise_depth(depth_f32, depth_max):
    if depth_f32 is None:
        return torch.zeros(1, 1, 1)
    d = np.clip(depth_f32, 0.0, depth_max) / depth_max
    return torch.from_numpy(d[np.newaxis])


class TemporalBedlamDataset(BedlamFrameDataset):
    """Two-frame dataset: centre (t) + past (t-1 in dataset-index = t-5 raw frames).

    Extra batch keys: rgb_prev (3,H,W), depth_prev (1,H,W).
    """

    def __init__(self, seq_paths, data_root, transform=None, depth_required=True):
        super().__init__(seq_paths, data_root, transform=None, depth_required=depth_required)
        self._outer_transform = transform
        self._rgb_mean  = np.array(RGB_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self._rgb_std   = np.array(RGB_STD,  dtype=np.float32).reshape(1, 1, 3)
        self._depth_max = DEPTH_MAX_METERS

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        label_path, body_idx, frame_idx = self.index[idx % len(self.index)]
        cached = self._label_cache.get(label_path)

        past_idx = max(0, frame_idx - 1)

        try:
            rgb_prev = self._read_frame(
                cached["folder_name"], cached["seq_name"], past_idx, label_path
            )
        except Exception:
            rgb_prev = sample["rgb"].copy()

        npy_path = os.path.join(self.data_root, "data", "depth", "npy",
                                cached["folder_name"], f"{cached['seq_name']}.npy")
        npz_path = os.path.join(self.data_root, "data", "depth", "npz",
                                cached["folder_name"], f"{cached['seq_name']}.npz")
        depth_prev = self._read_depth(npy_path, npz_path, past_idx, label_path)
        if depth_prev is None:
            d = sample.get("depth")
            depth_prev = d.copy() if isinstance(d, np.ndarray) else \
                         np.zeros(sample["rgb"].shape[:2], dtype=np.float32)

        sample["_rgb_prev_raw"]   = rgb_prev
        sample["_depth_prev_raw"] = depth_prev

        if self._outer_transform is not None:
            sample = self._outer_transform(sample)

        bbox = self._compute_bbox(self._label_cache[label_path], body_idx, frame_idx)
        rgb_raw = sample.pop("_rgb_prev_raw")
        dep_raw = sample.pop("_depth_prev_raw")

        rgb_t = sample["rgb"]
        out_h, out_w = rgb_t.shape[1], rgb_t.shape[2]

        if bbox is not None:
            H_raw, W_raw = rgb_raw.shape[:2]
            bbox_c = np.clip(
                bbox, [0, 0, 0, 0],
                [float(W_raw), float(H_raw), float(W_raw), float(H_raw)]
            ).astype(np.float32)
            if dep_raw is not None:
                dH, dW = dep_raw.shape[:2]
                if dH != H_raw or dW != W_raw:
                    dep_raw = cv2.resize(dep_raw, (W_raw, H_raw),
                                         interpolation=cv2.INTER_NEAREST)
            rgb_crop, depth_crop = _crop_frame_with_bbox(
                rgb_raw, dep_raw, bbox_c, out_h, out_w
            )
        else:
            rgb_crop  = cv2.resize(rgb_raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            depth_crop = cv2.resize(dep_raw, (out_w, out_h), interpolation=cv2.INTER_NEAREST) \
                if dep_raw is not None else None

        sample["rgb_prev"]   = _normalise_rgb(rgb_crop, self._rgb_mean, self._rgb_std)
        sample["depth_prev"] = _normalise_depth(depth_crop, self._depth_max)

        return sample


def build_temporal_dataloader(seq_paths, data_root, transform, batch_size,
                               shuffle, num_workers):
    dataset = TemporalBedlamDataset(seq_paths=seq_paths, data_root=data_root,
                                     transform=transform, depth_required=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=(2 if num_workers > 0 else None),
        multiprocessing_context=("spawn" if num_workers > 0 else None),
    )


# ── LR SCHEDULE ───────────────────────────────────────────────────────────────

def get_lr_scale(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


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
        rgb        = batch["rgb"].to(device, non_blocking=True)
        depth      = batch["depth"].to(device, non_blocking=True)
        rgb_prev   = batch["rgb_prev"].to(device, non_blocking=True)
        depth_prev = batch["depth_prev"].to(device, non_blocking=True)
        joints        = batch["joints"].to(device, non_blocking=True)
        gt_pelvis_abs = batch["pelvis_abs"].to(device, non_blocking=True)
        gt_pd         = batch["pelvis_depth"].to(device, non_blocking=True)
        gt_uv         = batch["pelvis_uv"].to(device, non_blocking=True)
        K             = batch["intrinsic"].to(device, non_blocking=True)

        x_t    = torch.cat([rgb, depth], dim=1)           # (B, 4, H, W)
        x_prev = torch.cat([rgb_prev, depth_prev], dim=1) # (B, 4, H, W)

        with torch.amp.autocast("cuda", enabled=args.amp):
            out    = model(x_t, x_prev)
            l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])
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
            pf   = out["joints"].float()
            body = mpjpe(pf, joints, BODY_IDX).item()
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
        del x_t, x_prev, out, l_pose, l_dep, l_uv, loss
        n += 1
        if n == 1:
            alloc  = torch.cuda.memory_allocated(device) / 1024**3
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
        "train_loss":           total["loss"] / n,
        "train_loss_pose":      total["pose"] / n,
        "train_mpjpe_body":     total["mpjpe_body"] / n,
        "train_pelvis_err":     total["pelvis_err"] / n,
        "train_mpjpe_weighted": total["weighted"] / n,
    }


def main():
    # ── Determinism ───────────────────────────────────────────────────────────
    import numpy as np
    _SEED = 2026
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.manual_seed_all(_SEED)

    args    = get_config()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nOutput: {out_dir}")
    print(f"Temporal mode: {args.temporal_mode}  (past frame frozen, no_grad)")

    import shutil as _shutil
    _shutil.copy(__file__, str(out_dir / "train_snapshot.py"))
    print(f"  Saved script snapshot -> {out_dir}/train_snapshot.py")

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
    print(f"  Sequences - train: {len(train_seqs)}, val: {len(val_seqs)}")

    train_tf = build_train_transform(args.img_h, args.img_w)
    val_tf   = build_val_transform(args.img_h, args.img_w)

    train_loader = build_temporal_dataloader(train_seqs, args.data_root, train_tf,
                                              batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers)
    val_loader   = build_dataloader(val_seqs, args.data_root, val_tf,
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
    print(f"  Frames  - train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    # Model
    print(f"\nBuilding {args.arch} ...")
    model = SapiensPose3D(
        arch=args.arch, img_size=(args.img_h, args.img_w), num_joints=NUM_JOINTS,
        head_hidden=args.head_hidden, head_num_heads=args.head_num_heads,
        head_num_layers=args.head_num_layers, head_dropout=args.head_dropout,
        drop_path_rate=args.drop_path, num_depth_bins=args.num_depth_bins,
    ).to(device)
    if args.pretrain and not args.resume:
        model.load_pretrained(args.pretrain)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Validation wrapper: single-frame mode
    original_forward = model.forward

    def _val_forward(x4: torch.Tensor) -> dict:
        return original_forward(x4, None)

    # ── LLRD helpers ──────────────────────────────────────────────────────────
    NUM_BLOCKS = 24
    GAMMA = args.llrd_gamma
    BASE_LR = args.base_lr_backbone
    UNFREEZE_EPOCH = args.unfreeze_epoch

    def _block_lr(block_idx: int) -> float:
        return BASE_LR * (GAMMA ** (NUM_BLOCKS - 1 - block_idx))

    def _embed_lr() -> float:
        return BASE_LR * (GAMMA ** NUM_BLOCKS)

    depth_pe_params = list(model.backbone.depth_bucket_pe.parameters())
    vit = model.backbone.vit

    def _build_optimizer_frozen():
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
        if epoch == UNFREEZE_EPOCH:
            print(f"  *** Unfreezing all backbone layers at epoch {epoch+1} ***")
            del optimizer
            import gc; gc.collect()
            torch.cuda.empty_cache()
            optimizer = _build_optimizer_full()
            for g in optimizer.param_groups:
                g["initial_lr"] = g["lr"]

        scale = get_lr_scale(epoch, args.epochs, args.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale

        if epoch < UNFREEZE_EPOCH:
            lr_bb = optimizer.param_groups[11]["lr"]
            lr_hd = optimizer.param_groups[13]["lr"]
        else:
            lr_bb = optimizer.param_groups[24]["lr"]
            lr_hd = optimizer.param_groups[26]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  lr_backbone={lr_bb:.2e}  lr_head={lr_hd:.2e}")

        t0 = time.time()
        model.forward = original_forward
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args,
                                  iter_logger=iter_logger)

        val_m = None
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            model.forward = _val_forward
            val_m = validate(model, val_loader, device, args)
            torch.cuda.empty_cache()
            model.train()
            model.forward = original_forward

        epoch_time = time.time() - t0
        print(f"  -> loss={train_m['train_loss']:.4f}"
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
