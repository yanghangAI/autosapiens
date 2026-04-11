# design001 — 2D Heatmap + Scalar Depth (40×24 native grid)

**Starting point:** `runs/idea014/design003/code/`

## Summary

Replace the final `joints_out = Linear(384, 3)` in `Pose3DHead` with a two-branch output:

1. **2D heatmap branch**: `Linear(384, 40*24=960)` → reshape `(B, 70, 40, 24)` → softmax over H×W → soft-argmax → `(u, v)` in normalized `[0,1]` coordinates → reprojected to root-relative metric `(x, y)` in metres.
2. **Depth branch**: `Linear(384, 1)` predicts scalar root-relative depth z per joint (unchanged from baseline, same head dim).

The trunk (decoder cross-attention, positional encoding, LLRD optimizer, all configs) is **untouched**.

## Problem

Direct 3D regression (`Linear → (x,y,z)`) ignores the spatial structure of the ViT feature map `(1024, 40, 24)`. Soft-argmax over a 2D spatial heatmap is a differentiable proxy for peak localization that naturally leverages the spatial layout of Sapiens features.

## Architecture Change: `model.py`

### New `Pose3DHead`

Replace the existing `joints_out` linear with the heatmap branch. Keep `depth_out` and `uv_out` unchanged.

```python
class Pose3DHead(nn.Module):
    def __init__(self, in_channels, num_joints=NUM_JOINTS, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1,
                 heatmap_h=40, heatmap_w=24):
        super().__init__()
        self.num_joints = num_joints
        self.hm_h = heatmap_h
        self.hm_w = heatmap_w

        self.input_proj   = nn.Linear(in_channels, hidden_dim)
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 2D heatmap branch: Linear(hidden, H*W)
        self.heatmap_out = nn.Linear(hidden_dim, heatmap_h * heatmap_w)

        # Per-joint depth branch (root-relative, metres)
        self.depth_joint_out = nn.Linear(hidden_dim, 1)

        # Pelvis auxiliary outputs (unchanged from baseline)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out    = nn.Linear(hidden_dim, 2)

        # Coordinate buffers for soft-argmax (precomputed, not recomputed each forward)
        coords_u = torch.linspace(0.0, 1.0, heatmap_w)   # (W,)
        coords_v = torch.linspace(0.0, 1.0, heatmap_h)   # (H,)
        grid_v, grid_u = torch.meshgrid(coords_v, coords_u, indexing="ij")
        # Both shapes: (H, W) → flatten to (H*W,)
        self.register_buffer("grid_u", grid_u.reshape(-1))
        self.register_buffer("grid_v", grid_v.reshape(-1))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.depth_joint_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        # Zero-init heatmap bias to avoid degenerate softmax at init
        nn.init.trunc_normal_(self.heatmap_out.weight, std=0.02)
        nn.init.zeros_(self.heatmap_out.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        B = feat.size(0)
        memory  = self.input_proj(feat.flatten(2).transpose(1, 2))  # (B, H*W, hidden)
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)   # (B, 70, hidden)

        # 2D heatmap soft-argmax
        hm_logits = self.heatmap_out(out)           # (B, 70, H*W)
        hm_soft   = torch.softmax(hm_logits, dim=-1)  # (B, 70, H*W)
        pred_u    = (hm_soft * self.grid_u).sum(dim=-1)  # (B, 70) in [0,1]
        pred_v    = (hm_soft * self.grid_v).sum(dim=-1)  # (B, 70) in [0,1]

        # Per-joint depth (root-relative, metres)
        pred_z = self.depth_joint_out(out).squeeze(-1)  # (B, 70)

        # Stack → (B, 70, 3) where columns are [u_norm, v_norm, z_metres]
        joints_out = torch.stack([pred_u, pred_v, pred_z], dim=-1)

        pelvis_token = out[:, 0, :]
        return {
            "joints":       joints_out,
            "pelvis_depth": self.depth_out(pelvis_token),
            "pelvis_uv":    self.uv_out(pelvis_token),
        }
```

## Loss / Metric Conversion: `train.py`

The training target `batch["joints"]` contains root-relative 3D coordinates in metres, stored as `(x, y, z)`. The pipeline in `SubtractRoot` / `CropPerson` sets:
- `joints[:, :, 0]` = x (left-right in camera frame, metres)
- `joints[:, :, 1]` = y (up-down, metres)
- `joints[:, :, 2]` = z (depth, metres)

The heatmap head produces `(u_norm, v_norm, z)` where `u_norm, v_norm ∈ [0,1]` are normalized image coordinates. To match training targets we must convert `(u, v)` → `(x, y)` using the camera intrinsics and the pelvis absolute depth.

**Conversion formula** (from `infra.py → recover_pelvis_3d`):
- `u_px = u_norm * crop_w`;  `v_px = v_norm * crop_h`
- `x = -(u_px - cx) * X_pelvis / fx`
- `y = -(v_px - cy) * X_pelvis / fy`
where `X_pelvis` is the pelvis absolute depth (from `batch["pelvis_abs"][:, 0]`).

However, applying this per-joint would require knowing each joint's actual depth (not just pelvis), creating a chicken-and-egg problem. **Simpler proxy used:**

The training targets `batch["joints"]` are already in root-relative metres. The heatmap head predicts `(u_norm, v_norm, z_metres)`. We need to align the coordinate space.

**Implementation:** Convert the GT joints into normalized UV space for the loss on XY, and keep Z loss in metres:

In `train.py`, compute `gt_uv_per_joint` from GT joints and intrinsics:
```python
# Convert GT joints (B, 70, 3) root-relative metres → normalized UV
# joints[:,:,0] = x_rel, joints[:,:,1] = y_rel, joints[:,:,2] = z_rel (metres)
# Pelvis absolute depth = gt_pelvis_abs[:, 0] (metres, shape (B,))
X_pel = gt_pelvis_abs[:, 0:1].unsqueeze(1)  # (B, 1, 1), pelvis X (depth axis)
# Approximate joint X as pelvis X (root-relative z is small relative to pelvis depth)
# Full joint world X: X_j ≈ X_pel + joints[:,:,0] — use pelvis X as reference depth
# u_px = fx * (-x_rel) / (X_pel + joints[:,:,2]) + cx  ← exact but circular for z
# SIMPLIFIED: use pelvis X_pel as reference depth for all joints (valid approximation)
X_ref = X_pel + joints[:, :, 2:3]  # (B, 70, 1) per-joint depth in world frame
u_px  = K[:, 0:1, 0:1] * (-joints[:, :, 0:1]) / X_ref + K[:, 0:1, 2:3]  # (B,70,1)
v_px  = K[:, 1:2, 1:2] * (-joints[:, :, 1:2]) / X_ref + K[:, 1:2, 2:3]  # (B,70,1)
u_norm = u_px / args.img_w   # (B, 70, 1) in [0,1]
v_norm = v_px / args.img_h   # (B, 70, 1)
gt_uv_joints = torch.cat([u_norm, v_norm], dim=-1)  # (B, 70, 2)
```

Then compute the loss:
```python
# out["joints"] = (B, 70, 3): [u_norm, v_norm, z_metres]
l_xy = pose_loss(out["joints"][:, BODY_IDX, :2], gt_uv_joints[:, BODY_IDX])
l_z  = pose_loss(out["joints"][:, BODY_IDX, 2:3], joints[:, BODY_IDX, 2:3])
l_pose = l_xy + l_z
```

The pelvis auxiliary losses (`l_dep`, `l_uv`) remain unchanged.

**Metric reporting:** For `mpjpe`, convert head output back to metre-space before computing MPJPE. Add a `decode_joints` helper:
```python
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
```
Use this in `train_one_epoch` for the MPJPE metric (not for loss — loss stays in UV+Z space to keep gradient signal clean).

## Configuration: `config.py`

All values inherited from `runs/idea014/design003/config.py` **unchanged** except:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea016/design001` | updated path |
| `heatmap_h` | `40` | new field — native ViT patch grid height |
| `heatmap_w` | `24` | new field — native ViT patch grid width |
| `lambda_depth` | `0.1` | unchanged |
| `lambda_uv` | `0.2` | unchanged |
| `lambda_z_joint` | `1.0` | weight on per-joint Z loss (relative to XY loss weight=1.0) |

All other fields unchanged:
- `head_hidden=384, head_num_heads=8, head_num_layers=4, head_dropout=0.1`
- `llrd_gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4, lr_depth_pe=1e-4`
- `weight_decay=0.3, warmup_epochs=3, grad_clip=1.0`
- `epochs=20, batch_size=4, accum_steps=8`

## Changes Required

1. **`model.py`**: Replace `Pose3DHead` with version above. Add `heatmap_h, heatmap_w` constructor params. Wire through `SapiensPose3D.__init__`.
2. **`config.py`**: Add `heatmap_h=40, heatmap_w=24, lambda_z_joint=1.0`. Update `output_dir`.
3. **`train.py`**: 
   - Add `gt_uv_joints` computation in `train_one_epoch`.
   - Replace `l_pose = pose_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX])` with `l_xy + l_z` formulation.
   - Add `decode_joints_heatmap` helper for MPJPE metric reporting.
   - Update MPJPE call to use decoded joints: `mpjpe(decode_joints_heatmap(...), joints, BODY_IDX)`.
   - Keep `pelvis_abs_error` call unchanged (still uses `out["pelvis_depth"]` and `out["pelvis_uv"]`).
4. **`transforms.py`**: No changes.

## Expected Behaviour

- Heatmap `(B, 70, 40, 24)` — 960 spatial bins per joint — provides dense spatial supervision at the native Sapiens feature resolution.
- Soft-argmax is sub-pixel accurate and fully differentiable.
- New params: `Linear(384, 960)` = 369K weights (replaces original `Linear(384, 3)` = 1.15K).
- Memory: `(4, 70, 960)` float32 ≈ 1.1 MB — negligible.
