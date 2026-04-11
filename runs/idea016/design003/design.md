# design003 — Full 3D Volumetric Heatmap (40×24×16, Integral Pose Regression)

**Starting point:** `runs/idea014/design003/code/`

## Summary

Replace the final output with a **full 3D volumetric heatmap**: `Linear(384, 40*24*16=15360)` → reshape `(B, 70, 40, 24, 16)` → softmax over the full 3D volume → soft-argmax on all three axes simultaneously. The 16-bin depth axis reuses the **same sqrt-spaced depth bin centres** already defined in `DepthBucketPE` (coherent with `num_depth_bins=16`). This is the canonical Integral Pose Regression (Sun et al. 2018) applied to the 3D case.

Output: single unified `(x, y, z)` in metric coordinates from one 3D soft-argmax — no separate depth head.

## Problem

Design001/002 keep depth as a scalar regression. A volumetric heatmap extends the spatial inductive bias to all three axes: the network can express multi-modal uncertainty in depth and resolve it via the soft-argmax probability mass. The 16-bin depth axis matches the existing depth PE geometry — same anchors, same scale, entirely coherent.

## Architecture Change: `model.py`

### Depth bin centres

The `DepthBucketPE` uses sqrt-spaced bins `d_k = (k / (num_depth_bins-1))^2 * D_max` where `D_max = DEPTH_MAX_METERS`. We register these as a buffer in the head.

```python
from infra import DEPTH_MAX_METERS  # e.g. 10.0 metres

class Pose3DHead(nn.Module):
    def __init__(self, in_channels, num_joints=NUM_JOINTS, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1,
                 heatmap_h=40, heatmap_w=24, num_depth_bins=16):
        super().__init__()
        self.num_joints    = num_joints
        self.hm_h          = heatmap_h
        self.hm_w          = heatmap_w
        self.num_depth_bins = num_depth_bins

        self.input_proj    = nn.Linear(in_channels, hidden_dim)
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3D volumetric output: H * W * D bins
        self.heatmap_3d_out = nn.Linear(hidden_dim, heatmap_h * heatmap_w * num_depth_bins)

        # Pelvis auxiliary (unchanged from baseline)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out    = nn.Linear(hidden_dim, 2)

        # Precomputed coordinate buffers for soft-argmax
        # UV: normalized [0,1]
        coords_u = torch.linspace(0.0, 1.0, heatmap_w)
        coords_v = torch.linspace(0.0, 1.0, heatmap_h)
        grid_v, grid_u = torch.meshgrid(coords_v, coords_u, indexing="ij")
        # Depth bins: sqrt-spaced (matching DepthBucketPE)
        # d_k = (k / (D-1))^2 * DEPTH_MAX  (k=0..D-1)
        k_vals = torch.arange(num_depth_bins, dtype=torch.float32) / (num_depth_bins - 1)
        depth_bin_centres = (k_vals ** 2) * DEPTH_MAX_METERS  # (D,), metres absolute

        # Register coordinate buffers: (H, W, D)
        # For soft-argmax we need coord grids over the 3D volume
        # grid_u: (H, W, D), grid_v: (H, W, D), grid_d: (H, W, D)
        u_3d = grid_u.unsqueeze(-1).expand(heatmap_h, heatmap_w, num_depth_bins)
        v_3d = grid_v.unsqueeze(-1).expand(heatmap_h, heatmap_w, num_depth_bins)
        d_3d = depth_bin_centres.unsqueeze(0).unsqueeze(0).expand(heatmap_h, heatmap_w, num_depth_bins)

        self.register_buffer("grid_u_3d", u_3d.reshape(-1))     # (H*W*D,)
        self.register_buffer("grid_v_3d", v_3d.reshape(-1))
        self.register_buffer("grid_d_3d", d_3d.reshape(-1))     # absolute depth (metres)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        # Zero-init 3D heatmap bias
        nn.init.trunc_normal_(self.heatmap_3d_out.weight, std=0.02)
        nn.init.zeros_(self.heatmap_3d_out.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        B = feat.size(0)
        memory  = self.input_proj(feat.flatten(2).transpose(1, 2))
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)   # (B, 70, hidden)

        # 3D volumetric heatmap
        vol_logits = self.heatmap_3d_out(out)       # (B, 70, H*W*D)
        vol_soft   = torch.softmax(vol_logits, dim=-1)  # (B, 70, H*W*D)

        # Soft-argmax over 3D volume → (u, v, d_abs) per joint
        pred_u     = (vol_soft * self.grid_u_3d).sum(dim=-1)   # (B, 70) in [0,1]
        pred_v     = (vol_soft * self.grid_v_3d).sum(dim=-1)   # (B, 70) in [0,1]
        pred_d_abs = (vol_soft * self.grid_d_3d).sum(dim=-1)   # (B, 70) absolute depth (metres)

        # joints_out stores [u_norm, v_norm, d_abs] — converted to (x,y,z) in loss
        joints_out = torch.stack([pred_u, pred_v, pred_d_abs], dim=-1)  # (B, 70, 3)

        pelvis_token = out[:, 0, :]
        return {
            "joints":       joints_out,
            "pelvis_depth": self.depth_out(pelvis_token),
            "pelvis_uv":    self.uv_out(pelvis_token),
        }
```

## Loss / Metric Conversion: `train.py`

The 3D head outputs `[u_norm, v_norm, d_abs]` where `d_abs` is the **absolute depth** in metres (not root-relative). The GT joint positions are in root-relative metres.

**Conversion for loss:**

```python
# GT: joints (B, 70, 3) root-relative metres; gt_pelvis_abs (B, 3)
# pelvis absolute depth = gt_pelvis_abs[:, 0]  (X-axis in camera coord = depth)

# For each joint, compute GT absolute depth in metres:
X_pel = gt_pelvis_abs[:, 0:1].unsqueeze(1)            # (B, 1, 1)
gt_d_abs = X_pel + joints[:, :, 2:3]                  # (B, 70, 1) world depth for each joint

# GT (u,v) normalized: same projection as design001
X_ref  = gt_d_abs                                      # (B, 70, 1)
u_px   = K[:, 0:1, 0:1] * (-joints[:, :, 0:1]) / X_ref + K[:, 0:1, 2:3]
v_px   = K[:, 1:2, 1:2] * (-joints[:, :, 1:2]) / X_ref + K[:, 1:2, 2:3]
gt_u   = u_px / args.img_w   # (B, 70, 1)
gt_v   = v_px / args.img_h   # (B, 70, 1)
gt_uvd = torch.cat([gt_u, gt_v, gt_d_abs], dim=-1)    # (B, 70, 3)

# Loss: Smooth L1 on [u_norm, v_norm, d_abs] space
# Scale d_abs to [0,1] range to balance with UV (both [0,1])
d_scale = DEPTH_MAX_METERS  # normalisation constant (e.g. 10.0)
pred_uvd_scaled = out["joints"].clone()
pred_uvd_scaled[:, :, 2] /= d_scale
gt_uvd_scaled   = gt_uvd.clone()
gt_uvd_scaled[:, :, 2]   /= d_scale
l_pose = pose_loss(pred_uvd_scaled[:, BODY_IDX], gt_uvd_scaled[:, BODY_IDX])
```

**MPJPE metric:** decode `(u, v, d_abs)` → root-relative `(x, y, z)`:
```python
def decode_joints_3d(pred_uvd, K, img_h, img_w, pelvis_abs):
    """Convert [u_norm, v_norm, d_abs_metres] → root-relative (x,y,z) metres."""
    u_px   = pred_uvd[:, :, 0] * img_w
    v_px   = pred_uvd[:, :, 1] * img_h
    d_abs  = pred_uvd[:, :, 2]           # world depth in metres
    cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
    fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]
    x_abs  = -(u_px - cx) * d_abs / fx
    y_abs  = -(v_px - cy) * d_abs / fy
    # Root-relative = subtract pelvis world position
    x_pel  = -(pelvis_abs[:, 1:2])   # note: pelvis_abs = (X, Y, Z) world → x_rel=y_world?
    # Use pelvis_abs directly:
    # pelvis world = pelvis_abs; joints world = (x_abs, y_abs, d_abs)
    # root-relative = joint - pelvis
    joints_world = torch.stack([x_abs, y_abs, d_abs], dim=-1)   # (B,70,3)
    pelvis_world = pelvis_abs.unsqueeze(1)                        # (B,1,3)
    return joints_world - pelvis_world
```

## Configuration: `config.py`

All values inherited from `runs/idea014/design003/config.py` unchanged except:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea016/design003` | updated path |
| `heatmap_h` | `40` | grid height (matches ViT patch grid) |
| `heatmap_w` | `24` | grid width |
| `num_depth_bins` | `16` | depth bins (same as depth PE, unchanged) |
| `d_scale` | `10.0` | DEPTH_MAX_METERS normalization for loss |

All other fields unchanged.

## Changes Required

1. **`model.py`**: Replace `Pose3DHead` with 3D volumetric version. Import `DEPTH_MAX_METERS` from `infra`. Add `heatmap_h, heatmap_w, num_depth_bins` to constructor (already exists, just wire). Remove `joints_out`, `depth_joint_out`; add `heatmap_3d_out`. Wire through `SapiensPose3D.__init__`.
2. **`config.py`**: Add `heatmap_h=40, heatmap_w=24, d_scale=10.0`. Update `output_dir`. `num_depth_bins=16` already exists.
3. **`train.py`**: Add `gt_uvd` computation (UV + absolute depth GT). Add loss normalisation by `d_scale`. Add `decode_joints_3d` helper for MPJPE. Update MPJPE call.
4. **`transforms.py`**: No changes.

## Memory Analysis

- `Linear(384, 15360)` weight: 5.9M params = 23.6 MB.
- Per-batch 3D volume: `(4, 70, 40, 24, 16) = 4.3M floats = 17.2 MB`. Within budget.
- Softmax over 15360 elements per joint: fast on GPU.

## Expected Behaviour

Full 3D volumetric soft-argmax treats x, y, z prediction symmetrically through the probability space. The depth axis uses the same sqrt-spaced geometry as the depth PE, which has already validated that this bin structure matches the depth distribution of the training data (pelvis at ~1-3m, which corresponds to bins 10-14 in sqrt-space). This design is the most principled but has the highest parameter count increase.
