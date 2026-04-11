# design002 — 2D Heatmap + Scalar Depth (80×48 upsampled grid)

**Starting point:** `runs/idea014/design003/code/`

## Summary

Same two-branch heatmap architecture as design001, but the 40×24 spatial heatmap is **bilinearly upsampled to 80×48** before softmax + soft-argmax. This tests whether the coarser native grid (40×24) limits sub-pixel localization accuracy, at the cost of a 4× larger softmax (3840 bins vs 960).

- **2D heatmap branch**: `Linear(384, 40*24=960)` → reshape `(B, 70, 40, 24)` → upsample to `(B, 70, 80, 48)` → softmax over H×W → soft-argmax → `(u, v)` normalized.
- **Depth branch**: `Linear(384, 1)` per joint, root-relative metres.
- Pelvis auxiliary heads unchanged.

## Problem

At 40×24, each bin covers 16×16 pixels. While the network can interpolate between bins during soft-argmax, finer resolution coordinate grids yield higher precision. Upsampling the logit map before softmax effectively increases the density of the probability mass while keeping the backbone and decoder frozen.

## Architecture Change: `model.py`

```python
class Pose3DHead(nn.Module):
    def __init__(self, in_channels, num_joints=NUM_JOINTS, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1,
                 heatmap_h=40, heatmap_w=24,
                 upsample_factor=2):
        super().__init__()
        self.num_joints = num_joints
        self.hm_h = heatmap_h
        self.hm_w = heatmap_w
        self.up_h = heatmap_h * upsample_factor   # 80
        self.up_w = heatmap_w * upsample_factor   # 48

        self.input_proj    = nn.Linear(in_channels, hidden_dim)
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heatmap at native 40×24 resolution
        self.heatmap_out = nn.Linear(hidden_dim, heatmap_h * heatmap_w)

        # Per-joint depth (root-relative, metres)
        self.depth_joint_out = nn.Linear(hidden_dim, 1)

        # Pelvis auxiliary (unchanged)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out    = nn.Linear(hidden_dim, 2)

        # Coordinate buffers for soft-argmax at upsampled resolution
        coords_u = torch.linspace(0.0, 1.0, self.up_w)
        coords_v = torch.linspace(0.0, 1.0, self.up_h)
        grid_v, grid_u = torch.meshgrid(coords_v, coords_u, indexing="ij")
        self.register_buffer("grid_u", grid_u.reshape(-1))
        self.register_buffer("grid_v", grid_v.reshape(-1))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.depth_joint_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.heatmap_out.weight, std=0.02)
        nn.init.zeros_(self.heatmap_out.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        B = feat.size(0)
        memory  = self.input_proj(feat.flatten(2).transpose(1, 2))
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)  # (B, 70, hidden)

        # Predict at native 40×24
        hm_native = self.heatmap_out(out)                       # (B, 70, 40*24)
        hm_2d     = hm_native.view(B, self.num_joints, self.hm_h, self.hm_w)  # (B, 70, 40, 24)

        # Upsample to 80×48 (bilinear, in logit space — before softmax)
        hm_up = F.interpolate(
            hm_2d.view(B * self.num_joints, 1, self.hm_h, self.hm_w),
            size=(self.up_h, self.up_w),
            mode="bilinear", align_corners=False,
        ).view(B, self.num_joints, -1)  # (B, 70, 80*48=3840)

        # Softmax + soft-argmax
        hm_soft = torch.softmax(hm_up, dim=-1)
        pred_u  = (hm_soft * self.grid_u).sum(dim=-1)  # (B, 70)
        pred_v  = (hm_soft * self.grid_v).sum(dim=-1)

        pred_z = self.depth_joint_out(out).squeeze(-1)  # (B, 70)
        joints_out = torch.stack([pred_u, pred_v, pred_z], dim=-1)  # (B, 70, 3)

        pelvis_token = out[:, 0, :]
        return {
            "joints":       joints_out,
            "pelvis_depth": self.depth_out(pelvis_token),
            "pelvis_uv":    self.uv_out(pelvis_token),
        }
```

## Loss / Metric Conversion: `train.py`

Identical to design001. The GT UV targets and decode_joints_heatmap helper are computed the same way (the softmax/grid resolution difference is internal to the head):

```python
# GT UV per joint (same as design001)
X_ref  = gt_pelvis_abs[:, 0:1].unsqueeze(1) + joints[:, :, 2:3]  # (B, 70, 1)
u_px   = K[:, 0:1, 0:1] * (-joints[:, :, 0:1]) / X_ref + K[:, 0:1, 2:3]
v_px   = K[:, 1:2, 1:2] * (-joints[:, :, 1:2]) / X_ref + K[:, 1:2, 2:3]
gt_uv_joints = torch.cat([u_px / args.img_w, v_px / args.img_h], dim=-1)  # (B, 70, 2)

# Loss
l_xy   = pose_loss(out["joints"][:, BODY_IDX, :2], gt_uv_joints[:, BODY_IDX])
l_z    = pose_loss(out["joints"][:, BODY_IDX, 2:3], joints[:, BODY_IDX, 2:3])
l_pose = l_xy + l_z
```

`decode_joints_heatmap` helper same as design001 for MPJPE reporting.

## Configuration: `config.py`

All values inherited from `runs/idea014/design003/config.py` unchanged except:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea016/design002` | updated path |
| `heatmap_h` | `40` | native grid height |
| `heatmap_w` | `24` | native grid width |
| `upsample_factor` | `2` | new field — 2× bilinear upsample before softmax |
| `lambda_z_joint` | `1.0` | weight on per-joint Z loss |

All other fields unchanged (identical to design001 which is identical to idea014/design003).

## Changes Required

1. **`model.py`**: Replace `Pose3DHead` with version above. Add `heatmap_h, heatmap_w, upsample_factor` constructor params. Wire through `SapiensPose3D.__init__`.
2. **`config.py`**: Add `heatmap_h=40, heatmap_w=24, upsample_factor=2, lambda_z_joint=1.0`. Update `output_dir`.
3. **`train.py`**: Same changes as design001 (GT UV targets, loss formulation, decode helper, MPJPE call).
4. **`transforms.py`**: No changes.

## Memory Analysis

- Native heatmap: `(4, 70, 40, 24)` = 268K floats = 1.0 MB.
- Upsampled heatmap: `(4, 70, 80, 48)` = 1.07M floats = 4.3 MB. Negligible.
- `F.interpolate` on `(280, 1, 40, 24) → (280, 1, 80, 48)` is cheap (bilinear on CPU-like size).

## Expected Behaviour

Higher-resolution coordinate grid (80×48 = 3840 bins vs 960) should improve sub-pixel soft-argmax accuracy. The bilinear interpolation in logit space is equivalent to inserting smoothly interpolated candidate positions, which the softmax can resolve more finely. Cost: 4× the softmax computation — still trivial vs the decoder cross-attention.
