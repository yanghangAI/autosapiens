# design004 — 2D Heatmap + Scalar Depth + Auxiliary Gaussian MSE Supervision

**Starting point:** `runs/idea014/design003/code/`

## Summary

Same as design001 (2D heatmap + scalar depth at native 40×24 resolution) with one addition: an **auxiliary MSE loss on the predicted heatmap against a Gaussian target** centred at the ground-truth `(u, v)` for each joint. Weight `lambda_hm = 0.1`. This provides dense spatial supervision during early epochs — a known technique for stabilising soft-argmax training when the coordinate loss alone may not generate enough gradient to shape the heatmap.

## Problem

Soft-argmax training with only coordinate loss can be slow to converge: initially the heatmap is nearly uniform and the soft-argmax gradient is very weak (because a diffuse distribution has near-zero derivative w.r.t. the peak). Adding a Gaussian MSE target directly shapes the spatial heatmap and accelerates early convergence, analogous to heatmap pre-training in bottom-up pose methods.

## Architecture Change: `model.py`

Identical to design001 **except** the head also returns the raw heatmap logits (after softmax) so the train loop can compute the Gaussian MSE loss.

```python
class Pose3DHead(nn.Module):
    def __init__(self, in_channels, num_joints=NUM_JOINTS, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1,
                 heatmap_h=40, heatmap_w=24):
        super().__init__()
        self.num_joints = num_joints
        self.hm_h = heatmap_h
        self.hm_w = heatmap_w

        self.input_proj    = nn.Linear(in_channels, hidden_dim)
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.heatmap_out     = nn.Linear(hidden_dim, heatmap_h * heatmap_w)
        self.depth_joint_out = nn.Linear(hidden_dim, 1)
        self.depth_out       = nn.Linear(hidden_dim, 1)
        self.uv_out          = nn.Linear(hidden_dim, 2)

        # Coordinate buffers for soft-argmax
        coords_u = torch.linspace(0.0, 1.0, heatmap_w)
        coords_v = torch.linspace(0.0, 1.0, heatmap_h)
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
        out = self.decoder(queries, memory)   # (B, 70, hidden)

        hm_logits = self.heatmap_out(out)               # (B, 70, H*W)
        hm_soft   = torch.softmax(hm_logits, dim=-1)    # (B, 70, H*W)
        pred_u    = (hm_soft * self.grid_u).sum(dim=-1)
        pred_v    = (hm_soft * self.grid_v).sum(dim=-1)
        pred_z    = self.depth_joint_out(out).squeeze(-1)

        joints_out = torch.stack([pred_u, pred_v, pred_z], dim=-1)  # (B, 70, 3)

        pelvis_token = out[:, 0, :]
        return {
            "joints":       joints_out,
            "heatmap_soft": hm_soft,          # (B, 70, H*W) — for Gaussian MSE loss
            "pelvis_depth": self.depth_out(pelvis_token),
            "pelvis_uv":    self.uv_out(pelvis_token),
        }
```

## Gaussian Target Generation: `train.py`

Add a helper function to generate Gaussian heatmap targets:

```python
def make_gaussian_targets(gt_uv_joints, hm_h, hm_w, sigma=2.0, device="cuda"):
    """
    Build Gaussian heatmap targets for each joint.
    
    Args:
        gt_uv_joints: (B, 70, 2) — GT normalized UV in [0,1]
        hm_h, hm_w:   int — heatmap spatial dims (40, 24)
        sigma:        float — Gaussian std in grid cells
        device:       torch device
    Returns:
        targets: (B, 70, H*W) normalized Gaussian maps (sum=1)
    """
    B, J, _ = gt_uv_joints.shape
    # GT UV in grid coordinates
    mu_u = gt_uv_joints[:, :, 0] * (hm_w - 1)  # (B, 70)
    mu_v = gt_uv_joints[:, :, 1] * (hm_h - 1)

    # Grid coordinates
    u_idx = torch.arange(hm_w, dtype=torch.float32, device=device)
    v_idx = torch.arange(hm_h, dtype=torch.float32, device=device)
    grid_v, grid_u = torch.meshgrid(v_idx, u_idx, indexing="ij")  # (H, W)
    grid_u = grid_u.reshape(1, 1, -1)  # (1, 1, H*W)
    grid_v = grid_v.reshape(1, 1, -1)

    # Gaussian: (B, 70, H*W)
    mu_u = mu_u.unsqueeze(-1)  # (B, 70, 1)
    mu_v = mu_v.unsqueeze(-1)
    gauss = torch.exp(-((grid_u - mu_u)**2 + (grid_v - mu_v)**2) / (2.0 * sigma**2))

    # Normalize to sum=1 (matching softmax output)
    gauss = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)
    return gauss
```

## Loss Computation: `train.py`

```python
# UV targets (same as design001)
X_ref  = gt_pelvis_abs[:, 0:1].unsqueeze(1) + joints[:, :, 2:3]
u_px   = K[:, 0:1, 0:1] * (-joints[:, :, 0:1]) / X_ref + K[:, 0:1, 2:3]
v_px   = K[:, 1:2, 1:2] * (-joints[:, :, 1:2]) / X_ref + K[:, 1:2, 2:3]
gt_uv_joints = torch.cat([u_px / args.img_w, v_px / args.img_h], dim=-1)  # (B, 70, 2)

# Coordinate losses
l_xy   = pose_loss(out["joints"][:, BODY_IDX, :2], gt_uv_joints[:, BODY_IDX])
l_z    = pose_loss(out["joints"][:, BODY_IDX, 2:3], joints[:, BODY_IDX, 2:3])
l_pose = l_xy + l_z

# Auxiliary Gaussian MSE loss (body joints only)
gauss_targets = make_gaussian_targets(
    gt_uv_joints, args.heatmap_h, args.heatmap_w,
    sigma=args.hm_sigma, device=device
)  # (B, 70, H*W)
pred_hm = out["heatmap_soft"][:, BODY_IDX]          # (B, 22, H*W)
gt_hm   = gauss_targets[:, BODY_IDX]                # (B, 22, H*W)
l_hm    = F.mse_loss(pred_hm, gt_hm)

# Total loss
loss = (l_pose + args.lambda_depth * l_dep + args.lambda_uv * l_uv
        + args.lambda_hm * l_hm) / args.accum_steps
```

Also log `l_hm` in `iter_logger`.

## Configuration: `config.py`

All values inherited from `runs/idea014/design003/config.py` unchanged except:

| Field | Value | Note |
|---|---|---|
| `output_dir` | `runs/idea016/design004` | updated path |
| `heatmap_h` | `40` | native grid height |
| `heatmap_w` | `24` | native grid width |
| `lambda_hm` | `0.1` | new — weight on Gaussian heatmap MSE loss |
| `hm_sigma` | `2.0` | new — Gaussian sigma in grid cells |
| `lambda_z_joint` | `1.0` | per-joint Z loss weight |

All other fields unchanged.

## Changes Required

1. **`model.py`**: Replace `Pose3DHead` with version above (adds `"heatmap_soft"` key to return dict). Add `heatmap_h, heatmap_w` constructor params. Wire through `SapiensPose3D.__init__`.
2. **`config.py`**: Add `heatmap_h=40, heatmap_w=24, lambda_hm=0.1, hm_sigma=2.0, lambda_z_joint=1.0`. Update `output_dir`.
3. **`train.py`**:
   - Add `make_gaussian_targets` helper function.
   - Add `gt_uv_joints` computation.
   - Replace `l_pose` with `l_xy + l_z`.
   - Add Gaussian MSE loss `l_hm`.
   - Update total loss to include `args.lambda_hm * l_hm`.
   - Add `decode_joints_heatmap` helper for MPJPE reporting (same as design001).
   - Log `l_hm` in iter_logger dict.
4. **`transforms.py`**: No changes.

## Memory Analysis

- Heatmap soft: `(4, 70, 40*24) = (4, 70, 960)` = 1.07M floats = 4.3 MB.
- Gaussian targets: same size. Generated on-the-fly in GPU memory.
- MSE on `(B, 22, 960)`: trivial computation.

## Expected Behaviour

The Gaussian MSE loss provides direct spatial gradient into the heatmap logits, bypassing the soft-argmax expectation gradient. This is known to help in early training when the heatmap is diffuse. After several epochs the coordinate loss dominates and the Gaussian loss acts as a light regulariser (`lambda_hm=0.1`). The `sigma=2.0` grid-cell Gaussian is tight enough to be informative (covers ~5% of the 40×24 grid) but not so sharp that it penalises sub-bin precision.
