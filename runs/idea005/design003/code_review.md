# Code Review — design003 (depth_conditioned_pe)

**Reviewer:** Designer agent  
**Date:** 2026-04-07  
**Verdict:** APPROVED

---

## Summary

The Builder's implementation faithfully matches the design specification for Design 003. All critical architectural requirements are satisfied. One minor deviation from a Builder implementation note (module placement) is acceptable and does not affect correctness.

---

## Checklist

### 1. `DepthConditionedPE` Module

**PASS.** Defined in `model.py` as a standalone `nn.Module`.

- Architecture: `Linear(3→128)` + GELU → `Linear(128→256)` + GELU → `Linear(256→1024)`. Matches spec exactly.
- Initialization: `nn.init.xavier_uniform_(layer.weight, gain=0.01)` and `nn.init.zeros_(layer.bias)` applied to all three layers. Correct.
- `forward(coords)`: `F.gelu(self.fc1(coords))` → `F.gelu(self.fc2(x))` → `self.fc3(x)`. Correct GELU placement (after fc1 and fc2, not after output layer).

Minor note: The design spec's Builder Implementation Note #6 suggested placing `DepthConditionedPE` in `train.py`. The Builder placed it in `model.py` instead, which is architecturally cleaner and fully correct. No functional impact.

---

### 2. `SapiensBackboneRGBD.forward` — Manual ViT Forward Pass

**PASS.** `vit.forward()` is never called. The manual forward correctly sequences:

1. `self.vit.patch_embed(x)` → `(B, 960, embed_dim)`
2. `pe_base + pe_correction` added to `patch_tokens`
3. `self.vit.drop_after_pos(x_tokens)` (dropout after PE, matching ViT internals)
4. `self.vit.pre_norm(x_tokens)` (identity in standard configs)
5. Loop over `self.vit.layers` with `self.vit.ln1` applied at the final layer when `final_norm=True`
6. Reshape to `(B, embed_dim, H_tok, W_tok)`

A clear comment is present: `# pos_embed added here; vit.forward() is NOT called`.

---

### 3. `pos_embed` Added Exactly Once

**PASS.** The line:

```python
x_tokens = patch_tokens + pe_base + pe_correction   # (B, 960, embed_dim)
```

adds `pos_embed` exactly once. `vit.forward()` is not called, so no double-add is possible.

---

### 4. Depth Coordinate Construction

**PASS.** Implementation matches spec:

- `depth_ch = x[:, 3:4, :, :]` → `(B, 1, 640, 384)`
- `F.avg_pool2d(depth_ch, kernel_size=16, stride=16)` → `(B, 1, 40, 24)` (640/16=40, 384/16=24)
- `depth_flat = depth_patches.reshape(B, -1)` → `(B, 960)`
- `rows = torch.arange(40) / (40 - 1)` → normalized to [0, 1]
- `cols = torch.arange(24) / (24 - 1)` → normalized to [0, 1]
- `torch.meshgrid(rows, cols, indexing='ij')` → correct row-major ordering
- `torch.stack([row_batch, col_batch, depth_flat], dim=-1)` → `(B, 960, 3)`

All shapes and normalization are correct.

---

### 5. Optimizer — Three Parameter Groups

**PASS.** Three groups are defined in `train.py`:

```python
backbone_params = [p for name, p in model.backbone.named_parameters()
                   if 'depth_cond_pe' not in name]
depth_pe_params = [p for name, p in model.backbone.named_parameters()
                   if 'depth_cond_pe' in name]
optimizer = torch.optim.AdamW(
    [{"params": backbone_params,  "lr": args.lr_backbone},
     {"params": depth_pe_params,  "lr": args.lr_depth_pe},
     {"params": model.head.parameters(), "lr": args.lr_head}],
    weight_decay=args.weight_decay,
)
```

- Backbone ViT (including `vit.pos_embed`) at `lr_backbone = 1e-5`. Correct.
- `depth_cond_pe` MLP at `lr_depth_pe = 1e-4`. Correct.
- Head at `lr_head = 1e-4`. Correct.
- `weight_decay = 0.03` applied uniformly via AdamW. Correct.
- Filter uses `'depth_cond_pe' in name` as specified.

---

### 6. Output Metrics Headers

**PASS.** Both CSV files verified against actual test output:

**metrics.csv** (from `_CSV_FIELDNAMES` in `infra.py`):
```
epoch, lr_backbone, lr_head, train_loss, train_loss_pose,
train_mpjpe_body, train_pelvis_err, train_mpjpe_weighted,
val_loss, val_loss_pose, val_mpjpe_body, val_pelvis_err,
val_mpjpe_weighted, epoch_time
```

**iter_metrics.csv** (from `_ITER_CSV_FIELDNAMES`):
```
epoch, iter, loss, loss_pose, loss_depth, loss_uv,
mpjpe_body, pelvis_err, mpjpe_weighted
```

Headers match what `train_one_epoch` logs. Actual test output confirmed both files are written and populated correctly.

---

### 7. Seed = 2026

**PASS.** `RANDOM_SEED = 2026` is defined in `/work/pi_nwycoff_umass_edu/hang/auto/infra.py` (line 139). `config.py` imports and uses it as `seed = RANDOM_SEED`, which is passed to `get_splits()` for reproducible train/val/test split generation. This matches baseline behavior across all designs.

---

## Additional Observations

- `config.py` correctly specifies all design-specific hyperparameters: `lr_backbone=1e-5`, `lr_depth_pe=1e-4`, `lr_head=1e-4`, `weight_decay=0.03`, `warmup_epochs=3`, `lambda_depth=0.1`, `lambda_uv=0.2`, `epochs=20`, `arch="sapiens_0.3b"`.
- The `SapiensBackboneRGBD` class header accurately documents Design 003's intent.
- Test output (`test_output/metrics.csv`, `test_output/iter_metrics.csv`) shows the model runs and produces sensible loss/metric values for 2 epochs, confirming no runtime errors.

---

## Verdict

**APPROVED**

The implementation is a faithful, correct, and complete realization of the design003 specification. No issues require fixing before submission.
