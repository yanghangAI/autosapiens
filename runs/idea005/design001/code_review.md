# Code Review — idea005/design001 (discretized_depth_pe)

**Verdict: APPROVED**

---

## Summary

The implementation faithfully follows the `design.md` specification. All architectural changes, hyperparameter values, optimizer groups, and initialization logic are correctly implemented.

---

## Checklist

### Config (`config.py`)
- `arch = "sapiens_0.3b"` ✓
- `epochs = 20` ✓
- `lr_backbone = 1e-5`, `lr_head = 1e-4`, `lr_depth_pe = 1e-4` ✓
- `weight_decay = 0.03`, `warmup_epochs = 3` ✓
- `lambda_depth = 0.1`, `lambda_uv = 0.2` ✓
- `num_depth_bins = 16` ✓

### `DepthBucketPE` module (`model.py`)
- Defined as a standalone `nn.Module` (placed in `model.py` rather than `train.py` as the spec suggested, which is a reasonable and better choice) ✓
- `row_emb (40, 1024)`, `col_emb (24, 1024)`, `depth_emb (16, 1024)` all `nn.Parameter` ✓
- Correct depth discretization: `avg_pool2d(kernel_size=16, stride=16)` → `* num_depth_bins` → `.long().clamp(0, num_depth_bins - 1)` ✓
- Composed PE: `row_emb[rows] + col_emb[cols] + depth_emb[depth_bins_flat]` with correct broadcasting ✓

### `vit.pos_embed` zeroing
- Done in `SapiensPose3D.__init__` via `del vit.pos_embed` + `vit.register_buffer("pos_embed", torch.zeros(...))` ✓
- The ViT's internal forward still adds `pos_embed`, but since the code manually bypasses ViT's `forward()` via `_run_vit_manual`, the zeroed buffer is effectively never applied ✓

### Manual ViT forward (`_run_vit_manual`)
- Correctly replicates ViT forward: `patch_embed` → custom PE injection → `drop_after_pos` → `pre_norm` → transformer layers → final norm → `_format_output` ✓
- Verified against `VisionTransformer.forward` source — the sequence matches ✓
- `out_indices` handling: collects output at the correct layer index ✓

### Weight initialization
- Pretrained `pos_embed` bicubic-interpolated from source grid to `(40, 24)` ✓
- `row_emb` initialized from row-mean (mean over cols), `col_emb` from col-mean (mean over rows) ✓
- `depth_emb` stays zero-initialized ✓
- `vit.pos_embed` key excluded from `load_state_dict` call ✓

### Optimizer groups (`train.py`)
- Three groups correctly defined: backbone params (excluding depth_pe), `depth_bucket_pe` params, head params ✓
- `depth_pe_ids` exclusion correctly uses `id()` comparison ✓
- LR scaling applied to all three groups via `initial_lr` pattern ✓

---

## Minor Notes (non-blocking)

1. `resize_pos_embed` is imported from `mmpretrain.models.utils` in `model.py` but is unused. This is harmless dead import and does not affect correctness.
2. `src_g` (the source grid size) is computed in `load_sapiens_pretrained` solely for the debug print message; it is never used for logic. This is fine.

---

## Conclusion

The implementation is complete, correct, and ready to run. All design-specified components are present and correctly implemented. No logical errors, missing features, or deviations from the spec were found.
