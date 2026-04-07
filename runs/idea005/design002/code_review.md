# Code Review — Design 002 (relative_depth_bias)

**Reviewer:** Designer agent  
**Date:** 2026-04-07  
**Verdict:** APPROVED

---

## Summary

The Builder's implementation in `runs/idea005/design002/code/` faithfully matches the design specification in `design.md`. All seven checklist items pass.

---

## Checklist

### 1. `DepthAttentionBias` module with `Linear(1, 70)`, zero-initialized

**PASS.**  
`model.py` lines 53–90 define `DepthAttentionBias` as a standalone `nn.Module`.  
- `self.depth_proj = nn.Linear(1, num_joints)` with `num_joints=NUM_JOINTS` (70).  
- Both weight and bias are explicitly zero-initialized:  
  ```python
  nn.init.zeros_(self.depth_proj.weight)
  nn.init.zeros_(self.depth_proj.bias)
  ```
- The module is instantiated inside `Pose3DHead.__init__` as `self.depth_attn_bias = DepthAttentionBias(num_joints=num_joints)`, placing it naturally in the head parameter group.

---

### 2. Manual decoder loop intercepts cross-attention and adds depth bias

**PASS.**  
`model.py` lines 164–185 replace the single `self.decoder(queries, memory)` call with a manual loop over `self.decoder.layers`. Each layer correctly applies:
1. Self-attention sublayer via `layer.norm1`, `layer.self_attn`, `layer.dropout1` (norm_first=True order).
2. Cross-attention sublayer via `layer.norm2`, `layer.multihead_attn(..., attn_mask=depth_bias_expanded)`, `layer.dropout2`.
3. FFN sublayer via `layer.norm3`, `layer.linear1/2`, `layer.activation`, `layer.dropout/dropout3`.

The post-decoder norm `self.decoder.norm` is applied if present. The attribute names match PyTorch's `nn.TransformerDecoderLayer` with `norm_first=True`.

---

### 3. Depth bias shape: `(B*num_heads, 70, 960)` passed as `attn_mask`

**PASS.**  
`model.py` lines 157–159:
```python
depth_bias_expanded = depth_bias.unsqueeze(1).expand(
    B, self.num_heads, -1, -1
).reshape(B * self.num_heads, self.num_joints, N_mem)
```
With `B=batch_size`, `num_heads=8`, `num_joints=70`, `N_mem=960`, this produces shape `(B*8, 70, 960)` as required.  
The bias is computed once before the loop (not redundantly per layer), which is consistent with the design intent.

---

### 4. `depth_ch` extracted and passed from `SapiensPose3D.forward` → `Pose3DHead.forward`

**PASS.**  
`model.py` lines 275–279:
```python
def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    depth_ch = x[:, 3:4, :, :]  # (B, 1, H, W)
    feat = self.backbone(x)     # (B, C, H_tok, W_tok)
    return self.head(feat, depth_ch)
```
`Pose3DHead.forward` signature correctly accepts `depth_ch` as an additional argument.

---

### 5. Optimizer groups match the design (backbone LR=1e-5, head LR=1e-4)

**PASS.**  
`train.py` lines 201–205:
```python
optimizer = torch.optim.AdamW(
    [{"params": model.backbone.parameters(), "lr": args.lr_backbone},
     {"params": model.head.parameters(),     "lr": args.lr_head}],
    weight_decay=args.weight_decay,
)
```
`config.py` sets `lr_backbone = 1e-5`, `lr_head = 1e-4`, `weight_decay = 0.03`.  
Since `DepthAttentionBias` is owned by `Pose3DHead`, it naturally falls into the head LR group via `model.head.parameters()`.

---

### 6. Output metrics have correct headers

**PASS.**  
`test_output/metrics.csv` header (line 1):
```
epoch,lr_backbone,lr_head,train_loss,train_loss_pose,train_mpjpe_body,train_pelvis_err,train_mpjpe_weighted,val_loss,val_loss_pose,val_mpjpe_body,val_pelvis_err,val_mpjpe_weighted,epoch_time
```
`test_output/iter_metrics.csv` header (line 1):
```
epoch,iter,loss,loss_pose,loss_depth,loss_uv,mpjpe_body,pelvis_err,mpjpe_weighted
```
Both files contain real data (2 epochs completed), confirming the training loop ran successfully.

---

### 7. Seed set to 2026

**PASS.**  
`train.py` lines 136–141:
```python
random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
```

---

## Additional Observations

- **`norm_first=True` handling:** The design spec showed a generic loop template, but the Builder correctly adapted it for `norm_first=True` (the norm is applied before the sublayer, not after). This is the correct behavior for the baseline's `TransformerDecoderLayer` configuration.
- **Depth bias computed once before the loop:** The design spec placed the bias computation inside the per-layer loop, but the implementation correctly hoists it outside (since the depth input does not change between layers). This is a minor, beneficial deviation — the mathematical behavior is identical, and the computation is more efficient.
- **`DepthAttentionBias.forward` signature:** The spec said `forward(depth_ch, B)` but the implementation uses just `forward(depth_ch)` with `B = depth_ch.size(0)` inferred internally. This is cleaner and functionally equivalent.
- The test run in `test_output/` shows the model trained for 2 epochs without errors, confirming the implementation is functional.

---

## Verdict

**APPROVED.** The implementation faithfully and correctly matches the design specification. All required components are present and correctly implemented. The minor deviations noted above are improvements over the spec, not regressions.
