# Review: idea019/design002 — Kinematic-Chain Soft Self-Attention Bias in Refinement Pass (Axis A2)

**Design_ID:** idea019/design002
**Verdict:** APPROVED

## Evaluation

### Completeness
All baseline hyperparameters are explicitly listed. New config fields `kin_bias_max_hops=3` and `kin_bias_scale_init=0.0` are specified. The `_build_kin_bias` BFS algorithm is fully defined with exact hop weights (1.0, 0.5, 0.25). No guessing required by the Builder.

### Mathematical Correctness
- BFS from each body joint (0-21) over the body subgraph of `SMPLX_SKELETON` correctly computes hop distances.
- Bias values: hop-1 → +1.0, hop-2 → +0.5, hop-3 → +0.25, beyond 3 hops or non-body → 0.0. These are additive positive boosts only; no row is set to all-negative-finite, and -inf is never introduced. The critical invariant holds.
- `kin_bias_scale = nn.Parameter(torch.zeros(1))` initializes at 0.0, so at training step 0 the bias is exactly zero and the model is identical to baseline. ✓
- The manual layer loop `for layer in self.decoder.layers: out2 = layer(out2, memory, tgt_mask=bias)` replicates the baseline `self.decoder(queries2, memory)` call when `bias=0`, and applies the additive self-attention mask in pass 2 only. ✓

### PyTorch API Correctness
`nn.TransformerDecoderLayer.forward` accepts `tgt_mask` as a float additive mask (shape broadcastable to `(B*heads, S, S)` or `(S, S)`). A `(70, 70)` float tensor is valid. No `is_causal` conflict since the decoder uses `batch_first=True` and does not set causal masking. ✓

The design correctly handles the optional final norm: `if self.decoder.norm is not None: out2 = self.decoder.norm(out2)`. ✓

### Architectural Feasibility
- 1 scalar parameter (`kin_bias_scale`) + one `(70,70)` float32 buffer (~20 KB). Negligible memory.
- `kin_bias_scale` is on `model.head`, auto-included in `head_params` group (LR=1e-4, weight_decay=0.3). ✓
- No OOM risk at batch=4.

### Constraint Adherence
- Soft additive bias only, never -inf ✓
- Learned scalar init=0.0 ✓
- Applied only in pass 2 ✓
- Loss unchanged from baseline ✓
- All baseline HPs preserved ✓
- No modifications to `infra.py` or transforms ✓

No issues found. Design is complete, correct, and feasible.
