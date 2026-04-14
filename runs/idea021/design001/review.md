# Review: idea021 / design001 — Kinematic Soft-Attention Bias in Refine Decoder Only (Axis A1)

**Design_ID:** idea021/design001
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design adds a BFS-derived kinematic attention bias to the refine decoder only. A learnable scalar `kin_bias_scale` (init=0.0) scales a precomputed `(70,70)` hop-distance buffer and is passed as `tgt_mask` to `self.refine_decoder`. The coarse decoder is unchanged. One new trainable parameter.

## Evaluation

### Completeness
All required config fields are specified. The `_compute_kin_bias()` helper function is fully provided in the design. Model changes are precisely listed (4 steps). No train.py changes. Config changes limited to `output_dir`. Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- The BFS hop-weight assignment (+1.0/+0.5/+0.25 for hops 1/2/3, 0 beyond) matches the idea.md spec exactly.
- The BFS implementation is correct: `visited` dict prevents re-visiting; `hop >= 3` early-exit is correct (processes exactly hops 0, 1, 2 to generate hop-1, hop-2, hop-3 neighbors).
- Float mask semantics: PyTorch `nn.TransformerDecoder` with a float `tgt_mask` (not boolean) adds it as an additive bias to the attention logits before softmax. Since all values are ≥ 0 (no -inf), no attention positions are blocked — the constraint from idea.md is satisfied.
- `kin_bias_scale` init=0.0: at training start, bias matrix is all zeros, so the refine decoder is identical to baseline. Correct zero-initialization behavior.
- `batch_first=True` with `tgt_mask` shape `(70,70)` is valid for PyTorch `TransformerDecoderLayer`. Confirmed: PyTorch broadcasts the mask across batch and heads when the mask is 2D and the decoder has `batch_first=True`.
- `kin_bias_scale` is in `model.head.parameters()` automatically → head group (LR=1e-4, WD=0.3). Correct.

### Constraint Adherence
- 1 new trainable scalar + 1 buffer. Negligible VRAM and parameter overhead.
- Architecture unchanged except for the refine decoder `tgt_mask` addition.
- Coarse decoder unchanged: `out1 = self.decoder(queries, memory)` with `tgt_mask=None`.
- All training hyperparameters preserved.

### Issues
None.

## Verdict: APPROVED
