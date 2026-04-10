# Design Review — idea014/design001

**Design:** Depth PE + Wide Head (No LLRD)
**Reviewer Verdict:** APPROVED

## Summary

This design combines the continuous depth PE from idea008/design003 with the wide head from idea009/design002 (hidden_dim=384), using a flat optimizer (no LLRD, no progressive unfreezing). Only config.py is meaningfully changed (head_hidden from 256 to 384).

## Evaluation

### Completeness
- All head dimension changes are explicitly listed: input_proj Linear(1024->384), joint_queries Embedding(70,384), decoder d_model=384, dim_feedforward=1536, output linears updated. These are correct for hidden_dim=384.
- The design correctly states that model.py requires no structural changes because Pose3DHead parameterizes all sizes through hidden_dim.
- The flat optimizer groups (backbone at 1e-5, depth_pe at 1e-4, head at 1e-4) match idea008/design003 exactly.
- All 19 config fields are explicitly listed with correct values.
- The file-level edit plan is clear: only config.py changes (head_hidden=384, output_dir).

### Mathematical Correctness
- 384/8 = 48 per head -- valid for multi-head attention.
- Parameter count: ~9.88M for wide head vs ~5.48M baseline. ~308M total. Well within 11GB VRAM at batch=4.

### Architectural Feasibility
- Continuous depth PE (sqrt spacing, 16 anchors) is explicitly preserved unchanged.
- vit.pos_embed remains zeroed and frozen.
- No LLRD, no progressive unfreezing -- correctly isolated to test depth-PE + wide-head interaction.

### Constraint Adherence
- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup_epochs=3, grad_clip=1.0 all fixed.
- weight_decay=0.03, lambda_depth=0.1, lambda_uv=0.2 match idea.md spec for design 1.
- lr_backbone=1e-5 (flat rate, not LLRD base rate) -- correctly noted in Builder instructions.
- infra.py, loss formulation (Smooth L1 beta=0.05) unchanged.

### Concerns
- None. Clean, minimal change that isolates the variable of interest.

## Verdict: APPROVED
