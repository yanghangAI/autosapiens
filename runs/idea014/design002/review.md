# Design Review — idea014/design002

**Design:** LLRD + Depth PE + Wide Head (Triple Combination)
**Reviewer Verdict:** APPROVED

## Summary

This design combines all three proven improvements: LLRD (gamma=0.90, unfreeze_epoch=5), continuous depth PE (sqrt spacing), and wide head (hidden_dim=384). It is the most ambitious combination in the pipeline. Starting from idea008/design003, it requires config.py changes and porting LLRD optimizer logic to train.py.

## Evaluation

### Completeness
- The LLRD schedule is fully specified for both phases:
  - Phase 1 (epochs 0-4): blocks 0-11 frozen, blocks 12-23 get per-layer LR with LLRD, depth PE at 1e-4, head at 1e-4. 14 param groups.
  - Phase 2 (epoch 5+): all unfrozen, patch+pos embed at lr ~ 8.014e-6 (= 1e-4 * 0.90^24), blocks 0-23 with LLRD, depth PE at 1e-4, head at 1e-4. 27 param groups.
- The LR formula `lr_i = 1e-4 * 0.90^(23-i)` for block i is explicitly stated.
- Depth PE params are correctly identified as NOT subject to LLRD decay (head-level LR).
- Wide head specs match idea009/design002: hidden_dim=384, num_heads=8, num_layers=4, ffn_dim=1536.
- All 21 config fields are explicitly listed with correct values.
- File-level edit plan is clear: config.py (4 changes), model.py (no changes), train.py (port LLRD from idea004/design002).
- Builder notes explicitly direct to port LLRD from runs/idea004/design002/code/train.py.

### Mathematical Correctness
- LLRD: 0.90^24 ~ 0.0798, so the shallowest block gets lr ~ 7.98e-6. The embed group gets 0.90^24 * 1e-4 ~ 8.0e-6. This matches the design's stated "~8.014e-6". Correct.
- Cosine schedule with warmup applied multiplicatively to all initial_lr values -- standard and correct.
- At optimizer rebuild (epoch 5), initial_lr is set per group and the current cosine scale is applied. This is the correct approach to avoid LR discontinuities.

### Architectural Feasibility
- Continuous depth PE preserved unchanged.
- Wide head propagates automatically from hidden_dim config.
- ~308M params total, well within 11GB VRAM at batch=4.
- The LLRD implementation pattern is proven from idea004/design002 and idea011 (which combined LLRD + depth PE successfully).

### Constraint Adherence
- BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20, warmup_epochs=3, grad_clip=1.0 all fixed.
- weight_decay=0.03 as specified for design 2 in idea.md.
- lambda_depth=0.1, lambda_uv=0.2, Smooth L1 beta=0.05 unchanged.
- infra.py and loss formulation unchanged.

### Concerns
- The train.py changes are non-trivial (porting LLRD logic), but the design correctly references the exact source (idea004/design002). The pattern has been implemented successfully multiple times (idea004, idea011). The Builder should have no ambiguity.

## Verdict: APPROVED
