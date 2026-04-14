# Experiment Summary

_Maintained by the Architect. Update at the start of each new idea definition session: reflect completed/failed experiments since the last update, then add a placeholder row for the new idea._

---

## Current SOTA

| Metric | Best | Run |
|---|---|---|
| val_mpjpe_weighted | **98.92 mm** | idea015/design004 |
| val_mpjpe_body | **102.51 mm** | idea015/design004 |
| val_pelvis_err | **89.95 mm** | idea015/design001 |

---

## Findings by Idea

| Idea | Name | Outcome | Notes |
|---|---|---|---|
| idea001 | RGB-D Modality Fusion | ~125–130 mm | Baseline-level; fusion strategy alone insufficient |
| idea002 | Kinematic Attention Masking | ~125–130 mm | No gain over baseline |
| idea003 | Curriculum Loss Weighting | ~143–163 mm | Degraded; homoscedastic weighting hurt convergence |
| idea004 | LLRD + Progressive Unfreezing | **112.3 mm** | First real gain; gamma=0.90, unfreeze=5 is best config |
| idea005 | Depth-Aware Positional Embeddings | ~116–127 mm | Bucketed depth PE marginal; continuous PE better (idea008) |
| idea006 | Data Augmentation | ~124–128 mm | Mostly neutral; flip+scale hurt pelvis badly |
| idea007 | LLRD + Bucketed Depth PE | ~130–133 mm | Degraded vs idea008; bucketed PE hurts when combined with LLRD |
| idea008 | Continuous Depth PE | **111.9 mm / 93.7 mm pelvis** | Breakthrough for pelvis; sqrt-mapped continuous PE (design003) is best |
| idea009 | Head Architecture Refinement | ~112–113 mm | Marginal; wide head (design002) best but small gain |
| idea010 | Multi-Scale Backbone Features | ~111–122 mm | Marginal; cross-scale gate (design004) best but small gain |
| idea011 | LLRD + Continuous Depth PE | **109.9 mm** | Additive combination worked; design001 (gamma=0.90, unfreeze=5) best |
| idea012 | Regularization | ~113–120 mm | Neutral to negative; R-Drop and combined reg hurt |
| idea013 | Loss Reformulation | ~112–117 mm | Neutral; bone-length aux loss (design003) no gain |
| idea014 | LLRD + Depth PE + Wide Head | **106.85 mm / 103.51 weighted** | Triple combo worked; design003 set SOTA at the time |
| idea015 | Iterative Refinement Decoder | **98.92 weighted** (design004) | design004 (two-decoder refine) = SOTA weighted+body; design001 = best pelvis (89.95); design002–003 failed |
| idea016 | 2.5D Heatmap + Soft-Argmax | ~178–211 mm (designs001/002/004) | Much worse than regression head; heatmap representation failed entirely |
| idea017 | Temporal Context (Adjacent Frames) | ALL FAILED | design001 cv2 data error; design002–004 OOM |
| idea018 | Weight Averaging (EMA / SWA) | ALL FAILED OOM | Extra model copy (~1.2 GB) pushed all variants over 11 GB limit |
| idea019 | Anatomical Priors on Iterative Refinement | ~102–107 mm | design002 (kinematic soft-attn bias) best body 105.77; design004 best weighted 102.22; design003 still training (poor at epoch 6); anatomical priors marginal vs SOTA |
| idea020 | Refinement-Specific Loss & Gradient Strategy | — (not started) | 5 designs varying loss/gradient flow on two-decoder SOTA (idea015/design004) |
| idea021 | Anatomical Priors on Two-Decoder SOTA | — (not started) | 4 designs re-testing best idea019 priors on the correct (stronger) baseline |

---

## Known Failure Modes

| Failure | Cause | Affected |
|---|---|---|
| **OOM** | GPU is 11 GB (1080 Ti); any design using >~10.5 GB allocated crashes | idea015/d002–003, idea016/d003, idea017/d002–004, idea018/d001–004 |
| **cv2 resize assertion** | Empty image array in DataLoader worker (bad/missing file path) | idea017/design001 |

### OOM-prone patterns to avoid
- Extra full model copy on GPU (EMA shadow, SWA accumulator, teacher model): ~1.2 GB per copy — **does not fit**
- Multi-frame memory with all frames trainable (idea017/design002): activations double
- 3D volumetric heatmap at native resolution (40×24×16): large intermediate tensors
- Multi-pass decoders with >2 passes and deep supervision: activation accumulation across passes (but 2-pass with independent refine decoder DOES fit — idea015/design004 succeeded)

---

## Dead Ends

- **Heatmap representation** (idea016): all completed designs scored ~178–211 mm — far worse than regression head; do not revisit
- **Temporal fusion** (idea017): memory budget incompatible with 11 GB; would need fp16 or multi-GPU
- **Weight averaging / EMA / SWA** (idea018): extra model copy does not fit; would need fp16 or offloading
- **Bucketed depth PE + LLRD** (idea007): worse than either alone

---

## Promising Directions

- **Two-decoder refinement** (idea015/design004) is the new overall SOTA (98.92 weighted, 102.51 body) — independent 2-layer refine decoder outperforms shared-decoder approach
- idea015/design001 retains best pelvis error (89.95 mm) suggesting query injection approach is better for pelvis specifically
- Anatomical priors (idea019) gave marginal body improvement (~105.77 best) but did not beat idea015/design004 on weighted metric
- The 22-28 mm train-val gap persists across all SOTA configs, suggesting overfitting remains the key bottleneck
- Continuous sqrt-mapped depth PE + LLRD + wide head combination remains the foundational stack
