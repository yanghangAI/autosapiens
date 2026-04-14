# Review: idea020 / design001 — Stop-Gradient on Coarse J1 Before Refinement (Axis A1)

**Design_ID:** idea020/design001
**Date:** 2026-04-13
**Verdict:** APPROVED

## Summary

This design implements a one-line stop-gradient change in `Pose3DHead.forward()`: `self.refine_mlp(J1.detach())` instead of `self.refine_mlp(J1)`. The motivation (decoupling gradient flows between coarse and refine decoders, standard in cascaded detection) is sound and well-explained.

## Evaluation

### Completeness
All required config fields are specified. The design clearly identifies exactly one change to `model.py` and zero changes to `train.py` and `config.py` (beyond `output_dir`). Builder instructions are unambiguous.

### Mathematical / Architectural Correctness
- The detach call is correctly placed: `J1.detach()` is passed to `refine_mlp`, so gradients from `l_pose2` cannot flow back through J1 into the coarse decoder.
- The coarse decoder still receives gradients from `0.5 * l_pose1` via `out1 → joints_out → J1 → l_pose1`, which is correct.
- `queries2 = out1 + R` still uses the non-detached `out1`, so gradients from `l_pose2` continue to flow back through `out1` into the coarse decoder. This is the intended partial decoupling: only the J1 coordinate path is stopped, not the feature path. This is consistent with the stated design intent ("The coarse decoder focuses entirely on its own 0.5*L(J1) objective").
- No risk of gradient disconnection or NaN.

### Constraint Adherence
- Zero new parameters. Identical VRAM to idea015/design004.
- Architecture unchanged: 4-layer coarse + 2-layer refine, wide head, depth PE, LLRD.
- All fixed config values (BATCH_SIZE=4, ACCUM_STEPS=8, epochs=20) respected.

### Issues
None.

## Verdict: APPROVED
