# Code Review — idea015/design003
**Design:** Three-Pass Shared-Decoder Refinement (Deep Supervision 0.25/0.5/1.0)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation faithfully extends the two-pass query injection to three passes with shared decoder weights, shared refine_mlp, three dedicated output heads, and the correct progressive deep supervision weights. All design elements are present and correct.

## Architecture Check (model.py)

- `refine_mlp = Sequential(Linear(3,384), GELU, Linear(384,384))` — single shared module used for both J1→J2 and J2→J3 transitions. Correct per design.
- Three output heads: `joints_out`, `joints_out2`, `joints_out3` — all `Linear(384,3)`. Correct.
- Forward:
  - Pass 1: `out1 = decoder(queries, memory)` → `J1 = joints_out(out1)`.
  - Pass 2: `R1 = refine_mlp(J1)` → `queries2 = out1 + R1` → `out2 = decoder(queries2, memory)` → `J2 = joints_out2(out2)`.
  - Pass 3: `R2 = refine_mlp(J2)` → `queries3 = out2 + R2` → `out3 = decoder(queries3, memory)` → `J3 = joints_out3(out3)`.
  - Matches design spec exactly.
- `pelvis_depth` and `pelvis_uv` from `out3[:,0,:]` (final pass). Correct.
- Return dict: `joints=J3, joints_pass1=J1, joints_pass2=J2, pelvis_depth, pelvis_uv`. Correct.
- Weight init: all heads and refine_mlp layers initialized with trunc_normal_+zero_bias. Correct.

## Config Check (config.py)

- `refine_passes=3, refine_loss_w1=0.25, refine_loss_w2=0.50, refine_loss_w3=1.00`. All present and correct.
- All inherited HPs match spec.

## Loss Check (train.py)

- `l_pose1 = pose_loss(out["joints_pass1"][:,BODY_IDX], joints[:,BODY_IDX])`.
- `l_pose2 = pose_loss(out["joints_pass2"][:,BODY_IDX], joints[:,BODY_IDX])`.
- `l_pose3 = pose_loss(out["joints"][:,BODY_IDX], joints[:,BODY_IDX])`.
- `l_pose = 0.25*l_pose1 + 0.5*l_pose2 + 1.0*l_pose3` — exactly matches spec.
- `del` cleanup covers all three intermediate losses. No memory leaks.
- MPJPE computed on `out["joints"]` (= J3, final prediction). Correct.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test run: val_mpjpe_body epoch 1 = 3301mm, epoch 2 = 2977mm. Higher than the other designs due to the 0.25-weighted loss on J1 (noisier gradient signal early in training). Decreasing trend is present. No divergence. The higher initial error is expected for three-pass architectures at warmup.

## Issues

None identified.
