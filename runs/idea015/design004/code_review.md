# Code Review — idea015/design004
**Design:** Two-Pass Two-Decoder Refinement (Independent 2-Layer Refine Decoder)
**Reviewer:** Reviewer Agent
**Date:** 2026-04-11
**Verdict:** APPROVED

---

## Summary

Implementation correctly implements the independent 2-layer refinement decoder. The refine_decoder, refine_mlp, and joints_out2 are all defined inside `Pose3DHead`, ensuring they are automatically captured in the head optimizer group. Architecture and loss match the design exactly.

## Architecture Check (model.py)

- Primary `decoder` = 4-layer TransformerDecoder (norm_first=True). Unchanged from baseline.
- `refine_mlp = Sequential(Linear(3,384), GELU, Linear(384,384))` — correct.
- `refine_decoder` = `TransformerDecoder(TransformerDecoderLayer(d=384, nhead=8, ffn=1536, dropout=0.1, batch_first=True, norm_first=True), num_layers=2)` — matches design exactly (2 independent layers, all params separate from coarse decoder).
- `joints_out2 = Linear(384,3)` — correct.
- Forward: `out1 = decoder(queries, memory)` → `J1 = joints_out(out1)` → `R = refine_mlp(J1)` → `queries2 = out1 + R` → `out2 = refine_decoder(queries2, memory)` → `J2 = joints_out2(out2)`. Matches design.
- `pelvis_depth`, `pelvis_uv` from `out2[:,0,:]`. Correct.
- Return: `joints=J2, joints_coarse=J1`. Correct.

## Config Check (config.py)

- `refine_passes=2, refine_decoder_layers=2, refine_loss_weight=0.5`. All present.
- All inherited HPs match spec. Output_dir correct.

## Optimizer Group Assignment

- `refine_decoder`, `refine_mlp`, `joints_out2` are all attributes of `model.head` (`Pose3DHead`). They are automatically included in `list(model.head.parameters())` → head optimizer group (LR=1e-4, WD=0.3). No separate optimizer group needed. Matches design spec.

## Loss Check (train.py)

- `l_pose = 0.5*l_pose1 + 1.0*l_pose2`. Matches spec.
- Uses `out["joints_coarse"]` and `out["joints"]`. Correct.

## Parameter Count

- refine_decoder: 2 layers × ~2.36M = ~4.72M params. Design stated ~4.87M (including refine_mlp + joints_out2). Actual count consistent with design's own corrected estimate.
- Idea.md stated "~3M" but design004 self-corrected to ~4.87M and confirmed it still fits in 11GB. Valid.

## Metrics Sanity (test_output/metrics.csv)

- 2-epoch test run: val_mpjpe_body epoch 1 = 1044mm, epoch 2 = 790mm. Steadily decreasing. No divergence. Healthy warmup profile.

## Issues

None identified.
