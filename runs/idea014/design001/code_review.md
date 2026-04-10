# Code Review — idea014/design001

**Design:** Depth PE + Wide Head (No LLRD)
**Reviewer verdict:** APPROVED

## Checklist

1. **Config verification:**
   - `output_dir` = correct path (idea014/design001)
   - `head_hidden` = 384 — widened from 256, matches design
   - `head_num_heads` = 8, `head_num_layers` = 4 — unchanged, correct
   - `lr_backbone` = 1e-5 — flat optimizer rate, NOT the LLRD base rate. Correct per design.
   - `lr_head` = 1e-4, `lr_depth_pe` = 1e-4 — correct
   - `weight_decay` = 0.03, `epochs` = 20, `warmup_epochs` = 3 — correct
   - `num_depth_bins` = 16 — correct
   - `grad_clip` = 1.0, `lambda_depth` = 0.1, `lambda_uv` = 0.2 — correct
   - No `gamma` or `unfreeze_epoch` fields — correct, no LLRD.

2. **model.py** — Uses DepthBucketPE with continuous interpolation and sqrt spacing. Pose3DHead parameterized via `hidden_dim` which reads from `head_hidden=384`. Input_proj Linear(1024->384), joint_queries Embedding(70, 384), decoder d_model=384, dim_feedforward=384*4=1536, output linears updated. All automatic from `hidden_dim` parameter. Correct.

3. **train.py — Flat optimizer** — Lines 201-213: Three param groups (backbone, depth_pe, head) with flat LRs. No LLRD, no progressive unfreezing. Correct per design.
   - `backbone_params` filters out depth_pe_ids. Correct.
   - `depth_pe_params` at `lr_depth_pe=1e-4`. Correct.
   - `head params` at `lr_head=1e-4`. Correct.

4. **No structural model.py changes needed** — Model parameterizes all sizes through `hidden_dim`. Verified: Pose3DHead uses `hidden_dim` for input_proj, joint_queries, decoder, and output heads. Correct.

5. **Loss formulation** — Standard pose_loss (Smooth L1 beta=0.05), lambda_depth=0.1, lambda_uv=0.2. Unchanged. Correct.

6. **vit.pos_embed zeroed and registered as buffer** — Line 302-306 in model.py. Correct.

## Issues

None found.
