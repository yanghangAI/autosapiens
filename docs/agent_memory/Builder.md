# ProxyEnvironmentBuilder Memory

## Current Task
- Idea: idea017 (Temporal Adjacent-Frame Fusion): ALL 4 DESIGNS IMPLEMENTED AND TESTED.

## idea017 Status
All 4 designs implemented and sanity-tested (2-epoch proxy runs), all starting from runs/idea014/design003/code/:
- design001 (8-channel stack, Delta-Input): PASSED job 55438467. val body MPJPE = 800.3mm. GPU mem: 7.35GB reserved.
- design002 (2-frame cross-attention, both trainable, grad ckpt): PASSED job 55438751. val body MPJPE = 863.1mm. GPU mem: 7.95GB reserved.
- design003 (2-frame cross-attention, past frozen no_grad): PASSED job 55438812. val body MPJPE = 975.0mm. GPU mem: 7.39GB reserved.
- design004 (3-frame symmetric, past+future frozen no_grad): PASSED job 55438813. val body MPJPE = 1173.1mm. GPU mem: 7.55GB reserved.

Key implementation notes:
- All 4 designs share the temporal dataset pattern: BedlamFrameDataset subclassed (transform=None to super), outer transform applied after aux frame fetch, manual bbox replication using _crop_frame_with_bbox(), same normalisation constants (RGB_MEAN/STD, DEPTH_MAX)
- design001: SapiensBackboneRGBD.__init__ takes in_channels param (default 4); load_sapiens_pretrained expands 3→4→8 (if n_ch==8, second expansion); SapiensPose3D.forward patched for validation (zero-pad 4ch→8ch) via _val_forward
- design002: Pose3DHead.forward(feat_t, feat_prev=None) → memory=cat([mem_prev, mem_t], dim=1) for 1920 tokens; SapiensPose3D.forward uses ckpt.checkpoint on BOTH passes; validation: _val_forward passes (x4, None)
- design003: Same head as design002; SapiensPose3D.forward: feat_t=backbone(x_t) full gradient, feat_prev=backbone(x_prev) inside no_grad+detach; validation: _val_forward passes (x4, None)
- design004: Pose3DHead.forward(feat_t, feats_context=None) → list of context feats → memory ordering [prev, t, next] = cat([mems[1], mems[0], mems[2]]); SapiensPose3D.forward: x_prev+x_next both no_grad+detach; validation: _val_forward passes (x4, None, None)
- All validation uses infra.validate() with model.forward monkey-patched to _val_forward (single-frame 4-channel) during val
- Temporal dataset fetch: past_idx=max(0, frame_idx-1), future_idx=min(n_frames-1, frame_idx+1) in dataset-index space

## Previous Task
- Idea: idea016 (2.5D Heatmap Soft-Argmax): ALL 4 DESIGNS IMPLEMENTED AND TESTED.

## idea016 Status
ALL 4 DESIGNS IMPLEMENTED AND TESTED. All starting from runs/idea014/design003/code/:
- design001 (2D Heatmap + Scalar Depth, 40×24 native): PASSED job 55438470. val body MPJPE = 562.5mm.
- design002 (2D Heatmap + Scalar Depth, 80×48 upsampled): PASSED job 55438752. val body MPJPE = 443.0mm.
- design003 (3D Volumetric Heatmap 40×24×16, integral pose): PASSED job 55438809. val body MPJPE = 800.2mm.
- design004 (2D Heatmap + Scalar Depth + Gaussian MSE aux): PASSED job 55438754. val body MPJPE = 437.1mm.

Key implementation notes:
- All heads output (u_norm, v_norm, ...) not (x,y,z) directly — requires custom validate function (validate_heatmap/validate_3d)
- design001/002/004: loss in UV+Z space; MPJPE uses decode_joints_heatmap()
- design003: loss in UVD_abs/d_scale space; MPJPE uses decode_joints_3d() (world coords minus pelvis)
- design002: upsample heatmap 40×24 → 80×48 via F.interpolate before softmax (in logit space)
- design003: imports DEPTH_MAX_METERS from infra; 3D head Linear(384, 40*24*16=15360); sqrt-spaced depth bins
- design003: decode_joints_3d BUG FIX — coordinate ordering must match GT: x_rel=Y_world_j-Y_world_pel, y_rel=Z_world_j-Z_world_pel, z_rel=X_world_j-X_world_pel
  - pelvis_abs ordering: (X_world=depth, Y_world=leftright, Z_world=updown)
  - joints ordering: (x_rel=Y_world_rel, y_rel=Z_world_rel, z_rel=X_world_rel=depth_rel)
- design004: returns heatmap_soft in output dict; make_gaussian_targets() helper; l_hm = F.mse_loss on body joints only
- SapiensPose3D updated to pass heatmap_h, heatmap_w (and upsample_factor for d002) to Pose3DHead
- design003 higher val MPJPE (~800mm vs ~500mm for 2D designs) is expected: 3D vol head outputs d_abs in [0,10m] range, untrained soft-argmax at 5m center, error ~3-4m from pelvis subtraction

## Previous Task
- Idea: idea015 (Iterative Refinement Decoder): ALL 4 DESIGNS IMPLEMENTED AND TESTED.

## idea015 Status
All 4 designs implemented and sanity-tested (2-epoch proxy runs), all starting from runs/idea014/design003/code/:
- design001 (Two-Pass Query Injection, shared decoder): PASSED, val body MPJPE = 1499.9mm, job 55438465. model.py: added refine_mlp (Linear 3->384->GELU->Linear 384->384), joints_out2 (Linear 384->3). train.py: loss = 0.5*l_pose1 + 1.0*l_pose2
- design002 (Two-Pass Gaussian Cross-Attn Bias, shared decoder): PASSED, val body MPJPE = 1380.4mm, job 55438473. model.py: added attn_bias_scale (scalar param, init=0), joints_out2, manual layer loop for pass2 with Gaussian bias on cross-attn logits. train.py: same loss as design001
- design003 (Three-Pass Query Injection, progressive deep sup): PASSED, val body MPJPE = 2977.3mm, job 55438474. model.py: shared refine_mlp + joints_out2 + joints_out3. train.py: loss = 0.25*l_pose1 + 0.5*l_pose2 + 1.0*l_pose3
- design004 (Two-Pass Independent 2-layer refine_decoder): PASSED, val body MPJPE = 790.96mm, job 55438475. model.py: added refine_mlp + refine_decoder (2-layer TransformerDecoder) + joints_out2. train.py: same loss as design001

Key implementation notes:
- design002: attn_bias_scale initialized to 0 (not sigmoid-wrapped), matching reviewer's correction note
- design002: H_tok=40, W_tok=24 for 640x384 image. Manual layer loop for pass2 with norm_first=True order
- design003: iter_logger still uses l_pose (combined), not per-pass losses (acceptable per design)
- design004: refine_decoder inside Pose3DHead so auto-included in head optimizer group (no changes to LLRD builder)
- GPU mem: design001=3.00GB, design002=3.00GB, design003~similar, design004~similar at batch=4

## Previous Task
- Idea: idea013 (Loss Function Variants): ALL 4 DESIGNS IMPLEMENTED AND TESTED.

## idea013 Status
All 4 designs implemented and sanity-tested (2-epoch proxy runs), all starting from runs/idea004/design002/:
- design001 (Small-Beta Smooth L1, beta=0.01): PASSED, val body MPJPE = 523.7mm, job 55387840. Train.py only: replaced pose_loss() with F.smooth_l1_loss(..., beta=0.01)
- design002 (Large-Beta Smooth L1, beta=0.1): PASSED, val body MPJPE = 525.1mm, job 55387841. Train.py only: replaced pose_loss() with F.smooth_l1_loss(..., beta=0.1)
- design003 (Bone-Length Auxiliary Loss): PASSED, val body MPJPE = 525.1mm, job 55387842. Train.py only: added bone_length_loss() using BODY_BONES from SMPLX_SKELETON, lambda_bone=0.1
- design004 (Hard-Joint-Weighted Loss): PASSED, val body MPJPE = 503.8mm, job 55387843. Train.py only: epoch 0 collects per-joint errors, computes weights clamped [0.5, 2.0], applies weighted smooth_l1_loss for epochs 1+

## Previous Task
- Idea: idea014 (Triple Combination: Depth PE + Wide Head +/- LLRD): ALL 3 DESIGNS IMPLEMENTED AND TESTED.

## idea014 Status
All 3 designs implemented and sanity-tested (2-epoch proxy runs), all starting from runs/idea008/design003/:
- design001 (Depth PE + Wide Head, no LLRD): PASSED, val body MPJPE = 755.3mm, job 55387836. Config-only: head_hidden 256->384
- design002 (LLRD + Depth PE + Wide Head): PASSED, val body MPJPE = 1188.5mm, job 55387837. Config: head_hidden=384, lr_backbone=1e-4, gamma=0.90, unfreeze_epoch=5. Train.py: LLRD optimizer logic from idea011
- design003 (LLRD + Depth PE + Wide Head + WD 0.3): PASSED, val body MPJPE = 1166.4mm, job 55387838. Same as design002 but weight_decay=0.3

Key implementation pattern:
- All designs from idea008/design003 (sqrt-spaced continuous depth PE)
- Design001: config-only change (head_hidden=384), flat optimizer unchanged
- Designs 002-003: identical train.py with LLRD logic (ported from idea011), differ only in config weight_decay
- model.py unchanged in all designs (head parameterized via hidden_dim from config)

## Previous Task
- Idea: idea011 (LLRD + Depth PE Combination): ALL 4 DESIGNS IMPLEMENTED AND TESTED.

## idea011 Status
All 4 designs implemented and sanity-tested (2-epoch proxy runs):
- design001 (LLRD gamma=0.90, unfreeze=5, sqrt depth PE): PASSED, val body MPJPE = 533.4mm, job 55375879
- design002 (LLRD gamma=0.85, unfreeze=5, sqrt depth PE): PASSED, val body MPJPE = 522.8mm, job 55375886
- design003 (LLRD gamma=0.90, unfreeze=10, sqrt depth PE): PASSED, val body MPJPE = 533.4mm, job 55375887
- design004 (LLRD gamma=0.90, unfreeze=5, gated depth PE): PASSED, val body MPJPE = 533.4mm, job 55375888

Key implementation pattern:
- All designs share same train.py LLRD logic; only config.py differs (gamma, unfreeze_epoch)
- Designs 001-003 from idea008/design003 (sqrt-spaced continuous depth PE)
- Design 004 from idea008/design002 (gated continuous depth PE with depth_gate param)
- model.py unchanged in all designs
- _build_optimizer_frozen(): blocks 0-11 frozen, train blocks 12-23 + depth_pe + head (14 groups)
- _build_optimizer_full(): all unfrozen, 27 groups (embed + 24 blocks + depth_pe + head)
- Optimizer rebuild at unfreeze_epoch with initial_lr set and LR scale applied

## Previous Task
- Idea: idea012 (Regularization Strategies): ALL 5 DESIGNS IMPLEMENTED AND TESTED.

## idea012 Status
All 5 designs implemented and sanity-tested (2-epoch proxy runs), all starting from runs/idea004/design002/:
- design001 (Head Dropout 0.2): PASSED, val body MPJPE = 555.2mm, job 55375881. Config-only: head_dropout 0.1->0.2
- design002 (Weight Decay 0.3): PASSED, val body MPJPE = 520.7mm, job 55375882. Config-only: weight_decay 0.03->0.3
- design003 (Stochastic Depth 0.2): PASSED, val body MPJPE = 521.4mm, job 55375883. Config-only: drop_path 0.1->0.2
- design004 (R-Drop Consistency): PASSED, val body MPJPE = 540.1mm, job 55375884. Config: rdrop_alpha=1.0. Train.py: second no_grad forward pass, MSE consistency loss on body joints
- design005 (Combined Regularization): PASSED, val body MPJPE = 545.9mm, job 55375885. Config-only: head_dropout=0.2, weight_decay=0.2, drop_path=0.2

## Previous Task
- Idea: idea010 (Multi-Scale Backbone Feature Aggregation): ALL 5 DESIGNS IMPLEMENTED AND TESTED.

## idea010 Status
All 5 designs implemented and sanity-tested (2-epoch proxy runs):
- design001 (Last-4-Layer Concat + Linear Proj): PASSED, val body MPJPE = 605.0mm, job 55374930
- design002 (Learned Layer Weights): PASSED, val body MPJPE = 513.5mm, job 55375139
- design003 (Feature Pyramid 3 Scales): PASSED, val body MPJPE = 612.5mm, job 55375141
- design004 (Cross-Scale Attention Gate): PASSED, val body MPJPE = 789.5mm, job 55375143
- design005 (Alternating Layer Average): PASSED, val body MPJPE = 594.7mm, job 55375144

Key implementation pattern for all designs:
- Custom backbone forward: `_run_vit_preamble()` replicates VisionTransformer preamble (patch_embed, resize_pos_embed, drop_after_pos, pre_norm), then manually iterates self.vit.layers
- `resize_pos_embed` imported from mmpretrain for correct pos_embed handling
- Designs 001-004 add aggregator module with own param group (lr=lr_head) in both optimizer builders
- Design005 has zero new params (averaging done in backbone forward)

## Previous Task

## idea009/design003 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design003/code/model.py
- Test job: 55349223, PASSED
- Final val body MPJPE: 547.5mm (2-epoch test run, proxy dataset)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design003/test_output/
- Change: model.py only — added `_sinusoidal_init` static method to Pose3DHead; added `import math`
- `_init_weights`: replaced `trunc_normal_(self.joint_queries.weight, std=0.02)` with `self._sinusoidal_init(self.joint_queries)`
- Verification: joint_queries[0, :4] = [0.0, 1.0, 0.0, 1.0] confirming sin(0)=0, cos(0)=1 as expected
- GPU mem: 1.76GB batch1 / 2.92GB batch2 — fits 11GB VRAM

## idea009/design002 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design002/code/config.py
- Test job: 55348946, PASSED
- Final val body MPJPE: 1258.2mm (2-epoch test run, proxy dataset)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design002/test_output/
- Change: config.py head_hidden = 256 → 384; no structural code changes needed
- model.py was already parameterized — hidden_dim flows through input_proj, joint_queries, decoder d_model, output heads
- GPU mem: 1.80GB batch1 / 3.00GB batch2 — fits 11GB VRAM (wider head uses slightly more than 256 baseline)

## idea009/design001 Status
- Status: APPROVED by Reviewer, marked Implemented via sync-status.

## idea005/design003 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea005/design003/code/model.py
- Test job: 55234104, PASSED
- Final val body MPJPE: 493.8mm (2-epoch test run)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea005/design003/test_output/
- Key implementation details:
  1. New `DepthConditionedPE` module in model.py: 3-layer MLP: Linear(3→128)+GELU, Linear(128→256)+GELU, Linear(256→1024)
     - Xavier uniform init with gain=0.01 on all layers, zero biases
  2. `SapiensBackboneRGBD.forward` manually bypasses `vit.forward()` to avoid double-adding pos_embed
     - Calls patch_embed(x) → returns (tokens, patch_resolution) tuple (unpack both)
     - Builds row/col grid normalized to [0,1], avg_pools depth channel: F.avg_pool2d(depth_ch, k=16, s=16) → (B,1,40,24)
     - Stacks coords: (B, 960, 3); passes through depth_cond_pe → (B, 960, 1024)
     - pe_base = vit.pos_embed (1, 960, 1024); x_tokens = patch_tokens + pe_base + pe_correction
     - Applies vit.drop_after_pos, vit.pre_norm, then manual loop: vit.layers + vit.ln1 at last layer
     - Reshapes: (B, 960, 1024) → (B, 40, 24, 1024) → .permute(0,3,1,2) → (B, 1024, 40, 24)
  3. Optimizer: 3 param groups — backbone ViT params (lr=1e-5), depth_cond_pe MLP (lr=1e-4), head (lr=1e-4)
     - Filter by name: 'depth_cond_pe' in name
  4. config.py: added lr_depth_pe = 1e-4
  5. train.py: lr_hd now uses optimizer.param_groups[2] (was [1]) since depth_pe is group [1]
  6. GPU memory: 2.33GB batch1 / 4.64GB batch2 — fits within 11GB VRAM

## idea005/design002 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea005/design002/code/model.py
- Test job: 55234001, PASSED
- Final val body MPJPE: 532.67mm (2-epoch test run)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea005/design002/test_output/
- Key implementation details:
  1. New `DepthAttentionBias` module in model.py: Linear(1, num_joints=70), zero-init weight+bias
  2. Computes depth patches via F.avg_pool2d(depth_ch, k=16, s=16), reshapes to (B, N_mem, 1),
     projects to (B, N_mem, 70), transposes to (B, 70, N_mem=960)
  3. `Pose3DHead.forward(feat, depth_ch)`: manual decoder loop over `self.decoder.layers`
     - norm_first=True layer order: norm1 → self_attn → dropout1, norm2 → multihead_attn → dropout2, norm3 → FFN → dropout3
     - depth_bias_expanded shape: (B*num_heads, num_joints, N_mem) = (B*8, 70, 960)
     - Passed as `attn_mask` to layer.multihead_attn (additive bias before softmax)
  4. `SapiensPose3D.forward`: extracts depth_ch = x[:, 3:4, :, :], passes to self.head(feat, depth_ch)
  5. Optimizer: 2 groups — backbone (lr=1e-5), head (includes DepthAttentionBias, lr=1e-4)
  6. train.py: added seed 2026, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  7. train.py: prints "FINAL_VAL_MPJPE_BODY: {value}" at end of main()
  8. GPU memory: 2.33GB batch1 / 4.63GB batch2 — fits 1080ti 11GB

## idea004/design001 Status
- Idea: idea004 (Layer-Wise Learning Rate Decay)
- design001 (Constant Decay LLRD, gamma=0.95, unfreeze_epoch=5): IMPLEMENTED, TEST PASSED.

## idea004/design001 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea004/design001/code/train.py
- Test job: 54988411, PASSED
- Final val body MPJPE: 530.07mm (2-epoch test run)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea004/design001/test_output/
- Key implementation details:
  1. LLRD: lr_i = 1e-4 * 0.95^(23-i) for block i; embed lr = 1e-4 * 0.95^24; head lr = 1e-4
  2. _build_optimizer_frozen(): epochs 0-4, blocks 0-11 + patch_embed + pos_embed frozen
     (requires_grad=False), param groups = blocks 12-23 (12) + head (1) = 13 total
  3. _build_optimizer_full(): epoch 5+, all unfrozen, param groups = embed(1) + blocks 0-23(24) + head(1) = 26 total
  4. At epoch == UNFREEZE_EPOCH (5): rebuild optimizer before LR scale is applied
  5. lr_backbone reported in metrics.csv = deepest block (block 23) LR
     - Frozen phase: optimizer.param_groups[11] (block 23 = index 11 in 0-based block 12..23)
     - Full phase: optimizer.param_groups[24] (block 23 = index 24 in [embed, block0..23, head])
  6. Random seed 2026 at start of main(); PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  7. build_dataloader does NOT accept worker_init_fn — omit it (num_workers=0 in test anyway)
  8. No weight saving; final val body MPJPE printed as single float at end of main()

## idea001 (Previous Work)
- Idea: idea001 (RGB-D Modality Fusion Strategy)
- design001 (early_4ch): IMPLEMENTED, TEST PASSED. APPROVED by Designer, marked Implemented.
- design002 (mid_fusion): IMPLEMENTED, TEST PASSED. Awaiting Designer code review.
- design003 (late_cross_attention): IMPLEMENTED, TEST PASSED. Awaiting Designer code review.

## idea001/design003 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea001/design003/train.py
- Test job: 54955332, PASSED
- Final val body MPJPE: 1480.5mm (2-epoch test run)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea001/design003/test_output/
- Key implementation details:
  1. New `DepthTokenizer`: Conv2d(1→64, k=16, s=16, padding=2) + LayerNorm(64);
     weight/bias zero-initialized → depth tokens = 0 at epoch 0
  2. New `DepthCrossAttention`: single cross-attention block with pre-norm+residual;
     q_proj(1024→256) Xavier, k/v_proj(64→256) Xavier, out_proj(256→1024) zero-init;
     FFN: Linear(1024→4096) → GELU → Dropout → Linear(4096→1024) → Dropout; Xavier init
     Uses F.scaled_dot_product_attention (PyTorch 2.6, scale kwarg supported)
  3. New `SapiensBackboneLateFusion`: 3-ch ViT unchanged; manually iterates ViT layers
     with per-layer grad_ckpt (use_reentrant=False) to reduce peak VRAM;
     also uses grad_ckpt on depth_cross_attn to avoid 60MB FFN activation OOM
  4. VisionTransformer.forward returns tuple(outs); use vit_out[-1] for the featmap
     (discovered during debugging; fixed with manual ViT iteration in _run_vit_with_ckpt)
  5. Memory: ViT backbone alone fills ~5GB → 10.65GB at FFN without grad ckpt;
     with per-layer grad ckpt on ViT layers + depth_cross_attn: 2.40GB batch1 / 4.76GB batch2
  6. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set via os.environ.setdefault
  7. Optimizer: 3 groups — vit (lr=1e-5), depth_adapter (tokenizer+cross_attn, lr=1e-4), head (lr=1e-4)
  8. fusion_strategy = "late_cross_attention", output_dir = runs/idea001/design003

## idea001/design002 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea001/design002/train.py
- Test job: 54933311, PASSED
- Final val body MPJPE: 600.88mm (2-epoch test run)
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea001/design002/test_output/
- Key changes from design001/train.py:
  1. New `DepthProjector` module: Conv2d(1→1024, k=16, s=16, padding=2) + LayerNorm;
     weight/bias zero-initialized so epoch-0 = pure RGB
  2. New `SapiensBackboneMidFusion`: 3-channel ViT, manually iterates `vit.layers`,
     injects `depth_proj(depth)` additively before block 12 (INJECT_BEFORE = 12)
  3. `resize_pos_embed` imported from `mmpretrain.models.utils` for manual forward
  4. `SapiensPose3D.forward(x)` accepts concatenated (B,4,H,W) and splits internally
     → rgb = x[:, :3], depth = x[:, 3:] → keeps infra.validate compatible
  5. `load_sapiens_pretrained`: skips 4-ch expansion; loads 3-ch weights directly;
     skips backbone.depth_proj.* keys (zero-init intentional)
  6. Optimizer: 3 groups — vit (lr=1e-5), depth_proj (lr=1e-4), head (lr=1e-4)
  7. `fusion_strategy = "mid_fusion"`, `output_dir = runs/idea001/design002`
- GPU memory: 2.33GB epoch 1 / 4.64GB epoch 2 — fits 1080ti 11GB

## idea002 (Previous Work)
- Idea: idea002 (Kinematic Attention Masking)
- design001 (Baseline Dense Control): train.py IMPLEMENTED and TEST PASSED. Awaiting Experiment Designer approval before proceeding.
- design002 (Soft Kinematic Mask): IMPLEMENTED, TEST PASSED. Awaiting Experiment Designer review.
- design003 (Hard Kinematic Mask): IMPLEMENTED, TEST PASSED. Awaiting Experiment Designer review.

## design001 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea002/design001/train.py
- Test job: 54892104, PASSED
- Final val body MPJPE: 517.03mm
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea002/design001/test_output/
- Key changes from baseline.py:
  1. Added `HOP_DIST` module-level constant (BFS on SMPLX_SKELETON from infra.py)
  2. Pose3DHead accepts `attention_method: str` arg (default "dense")
  3. Pose3DHead.forward passes `tgt_mask=None` explicitly
  4. SapiensPose3D passes `attention_method="dense"` to head
  5. Added global seed 2026 at start of main()
  6. Added _worker_init_fn for deterministic DataLoader workers
  7. Prints FINAL_VAL_MPJPE_BODY as single float to stdout
  8. output_dir default updated to runs/idea002/design001

## design002 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea002/design002/train.py
- Test job: 54893737, PASSED
- Final val body MPJPE: 307.12mm
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea002/design002/test_output/
- Key changes from design001:
  1. Pose3DHead.__init__: when attention_method="soft_kinematic_mask", computes
     soft_bias = HOP_DIST.float() * math.log(0.5) and registers as buffer
  2. Pose3DHead.forward: if attention_method=="soft_kinematic_mask", calls
     self.decoder(queries, memory, tgt_mask=self.soft_bias)
  3. SapiensPose3D in main() instantiated with attention_method="soft_kinematic_mask"
  4. output_dir updated to runs/idea002/design002

## design003 Status
- Path: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea002/design003/train.py
- Test job: 54894734, PASSED
- Final val body MPJPE: 297.41mm
- Test output: /work/pi_nwycoff_umass_edu/hang/auto/runs/idea002/design003/test_output/
- Key changes from design002:
  1. Pose3DHead.__init__: when attention_method="hard_kinematic_mask", computes
     hard_mask (70,70) float32: 0.0 where hop distance <= 2, -inf otherwise
     NaN guard: fully-masked rows (if any) set to dense (all True)
     Registered as self.register_buffer("hard_mask", hard_mask)
  2. Pose3DHead.forward: if attention_method=="hard_kinematic_mask", calls
     self.decoder(queries, memory, tgt_mask=self.hard_mask)
  3. SapiensPose3D in main() instantiated with attention_method="hard_kinematic_mask"
  4. output_dir updated to runs/idea002/design003
  5. No warmup or annealing; hard masking from epoch 0

## Key Infrastructure Notes
- baseline.py is at /work/pi_nwycoff_umass_edu/hang/auto/baseline.py
- infra.py exports: NUM_JOINTS, SMPLX_SKELETON, PELVIS_IDX, ACTIVE_JOINT_INDICES, etc.
- SMPLX_SKELETON uses remapped 0..69 indices (via _ORIG_TO_NEW)
- Test script: ./scripts/submit_test.sh — handles output_dir creation and overrides _Cfg.output_dir to <design_dir>/test_output
- Test job sets: epochs=2, max_train_seqs=2, max_val_seqs=1, num_workers=0
- CSV headers required:
  - iter_metrics.csv: epoch,iter,loss,loss_pose,loss_depth,loss_uv,mpjpe_body,pelvis_err,mpjpe_weighted
  - metrics.csv: epoch,lr_backbone,lr_head,train_loss,train_loss_pose,train_mpjpe_body,train_pelvis_err,train_mpjpe_weighted,val_loss,val_loss_pose,val_mpjpe_body,val_pelvis_err,val_mpjpe_weighted,epoch_time
- BODY_IDX = slice(0, 22) — body + pelvis joints only for MPJPE reporting
- GPU: 1080ti, 11GB VRAM — no FP16, amp=False. Batch 1 allocates ~2.3GB for design001
