# Reviewer Memory

This file serves as the persistent memory storage for the Reviewer agent to store state, notes, and lessons learned across workflow executions. Keep it concise.

---

## Completed Reviews

| Design_ID | Date | Verdict | Notes |
|-----------|------|---------|-------|
| idea006/design001 | 2026-04-08 | APPROVED | Horizontal Flip only; mathematically correct; FLIP_PAIRS matches infra.py; pipeline order correct (flip before SubtractRoot); all config fields specified. |
| idea006/design001 (code) | 2026-04-08 | APPROVED | Implementation matches design exactly; all 16 config fields correct; no hardcoded params in train.py; one inert dead param (scale_jitter) in build_train_transform — harmless. |
| idea006/design002 | 2026-04-08 | APPROVED | Scale/Crop Jitter only; bbox half-extents scaled by s~Uniform(0.8,1.2) around center; 3D metric joints unaffected; CropPerson handles OOB via padding; both bbox formats covered; all config fields specified. |
| idea006/design002 (code) | 2026-04-09 | APPROVED | Implementation matches design exactly; RandomScaleJitter formula correct; pipeline order correct; bbox.copy() prevents aliasing; OOB handled by CropPerson padding; all 17 config fields correct; train.py and model.py unchanged. |
| idea006/design003 | 2026-04-08 | APPROVED | Color Jitter only (RGB); RGBColorJitter wrapper applied after ToTensor; depth and joints unchanged; params match idea.md spec exactly; full class code provided; all config fields specified. |
| idea006/design003 (code) | 2026-04-09 | APPROVED | RGBColorJitter matches design: exact jitter params, RGB-only application after ToTensor, depth untouched, config matches spec, train.py/model.py unchanged; wrapper safely unnormalizes and re-normalizes around torchvision jitter. |
| idea006/design004 | 2026-04-08 | APPROVED | Depth Channel Augmentation only (Gaussian noise σ=0.02 + pixel dropout 10%, each p=0.5); applied after ToTensor on float32 depth in [0,1]; labels and RGB untouched; full class code provided; all config fields specified. |
| idea006/design005 | 2026-04-08 | APPROVED | Combined Geometric (Flip + Scale Jitter); ordering correct (ScaleJitter→CropPerson→Flip→SubtractRoot→ToTensor); both classes identical to APPROVED design001/002; pelvis_uv not manually negated (SubtractRoot handles implicitly); all config fields specified. |
| idea006/design006 | 2026-04-08 | APPROVED | Full Stack (Flip + Color Jitter + Depth Noise); composes APPROVED designs 001, 003, 004; pipeline ordering correct; all params match idea.md Axis 6; Scale Jitter correctly excluded; all config fields specified. |
| idea006/design006 (code) | 2026-04-09 | APPROVED | Builder fixed the rejected value-domain bug in RGBColorJitter by unnormalizing to [0,1], applying ColorJitter, then renormalizing; full stack now matches design, config/train/model remain aligned, and the sanity check passed. |

---

## Lessons Learned

- For horizontal flip designs: verify that pelvis_uv is NOT manually negated when SubtractRoot follows the flip transform — SubtractRoot recomputes it automatically from the negated-Y joints.
- FLIP_PAIRS in infra.py is in remapped 0..69 index space; designs should import directly from infra rather than redefine.
- For RGBColorJitter after ToTensor: if ToTensor normalizes RGB, the implementation must unnormalize before calling torchvision ColorJitter and then renormalize afterward.
idea007/design002 reviewed on 2026-04-09. Approved: strong LLRD on depth-bucket PE, correct freeze/unfreeze and optimizer grouping, config matches spec, sanity check passed.
idea007/design003 reviewed on 2026-04-09. Approved: earlier unfreeze at epoch 3 is implemented correctly; LLRD formulas and optimizer groups match design; config matches spec; sanity check passed.
idea009/design001 reviewed on 2026-04-09. Approved (design + code): 6-layer decoder (Axis A1); param math correct (+2.1M, head ~7.9M); all 16 config fields specified; single-integer change in config.py; train.py passes head_num_layers through correctly; LLRD grouping automatically covers new layers; 2-epoch test passed.
idea009/design002 reviewed on 2026-04-09. Approved (design + code): wide head hidden_dim=384 (Axis A2); divisibility 384/8=48 valid; param math correct (+4.4M, head ~9.88M, total ~303M); all 16 config fields specified; no structural code change needed; head_hidden propagates automatically through Pose3DHead; LLRD grouping auto-covers wider layers; stale docstring in train.py cosmetic only; 2-epoch test passed.
idea009/design003 reviewed on 2026-04-09. Approved: sine-cosine joint query init (Axis B1); PE formula and log-space numerics correct; initialization-only change, no parameter count or VRAM impact; isolated to model.py _init_weights; all 16 config fields specified; Builder instructions unambiguous.
idea009/design003 (code) reviewed on 2026-04-09. Approved: _sinusoidal_init correctly implemented as @staticmethod; formula and log-space numerics match design exactly; torch.no_grad() wraps copy_(); trunc_normal_ preserved for regression heads; all 16 config fields correct; 2-epoch smoke test passed with sanity check [0.0, 1.0, 0.0, 1.0].
idea009/design004 reviewed on 2026-04-09. Approved (design + code): per-layer input feature gate (Axis B2); 4 scalar gates via nn.ParameterList in Pose3DHead; sigmoid(4.6)≈0.99 init is honest and acceptable; forward loop over decoder.layers with norm guard is correct; 0-dim scalar broadcasts over (B,S,D) memory correctly; gates auto-include in lr_head group; 4 scalar params negligible; all 16 config fields specified; smoke test passed 308.8M params, 1.76/4.22GB VRAM.
idea009/design005 reviewed on 2026-04-09. Approved: output LayerNorm before regression (Axis B3); single nn.LayerNorm(256) after decoder output, before pelvis_token extraction; correct for pre-norm TransformerDecoder with no final norm arg; 512 params auto-included in lr_head; no VRAM impact; two one-line changes to model.py only; all 16 config fields specified.
idea010/design001 reviewed on 2026-04-10. Approved: last-4-layer concat (layers 20-23) + Linear(4096,1024) Xavier init; ~4.2M new params; aggregator optimizer group; all config fields specified.
idea010/design002 reviewed on 2026-04-10. Approved: learned layer weights, 4 softmax scalars init to 0; 4 new params; output shape unchanged; all config fields specified.
idea010/design003 reviewed on 2026-04-10. Approved: feature pyramid layers {7,15,23}; 3x Linear(1024,256) + Linear(768,1024); ~1.57M params; Xavier init; all config fields specified.
idea010/design004 reviewed on 2026-04-10. Approved: cross-scale gate layers {11,23}; Linear(1024,1) zero-weight bias=-5.0; residual form correct; Designer fixed idea.md's zero-bias error; 1025 params; all config fields specified.
idea010/design005 reviewed on 2026-04-10. Approved: alternating layer average (12 odd-indexed layers); zero params; running-sum recommended; no optimizer changes; all config fields specified.
idea010/design001 (code) reviewed on 2026-04-10. Approved: backbone extracts layers {20-23}, LN on each, concat to 4096, MultiScaleConcat Linear(4096,1024) Xavier init; aggregator wired in optimizer at lr_head; all 16 config fields match; 2-epoch passed.
idea010/design002 (code) reviewed on 2026-04-10. Approved: backbone returns list of 4 normed maps, LearnedLayerWeights with zeros(4) -> softmax uniform 0.25; aggregator wired in optimizer; all 16 config fields match; 2-epoch passed.
idea010/design003 (code) reviewed on 2026-04-10. Approved: backbone returns 3 normed maps from {7,15,23}; FeaturePyramid 3x Linear(1024,256) + Linear(768,1024) Xavier init; ~1.57M params; aggregator wired in optimizer; all 16 config fields match; 2-epoch passed.
idea010/design004 (code) reviewed on 2026-04-10. Approved: backbone returns [layer_11, layer_23]; CrossScaleGate Linear(1024,1) zero-weight bias=-5.0; residual form final*(1+gate); 1025 params; aggregator wired in optimizer; all 17 config fields match; 2-epoch passed.
idea010/design005 (code) reviewed on 2026-04-10. Approved: backbone uses running-sum over 12 odd-indexed layers with LN; zero new params; no aggregator; optimizer unchanged; all 16 config fields match; 2-epoch passed.
idea011/design001 reviewed on 2026-04-10. Approved: LLRD gamma=0.90 unfreeze=5 + sqrt continuous depth PE; direct combo of idea004/d002 + idea008/d003; all 19 config fields specified.
idea011/design002 reviewed on 2026-04-10. Approved: LLRD gamma=0.85 unfreeze=5 + sqrt continuous depth PE; same as d001 with steeper decay.
idea011/design003 reviewed on 2026-04-10. Approved: LLRD gamma=0.90 unfreeze=10 + sqrt continuous depth PE; later unfreeze variant.
idea011/design004 reviewed on 2026-04-10. Approved: LLRD gamma=0.90 unfreeze=5 + gated depth PE from idea008/d002; depth_gate in PE group.
idea012/design001 reviewed on 2026-04-10. Approved: head_dropout 0.1->0.2; config-only change.
idea012/design002 reviewed on 2026-04-10. Approved: weight_decay 0.03->0.3; config-only change; idea.md has minor error (says baseline 0.1, actual 0.03).
idea012/design003 reviewed on 2026-04-10. Approved: drop_path 0.1->0.2; config-only change.
idea012/design004 reviewed on 2026-04-10. Approved: R-Drop MSE consistency alpha=1.0; no_grad second pass; body joints only; train.py change.
idea012/design005 reviewed on 2026-04-10. Approved: combined head_dropout=0.2 + weight_decay=0.2 + drop_path=0.2; config-only change.
idea011/design001 (code) reviewed on 2026-04-10. Approved: LLRD gamma=0.90 unfreeze=5 + sqrt depth PE; all 19 config fields correct; LLRD logic, freeze/unfreeze, depth PE grouping all match design; model.py/transforms.py unchanged from idea008/design003.
idea011/design002 (code) reviewed on 2026-04-10. Approved: config-only variant of d001 (gamma=0.85); train.py identical, parameterized.
idea011/design003 (code) reviewed on 2026-04-10. Approved: config-only variant of d001 (unfreeze_epoch=10); train.py identical, parameterized.
idea011/design004 (code) reviewed on 2026-04-10. Approved: LLRD gamma=0.90 unfreeze=5 + gated depth PE; model.py from idea008/design002 with depth_gate; depth_gate auto-captured in depth_pe group.
idea012/design001 (code) reviewed on 2026-04-10. Approved: head_dropout=0.2; config-only change from idea004/design002 baseline.
idea012/design002 (code) reviewed on 2026-04-10. Approved: weight_decay=0.3; config-only change.
idea012/design003 (code) reviewed on 2026-04-10. Approved: drop_path=0.2; config-only change.
idea012/design004 (code) reviewed on 2026-04-10. Approved: R-Drop MSE consistency alpha=1.0; no_grad second pass; body joints only; train.py change correct.
idea012/design005 (code) reviewed on 2026-04-10. Approved: combined head_dropout=0.2 + weight_decay=0.2 + drop_path=0.2; config-only change.
idea013/design001 reviewed on 2026-04-10. Approved: small-beta Smooth L1 (beta=0.01); single constant change in train.py; all config fields specified.
idea013/design002 reviewed on 2026-04-10. Approved: large-beta Smooth L1 (beta=0.1); single constant change in train.py; all config fields specified.
idea013/design003 reviewed on 2026-04-10. Approved: bone-length auxiliary loss (lambda_bone=0.1); 21 body edges from SMPLX_SKELETON; no new params; all config fields specified.
idea013/design004 reviewed on 2026-04-10. Approved: hard-joint-weighted loss; one-shot weights after epoch 0; clamp [0.5,2.0] sum=22; all config fields specified.
idea014/design001 reviewed on 2026-04-10. Approved: depth PE + wide head (hidden=384) no LLRD; config-only change (head_hidden=384); flat optimizer preserved; all 19 config fields specified.
idea014/design002 reviewed on 2026-04-10. Approved: LLRD + depth PE + wide head triple combo; gamma=0.90 unfreeze=5; depth PE at head-level LR; all 21 config fields specified.
idea014/design003 reviewed on 2026-04-10. Approved: triple combo + weight_decay=0.3; config-only variant of design002; all 21 config fields specified.
idea013/design001 (code) reviewed on 2026-04-10. Approved: F.smooth_l1_loss beta=0.01; single change in train.py; depth/UV losses unchanged; all config fields correct; LLRD logic correct.
idea013/design002 (code) reviewed on 2026-04-10. Approved: F.smooth_l1_loss beta=0.1; single change in train.py; all config fields correct; LLRD logic correct.
idea013/design003 (code) reviewed on 2026-04-10. Approved: bone_length_loss from SMPLX_SKELETON body edges; 0.1*l_bone in loss; all config fields correct.
idea013/design004 (code) reviewed on 2026-04-10. Approved: one-shot joint weights after epoch 0; clamp [0.5,2.0] sum=22; weighted per-joint Smooth L1; err_accum dict pattern; all config fields correct.
idea014/design001 (code) reviewed on 2026-04-10. Approved: head_hidden=384 config-only; flat 3-group optimizer (backbone 1e-5, depth_pe 1e-4, head 1e-4); DepthBucketPE with sqrt spacing; all config fields correct.
idea014/design002 (code) reviewed on 2026-04-10. Approved: triple combo LLRD+depth PE+wide head; per-block LLRD gamma=0.90; freeze/unfreeze at epoch 5; depth PE at head-level LR; 14->27 groups; all config fields correct.
idea014/design003 (code) reviewed on 2026-04-10. Approved: config-only variant of design002 with weight_decay=0.3; train.py/model.py identical to design002.
idea015/design001 reviewed on 2026-04-11. Approved: two-pass shared decoder query injection; refine_mlp Linear(3→384→384); out1+R query conditioning; deep supervision 0.5/1.0; ~151K new params.
idea015/design002 reviewed on 2026-04-11. Approved: Gaussian cross-attention bias in pass 2; sigmoid(0)*10=5.0 init is non-zero at step 0 (spec violation, non-fatal); manual norm_first layer loop correct; clamped to -1e4 (not -inf); ~1.2K params.
idea015/design003 reviewed on 2026-04-11. Approved: three-pass shared decoder; progressive DS 0.25/0.5/1.0; shared refine_mlp across transitions; ~152K params; no issues.
idea015/design004 reviewed on 2026-04-11. Approved: independent 2-layer refine_decoder (~4.87M params, slightly above 3M spec estimate but within budget); head optimizer group auto-includes; ~4.87M new params.
idea016/design001 reviewed on 2026-04-11. Approved: 2D heatmap + scalar depth (40×24); exact per-joint UV projection using K and gt_pelvis_abs; decode_joints_heatmap helper; scale mismatch (UV vs metres) is experimental; ~369K new params.
idea016/design002 reviewed on 2026-04-11. Approved: bilinear upsample logits to 80×48 before softmax; 4× more bins; no param increase; clean extension of d001.
idea016/design003 reviewed on 2026-04-11. Approved: full 3D volumetric heatmap (15360 bins); sqrt depth bins match depth PE; stale unused x_pel variable in decode helper (non-fatal); ~5.9M params.
idea016/design004 reviewed on 2026-04-11. Approved: 2D heatmap + Gaussian MSE auxiliary (lambda_hm=0.1, sigma=2.0, BODY_IDX); make_gaussian_targets fully specified; MSE on normalized softmax output; no issues.
idea017/design001 reviewed on 2026-04-11. Approved: 8-channel channel stacking [RGB_t,D_t,RGB_t-5,D_t-5]; patch_embed widened, mean init for new channels; DepthBucketPE unchanged (reads index 3); single pass.
idea017/design002 reviewed on 2026-04-11. Approved: both backbone passes gradient-checkpointed; 1920-token memory; shared backbone LLRD unchanged; OOM dry-run required; fallback batch=2/accum=16 specified.
idea017/design003 reviewed on 2026-04-11. Approved: past frozen (no_grad+detach), centre trainable; 1920-token memory; ~7-8 GB peak; no checkpointing needed; clean design.
idea017/design004 reviewed on 2026-04-11. Approved: three-frame symmetric [prev,t,next]; past+future frozen; 2880-token memory; ~8-9 GB estimated; dry-run recommended; minor mems[] indexing clarity issue (non-fatal).
idea015/design001 (code) reviewed on 2026-04-11. Approved: two-pass shared decoder query injection; refine_mlp+joints_out2 in Pose3DHead; loss 0.5/1.0; LLRD grouping auto-includes new head params; 2-epoch test passed.
idea015/design002 (code) reviewed on 2026-04-11. Approved: Gaussian cross-attention bias; manual norm_first loop correct; attn_bias_scale init=0 (raw param, not sigmoid*10 — minor deviation, non-fatal); 2-epoch test passed.
idea015/design003 (code) reviewed on 2026-04-11. Approved: three-pass shared decoder; shared refine_mlp; joints_out/out2/out3; loss 0.25/0.5/1.0; 2-epoch test passed (higher initial MPJPE expected for 3-pass warmup).
idea015/design004 (code) reviewed on 2026-04-11. Approved: independent 2-layer refine_decoder inside Pose3DHead (auto head group); ~4.87M new params; loss 0.5/1.0; 2-epoch test passed.
idea016/design001 (code) reviewed on 2026-04-11. Approved: heatmap_out(384→960); grid buffers registered; exact GT UV projection with per-joint X_ref; custom validate_heatmap; decode_joints_heatmap; 2-epoch test passed.
idea016/design002 (code) reviewed on 2026-04-11. Approved: bilinear upsample to 80×48 before softmax; buffers at upsampled resolution; train loop identical to d001; 2-epoch test passed.
idea016/design003 (code) reviewed on 2026-04-11. Approved: 3D volumetric heatmap 15360 bins; sqrt depth bins from infra.DEPTH_MAX_METERS; d_scale=10.0 normalization; stale x_pel in decode helper (cosmetic); 2-epoch test passed.
idea016/design004 (code) reviewed on 2026-04-11. Approved: heatmap_soft exposed in return dict; make_gaussian_targets normalized; l_hm on BODY_IDX; lambda_hm=0.1; faster epoch-2 convergence vs d001; 2-epoch test passed.
idea017/design001 (code) reviewed on 2026-04-11. Approved: 8-channel patch_embed; mean init for new channels; DepthBucketPE reads index 3 (unchanged); TemporalBedlamDataset with _crop_frame_with_bbox; 2-epoch test passed.
idea017/design002 (code) reviewed on 2026-04-11. Approved: both backbone passes gradient-checkpointed (use_reentrant=False); 1920-token cat memory; single-frame validation fallback; no OOM; 2-epoch test passed (73s/epoch).
idea017/design003 (code) reviewed on 2026-04-11. Approved: past frame no_grad+detach; centre full gradient; no checkpointing; 1920-token memory; 2-epoch test passed (41s/epoch).
idea017/design004 (code) reviewed on 2026-04-11. Approved: past+future frozen; 2880-token memory [prev,t,next]; three-frame dataset; boundary clamped; minor mems[] indexing non-obvious but correct; 2-epoch test passed (50s/epoch).
idea018/design001 reviewed on 2026-04-11. Approved: EMA decay=0.999; update after each effective step; validate on EMA model; correct formula; 345 MB extra; sanity checks valid.
idea018/design002 reviewed on 2026-04-11. Approved: EMA warmup decay_t=min(0.9995,(1+s)/(10+s)); dict step counter; validates on EMA; checkpoint saves ema_step; all config fields specified.
idea018/design003 reviewed on 2026-04-11. Approved: SWA epochs 15-19; constant LR=0.5×cosine@ep15≈0.051×initial_lr; running average (n+1) formula correct; SWA model allocated only at epoch 15; LayerNorm no recalibration needed.
idea018/design004 reviewed on 2026-04-11. Approved (with flagged risk): EMA same as d001 + polish pass at epoch 21 (LR=1e-6); metrics.csv append workaround provided; iter_metrics.csv append mode NOT addressed — Builder must handle IterLogger header issue.
idea018/design001 (code) reviewed on 2026-04-11. Approved: EMA decay=0.999; update at accum boundary; validate(ema_model); checkpoint saves live+ema; 2-epoch test passed (4.17GB/8.51GB GPU).
idea018/design002 (code) reviewed on 2026-04-11. Approved: warmup formula min(0.9995,(1+s)/(10+s)); dict step counter; ema_step in checkpoint; ema_warmup config field informational only (harmless); 2-epoch test passed.
idea018/design003 (code) reviewed on 2026-04-11. Approved: SWA start_epoch=15 (never fires in 2-epoch test, correct); constant LR locked at first SWA entry; running avg formula exact; 2-epoch test passed (3.00GB — no SWA copy allocated).
idea018/design004 (code) reviewed on 2026-04-11. Approved: EMA same as d001; polish pass deletes optimizer/scaler/ema_model before proceeding; flat AdamW at 1e-6; writes to iter_metrics_polish.csv (avoids header collision); csv.DictWriter append to metrics.csv; 3-stage smoke test passed (polished.pth created, polish val body=556.8mm).
idea019/design001 reviewed on 2026-04-12. Approved: bone_length_loss on J2 with lambda_bone=0.1; 21 body edges from SMPLX_SKELETON (a<22 and b<22); zero new params; all config fields specified.
idea019/design002 reviewed on 2026-04-12. Approved: kin_bias (70,70) buffer + kin_bias_scale scalar (init=0.0); BFS hop weights 1.0/0.5/0.25; manual pass-2 decoder loop with tgt_mask=bias; never -inf; 1 new param; auto in head_params.
idea019/design003 reviewed on 2026-04-12. Approved: symmetry_loss on J2 with lambda_sym=0.05; 6 L/R limb segment pairs all verified against infra.py edges; all indices within BODY_IDX range; zero new params.
idea019/design004 reviewed on 2026-04-12. Approved (non-fatal flag): group_emb Embedding(4,384) zero-init added to queries2 before pass 2; body joints 0-21 all correctly assigned; _TORSO uses [23,24] as new-space indices causing minor misassignment of two non-body joints (left_eye→group3, hand joint→group0); cosmetic only, no loss impact; Builder should fix to [22,23]; 1536 new params.
idea019/design005 reviewed on 2026-04-12. Approved: union of designs 001+002+003; all three anatomical priors correctly combined; 1 new scalar param; redundant import alias in _build_kin_bias inert.
idea019/design001 (code) reviewed on 2026-04-12. Approved: bone_length_loss on J2; BODY_EDGES from SMPLX_SKELETON; lambda_bone=0.1 from config; loss formula 0.5/1.0 + 0.1*bone; no model.py changes; clean.
idea019/design002 (code) reviewed on 2026-04-12. Approved: _build_kin_bias BFS correct; kin_bias buffer+kin_bias_scale(init=0) in head; manual pass-2 loop with tgt_mask=bias; norm guard; auto head_params; loss unchanged.
idea019/design003 (code) reviewed on 2026-04-12. Approved: symmetry_loss on J2; SYM_PAIRS 6 pairs all in BODY_IDX; lambda_sym=0.05 from config; loss formula correct; no model.py changes.
idea019/design004 (code) reviewed on 2026-04-12. Approved: group_emb Embedding(4,384) zero-init; joint_group_ids buffer (70,) correct assignments; delta added to queries2 before pass 2; auto head_params; loss unchanged; minor eye-index cosmetic non-fatal.
idea019/design005 (code) reviewed on 2026-04-12. Approved: clean union of 001+002+003; module-level collections import; kin_bias+scale in model.py; bone_length_loss+symmetry_loss in train.py; combined loss formula correct; all 4 config fields present.
idea020/design001 reviewed on 2026-04-13. Approved: stop-gradient J1.detach() before refine_mlp; correct gradient decoupling; zero new params; all config fields specified.
idea020/design002 reviewed on 2026-04-13. Approved: coarse weight 0.5→0.1; refine_loss_weight=0.1 in config; zero new params; all config fields specified.
idea020/design003 reviewed on 2026-04-13. Approved: l_pose2 = F.l1_loss (pure L1) on J2 only; coarse retains Smooth L1; zero new params; all config fields specified.
idea020/design004 reviewed on 2026-04-13. Approved: optimizer split into coarse-head (1e-4) and refine-head (2e-4); helper functions specified; group index references updated; lr_refine_head=2e-4 in config; zero new params.
idea020/design005 reviewed on 2026-04-13. Approved: residual J2 = J1 + delta; depth_out/uv_out unaffected; zero-init warm-start correct; zero new params; all config fields specified.
idea021/design001 reviewed on 2026-04-13. Approved: kin_bias in refine_decoder only; BFS helper correct; float additive mask; kin_bias_scale=0 init; 1 new scalar param in head group.
idea021/design002 reviewed on 2026-04-13. Approved: group_emb Embedding(4,384) zero-init; joint_group_ids buffer 70 joints; unsqueeze(0) broadcast correct; 1536 new params in head group.
idea021/design003 reviewed on 2026-04-13. Approved: bone_length_loss on J2 with lambda=0.05; BODY_EDGES filter correct; zero new params; applied to J2 only.
idea021/design004 reviewed on 2026-04-13. Approved: clean union of idea021/design001+design002; both zero-init; 1537 new params; coarse decoder unchanged; no bone/sym loss.
idea020/design001 (code) reviewed on 2026-04-13. Approved: J1.detach() confirmed in model.py line 217; train.py loss 0.5/1.0 unchanged; config correct; 2-epoch passed (w=803.6mm).
idea020/design002 (code) reviewed on 2026-04-13. Approved: 0.1*l_pose1 + 1.0*l_pose2 in train.py; model.py unchanged; config refine_loss_weight=0.1; 2-epoch passed (w=825.9mm).
idea020/design003 (code) reviewed on 2026-04-13. Approved: F.l1_loss for l_pose2; coarse Smooth L1 unchanged; model.py unchanged; config refine_loss_weight=0.5; 2-epoch passed (w=787.6mm).
idea020/design004 (code) reviewed on 2026-04-13. Approved: _coarse_head_params()/_refine_head_params() split correct; group indices exact match to spec; config lr_refine_head=2e-4; 2-epoch passed (w=648.3mm).
idea020/design005 (code) reviewed on 2026-04-13. Approved: delta=joints_out2(out2); J2=J1+delta; zero-bias init preserved; train.py/config unchanged; 2-epoch passed (w=988.5mm; warm-start expected).
idea021/design001 (code) reviewed on 2026-04-13. Approved: _compute_kin_bias() BFS correct; kin_bias buffer + kin_bias_scale(zeros(1)); tgt_mask=bias_matrix to refine_decoder only; train.py unchanged; 2-epoch passed (w=797.9mm).
idea021/design002 (code) reviewed on 2026-04-13. Approved: group_emb Embedding(4,384) zero-init; joint_group_ids buffer correct; queries2+group_bias.unsqueeze(0) before refine_decoder; train.py unchanged; 2-epoch passed (w=929.5mm).
idea021/design003 (code) reviewed on 2026-04-13. Approved: bone_length_loss on J2 with BODY_EDGES; SMPLX_SKELETON import+filter correct; lambda_bone=0.05 from config; model.py unchanged; 2-epoch passed (w=795.6mm).
idea021/design004 (code) reviewed on 2026-04-13. Approved: clean union of d001+d002; ordering correct (group_emb first, then kin_bias to tgt_mask); 1537 new params all zero-init; train.py unchanged; 2-epoch passed (w=929.5mm).
