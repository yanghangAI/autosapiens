2026-04-09: Drafted idea008/design001 as continuous interpolated depth positional encoding from runs/idea005/design001. Stopped after single design draft for reviewer handoff.
2026-04-09: Fixed idea008/design001 review issue by moving interpolation/backbone-forward responsibilities explicitly to code/model.py and limiting code/train.py to optimizer wiring only if needed.
2026-04-09: Began idea009 (Head Architecture Refinement). 5 designs planned.
  - design001: 6-layer decoder (head_num_layers=6), all else from idea004/design002. APPROVED.
  - design002: Wide head (head_hidden=384, head_num_heads=8, head_num_layers=4). APPROVED.
  - design003: Sine-cosine joint query init (head_num_layers=4, head_hidden=256; sinusoidal init replaces trunc_normal_ in _init_weights). APPROVED.
  - design004: Per-layer input feature gate (head_num_layers=4, head_hidden=256; nn.ParameterList of 4 scalar sigmoid gates on memory, init=4.6 so sigmoid≈0.99; per-layer decoder loop in forward). Awaiting review.
2026-04-10: Drafted idea010 (Multi-Scale Backbone Feature Aggregation). 5 designs, all from runs/idea004/design002.
  - design001: Last-4-layer concat + Linear(4096,1024). Xavier init. ~4.2M new params. LLRD unchanged.
  - design002: Learned layer weights -- 4 softmax scalars (init uniform). Only 4 new params. LLRD unchanged.
  - design003: Feature pyramid 3 scales (layers 7,15,23). Three Linear(1024,256) + Linear(768,1024). ~1.57M params.
  - design004: Cross-scale attention gate (layers 11,23). sigmoid(Linear(1024,1)) * (1+gate). bias_init=-5.0. ~1K params.
  - design005: Alternating layer average (12 even-spaced layers). Running-sum approach. 0 new params.
  Key note: Sapiens 0.3B has 24 blocks (not 12). Corrected all layer indices from idea.md accordingly.
2026-04-10: Drafted idea011 (LLRD + Continuous Depth PE Combination). 4 designs.
  - design001: LLRD gamma=0.90, unfreeze=5, from idea008/design003 (sqrt depth PE). Direct combo.
  - design002: LLRD gamma=0.85, unfreeze=5, from idea008/design003. More aggressive decay.
  - design003: LLRD gamma=0.90, unfreeze=10, from idea008/design003. Later unfreezing.
  - design004: LLRD gamma=0.90, unfreeze=5, from idea008/design002 (gated depth PE). Gate + LLRD.
  All designs: model.py unchanged, only train.py gets LLRD per-block groups + freeze/unfreeze.
2026-04-10: Drafted idea012 (Regularization for Generalization). 5 designs, all from runs/idea004/design002.
  - design001: Head dropout 0.2 (head_dropout=0.2). Config-only change.
  - design002: Weight decay 0.3 (weight_decay=0.3). Config-only change.
  - design003: Stochastic depth 0.2 (drop_path=0.2). Config-only change.
  - design004: R-Drop consistency (rdrop_alpha=1.0). New config field + train.py loop change (second no_grad forward pass, MSE on body joints).
  - design005: Combined regularization (head_dropout=0.2, weight_decay=0.2, drop_path=0.2). Config-only change.
2026-04-11: Drafted idea017 (Temporal Adjacent-Frame Fusion). 4 designs, all from runs/idea014/design003/code/.
  - design001: Delta-input channel stacking. 8-channel input [RGB_t, D_t, RGB_t-5, D_t-5]. Widen patch_embed Conv2d(4→8). Single backbone pass. Init 4 new channel weights = mean of original 4.
  - design002: Cross-frame memory attention (2-frame, both trainable). Shared backbone called twice with torch.utils.checkpoint. Memory = cat([proj(feat_prev), proj(feat_t)], dim=1) → (B, 1920, 384). May need batch_size=2+accum=16 if OOM.
  - design003: Cross-frame memory attention (2-frame, past frozen no_grad). Past-frame pass in torch.no_grad()+detach(). Only centre-frame backbone trained. Memory = (B, 1920, 384). ~7-8 GB estimate.
  - design004: Three-frame symmetric fusion (t-5, t, t+5). Past+future in torch.no_grad()+detach(). Memory = cat([prev, t, next]) → (B, 2880, 384). ~8-9 GB estimate.
  Key notes: All designs share the same LLRD optimizer structure. Dataloader subclasses/wraps BedlamFrameDataset to add temporal frames using the SAME CROP BBOX as the centre frame. past_frame_idx = max(0, frame_idx-1) in dataset-index space. Validation runs in single-frame fallback mode (x_prev=None, x_next=None).
2026-04-11: Drafted idea018 (Weight Averaging: EMA and SWA on SOTA Triple-Combo). 4 designs, all from runs/idea014/design003/code/.
  - design001: EMA fixed decay=0.999. EMA updated after each effective optimizer step. Validate on EMA model. Sanity: decay=0 → live model.
  - design002: EMA warmup decay schedule: decay_t = min(0.9995, (1+step)/(10+step)). Ramps from ~0 to 0.9995 over ~2000 steps. Same validation protocol as design001.
  - design003: SWA over last 5 epochs (epochs 15-19). Constant LR = 0.5×cosine_LR@epoch15. Uniform running average of epoch-end snapshots. Validate on SWA model during SWA phase.
  - design004: EMA (decay=0.999) for 20 epochs + 1 polish epoch at flat LR=1e-6. Load EMA weights into live model, train 1 more epoch. Final metric from polished weights.
  Key notes: All designs modify only train.py + add ema_decay/swa fields to config.py. No model.py changes. LayerNorm-only model needs no BN recalibration. Extra model copy ≈345 MB, within 11 GB budget.
2026-04-11: Drafted idea015 (Iterative Refinement Decoder on SOTA Triple Combo). 4 designs, all from runs/idea014/design003/code/.
  - design001: Two-pass shared-decoder refinement (query injection). refine_mlp: Linear(3→384→384), joints_out2: Linear(384→3). Loss: 0.5*L(J1) + 1.0*L(J2). ~151K new params.
  - design002: Two-pass shared-decoder refinement (Gaussian cross-attn bias from J1). Learnable attn_bias_scale init=0. Manual norm_first=True decoder loop for pass 2 with additive Gaussian bias (sigma=2.0, expanded to (B*heads,70,960)). pelvis_uv anchor from self.uv_out(out1[:,0,:]). Loss: 0.5*L(J1) + 1.0*L(J2). ~1.2K new params.
  - design003: Three-pass shared-decoder refinement. Shared refine_mlp across 2 transitions. joints_out2+joints_out3. Loss: 0.25*L(J1)+0.5*L(J2)+1.0*L(J3). ~152K new params.
  - design004: Two-pass two-decoder refinement (independent 2-layer refine decoder). refine_decoder: TransformerDecoder 2-layer/384/8/1536/norm_first=True. All new params in head group (LR=1e-4, WD=0.3). ~4.87M new params. Loss: 0.5*L(J1)+1.0*L(J2).
  Key notes: All designs modify only model.py (Pose3DHead) and train.py (loss). config.py gets informational fields only. model.head.parameters() automatically picks up new submodules.
2026-04-10: Drafted idea014 (Best-of-Breed Combination: LLRD + Depth PE + Wide Head). 3 designs, all from runs/idea008/design003.
  - design001: Depth PE + Wide Head (no LLRD). head_hidden=384, flat optimizer (lr_backbone=1e-5). Config-only change.
  - design002: LLRD + Depth PE + Wide Head (triple combo). head_hidden=384, gamma=0.90, unfreeze_epoch=5, lr_backbone=1e-4. train.py needs LLRD logic from idea004/design002.
  - design003: Same as design002 but weight_decay=0.3 (from 0.03). Regularization counterbalance for larger param space.
2026-04-10: Drafted idea013 (Joint Prediction Loss Reformulation). 4 designs, all from runs/idea004/design002.
  - design001: Small-beta Smooth L1 (beta=0.01). Single line change in train.py.
  - design002: Large-beta Smooth L1 (beta=0.1). Single line change in train.py.
  - design003: Bone-length auxiliary loss (lambda_bone=0.1). 21 body-only edges from SMPLX_SKELETON. New bone_length_loss function in train.py.
  - design004: Hard-joint-weighted loss. One-shot per-joint weight computation after epoch 0, clamp [0.5, 2.0], normalize to sum=22. Weighted Smooth L1 for epochs 1-19.
  Key note: pose_loss() in infra.py uses beta=0.05. Designs 1-2 bypass it with direct F.smooth_l1_loss call. Body joints are indices 0-21 (BODY_IDX = slice(0,22)).
2026-04-11: Drafted idea016 (2.5D Heatmap Soft-Argmax). 4 designs, all from runs/idea014/design003/code.
  - design001: 2D heatmap (40×24) + scalar depth per joint. Linear(384,960). Soft-argmax in [0,1] UV. GT UV computed from intrinsics+pelvis depth. Loss in UV+Z space. decode_joints_heatmap helper for MPJPE.
  - design002: Same as design001 but bilinearly upsample 40×24→80×48 logits before softmax. Tests grid resolution limit.
  - design003: Full 3D volumetric heatmap (40×24×16=15360 bins). Linear(384,15360)=5.9M params. sqrt-spaced depth bins reused from DepthBucketPE. d_abs normalized by DEPTH_MAX_METERS in loss. decode_joints_3d helper.
  - design004: Same as design001 + auxiliary Gaussian MSE loss (lambda_hm=0.1, sigma=2.0 grid cells). make_gaussian_targets helper in train.py. heatmap_soft returned in model output dict.
  Key notes: pelvis auxiliary heads (depth_out, uv_out) unchanged in all 4 designs. MPJPE needs decode helper (UV+Z→metres). ViT patch grid h_tok=40, w_tok=24. IMG_H=640, IMG_W=384.
2026-04-12: Drafted idea019 (Anatomical Structure Priors for Iterative Refinement). 5 designs, all from runs/idea015/design001/code/.
  - design001: Bone-length auxiliary loss on J2 (lambda_bone=0.1). BODY_EDGES from SMPLX_SKELETON filtered to a<22 and b<22. Pure train.py change, 0 new params.
  - design002: Kinematic-chain soft self-attention bias in refinement pass 2 only. BFS hop distances (1→+1.0, 2→+0.5, 3→+0.25) in kin_bias (70,70) buffer. Learnable kin_bias_scale init=0.0. Manual layer loop for pass 2, tgt_mask=kin_bias_scale*kin_bias. 1 new param.
  - design003: Left-right symmetry loss on J2 (lambda_sym=0.05). 6 symmetric limb segment pairs. Pure train.py change, 0 new params.
  - design004: Joint-group query initialization in refinement pass 2. group_emb Embedding(4,384) zero-init added to queries2 before pass 2. joint_group_ids (70,) buffer: 0=torso, 1=arms, 2=legs, 3=extremities(22+). 1536 new params.
  - design005: Combined (A1+A2+B1): bone_loss(0.1) + sym_loss(0.05) + kin_bias in pass 2. Minimal new params (1 scalar). Tests max-prior configuration.
  Key notes: BODY_IDX = slice(0,22). SMPLX_SKELETON in infra.py remapped via _ORIG_TO_NEW. Manual decoder loop for kin_bias requires iterating self.decoder.layers with optional self.decoder.norm. Symmetric limb pair indices must be verified from _SMPLX_BONES_RAW + _ORIG_TO_NEW.
