# SearchSpaceArchitect Memory

## Baseline Architecture Summary

- **Backbone:** Sapiens ViT-B, outputs `(B, 1024, 40, 24)` feature map.
- **Head:** `Pose3DHead` — 70 learnable joint query embeddings, `nn.TransformerDecoder` (4 layers, 8 heads, hidden_dim=256). Joint queries **cross-attend** to the flattened backbone feature map. The decoder also has a **self-attention** sub-layer among joint queries (standard `TransformerDecoderLayer`).
- **Key API points for masking:** `nn.TransformerDecoder.forward(tgt, memory, tgt_mask=..., memory_mask=...)`. `tgt_mask` controls self-attention among the 70 joint queries. `memory_mask` controls cross-attention to image patches.
- **Optimizer:** AdamW, `lr=5e-4`, `weight_decay=0.1` (in baseline).
- **Fixed constants (infra.py, do not modify):** `BATCH_SIZE=4`, `ACCUM_STEPS=8`, `FRAME_STRIDE=5`, `NUM_JOINTS=70`.
- **Kinematic graph:** `SMPLX_SKELETON` in `infra.py` — edges over the 70 `ACTIVE_JOINT_INDICES` after `_ORIG_TO_NEW` remapping. BFS on this undirected graph gives hop distances.

## Ideas Log

### idea001 — RGB-D Modality Fusion Strategy
- **Status:** design001 APPROVED, design002 APPROVED, design003 APPROVED (2026-04-02). Designs needed: 3. All 3 approved. Design phase COMPLETE.
- **Review log:** `runs/idea001/review_log.md`

### idea002 — Kinematic Attention Masking
- **Status:** All 3 designs APPROVED on second review (2026-04-02). Design phase COMPLETE.
- **Designs needed:** 3 — all 3 approved.
- **Review log:** `runs/idea002/review_log.md`
- **idea_overview.csv status:** Updated to 'Designed'.
- **Next step:** Orchestrator should spawn Proxy Environment Builder to implement all 3 approved designs.

### idea003 — Curriculum-Based Loss Weighting
- **Status:** 2 designs APPROVED (2026-04-02). Design phase COMPLETE.
- **Designs needed:** 2 — both approved.
- **design001 (revised):** Homoscedastic Uncertainty Loss Weighting — learnable log-variance parameters (log_var_pose, log_var_depth, log_var_uv) as `nn.Parameter`, replaces the invalid Constant Weights (Baseline) design. Uses Kendall et al. 2018 formulation: `exp(-s) * L + s` per task. APPROVED.
- **design002:** Linear Warmup for Depth Loss — depth weight linearly ramps from 0 to 1.0 over first 10 epochs, pose/uv constant. APPROVED (previously reviewed).
- **idea_overview.csv status:** Updated to 'Designed'.
- **Note:** design001 originally was "Constant Weights (Baseline)" which was invalidated since the baseline is already trained. Replaced with homoscedastic uncertainty weighting.

### idea004 — Layer-Wise Learning Rate Decay & Progressive Unfreezing
- **Status:** All 6 designs reviewed and APPROVED. Design phase COMPLETE. (2026-04-02)
- **Designs:** 6 total — a 3×2 grid: gamma ∈ {0.95, 0.90, 0.85} × unfreeze_epoch ∈ {5, 10}
  - design001: gamma=0.95, unfreeze=5
  - design002: gamma=0.90, unfreeze=5
  - design003: gamma=0.85, unfreeze=5
  - design004: gamma=0.95, unfreeze=10
  - design005: gamma=0.90, unfreeze=10
  - design006: gamma=0.85, unfreeze=10
- **Key implementation notes:**
  - base_learning_rate = 1e-4 for shallowest backbone layers (consistent across all 6)
  - head_learning_rate = 1e-4
  - Per-layer LR: `LR_i = 1e-4 * gamma^i` where i=0 is shallowest block
  - Phase 1 (epochs 0 to unfreeze-1): freeze backbone, train head only
  - Phase 2 (epoch unfreeze onwards): unfreeze backbone, build per-layer param groups
  - Use same `get_lr_scale()` cosine schedule with warmup_epochs=3
  - Baseline uses flat LR=1e-5 for backbone (no LLRD), unfrozen from epoch 0 — all 6 designs are distinct
- **Design docs:** All 6 `design.md` files now have complete implementation specs (updated 2026-04-02)
- **Baseline check:** Baseline uses flat 1e-5 for backbone with no progressive unfreezing. None of the 6 designs duplicate this.
- **Trimming decision:** Kept all 6. The 3×2 grid is a minimal complete sweep over gamma and unfreeze_epoch axes. Removing any would create asymmetric coverage.
- **Results context:** results.csv only has data from idea001 and idea002 runs; idea004 not yet run.

### idea005 — Depth-Aware Positional Embeddings
- **Status:** All 3 designs APPROVED on self-review (2026-04-02). Design phase COMPLETE.
- **Designs needed:** 3 — all 3 approved.
- **Designs:**
  - design001: `discretized_depth_pe` — decomposed 1D row+col+depth-bucket embeddings (16 bins), zero-init depth_emb, head LR for PE module.
  - design002: `relative_depth_bias` — additive per-joint per-depth-bin bias on cross-attention logits (70×16 params), zero-init, manual decoder loop required.
  - design003: `depth_conditioned_pe` — continuous MLP (3→128→256→1024) maps (row_norm, col_norm, depth_norm) to additive PE correction; Xavier near-zero init; retains pretrained 2D pos_embed.
- **Review log:** `runs/idea005/review_log.md`
- **Key implementation notes:**
  - All designs require hooking into mmpretrain ViT between patch_embed and transformer layers.
  - Clamp depth bin indices to [0, num_depth_bins-1] (designs 1 and 2).
  - design002: verify attn_mask shape for nn.MultiheadAttention — test manually before full training.
  - design003: Builder must inspect vit attribute names via `print(model.backbone.vit)` before writing the manual forward override.
- **idea_overview.csv status:** Updated to 'Designed'.

### idea006 — Training Data Augmentation for Generalization
- **Status:** Defined (2026-04-08). Design phase NOT yet started.
- **Designs needed:** 6 novel designs.
- **Baseline starting point:** `baseline.py`
- **Key motivation:** All 5 prior ideas show a persistent 35–60mm train-val MPJPE gap. Zero augmentation is applied in the baseline. Augmentation is the most prominent unexplored axis.
- **Designs:**
  - design001: Horizontal Flip only (p=0.5) — with proper left-right joint swap
  - design002: Scale/Crop Jitter only (±20% bbox scale)
  - design003: Color Jitter only (RGB, torchvision, brightness/contrast/saturation/hue)
  - design004: Depth Channel Augmentation (Gaussian noise + 10% pixel dropout)
  - design005: Combined Geometric: Flip + Scale Jitter
  - design006: Full Stack: Flip + Color Jitter + Depth Noise
- **Key constraint for Designer:** Horizontal flip requires left-right joint swap in SMPLX topology and pelvis_uv[0] negation. Scale jitter modifies bbox before CropPerson; root-relative joints unaffected. Augmentations go into `build_train_transform()` only.
- **idea_overview.csv status:** Updated to 'Not Designed'.

### idea009 — Head Architecture Refinement
- **Status:** Defined (2026-04-09). Design phase NOT yet started.
- **Designs needed:** 5 novel designs.
- **Baseline starting point:** `runs/idea004/design002/train.py` (best completed: 112.3 mm val_mpjpe_body)
- **Key motivation:** No prior idea has varied the transformer decoder head. LLRD schedule is fixed to idea004/design002 (gamma=0.90, unfreeze=5) so gains are attributable to head changes only.
- **Designs:**
  - design001: 6-layer decoder (head_num_layers=6, head_hidden=256)
  - design002: Wide head (head_num_layers=4, head_hidden=384, input_proj Linear(1024→384))
  - design003: Sine-cosine joint query init (learnable but sinusoidal initialization for 70 queries)
  - design004: Per-layer input feature gate (4 learned scalar gates on cross-attention input, init=1.0)
  - design005: Output LayerNorm before final Linear(256→3)
- **Category A axes:** derive from idea004/design002 and idea001/design003
- **Category B axes:** query init priors, per-layer gating, output norm — all novel
- **idea_overview.csv status:** Updated to 'Not Designed'.

### idea010 — Multi-Scale Backbone Feature Aggregation
- **Status:** Defined (2026-04-10). Design phase NOT yet started.
- **Designs needed:** 5 novel designs.
- **Baseline starting point:** `runs/idea004/design002/train.py` (best completed: 112.3 mm val_mpjpe_body)
- **Key motivation:** All prior ideas consume only the final ViT layer output. Intermediate layers encode complementary spatial/structural info. Multi-scale aggregation (DPT/ViTDet-style) is standard in dense prediction but untried here.
- **Designs:**
  - design001: Last-4-layer concatenation + Linear(4096->1024) projection
  - design002: Last-4-layer learned weighted sum (4 softmax scalars)
  - design003: Feature pyramid from layers {4,8,12}, project each to 256ch, concat+project
  - design004: Cross-scale attention gate from layer 6 applied to layer 12
  - design005: Alternating layer average (layers {2,4,6,8,10,12}), zero extra params
- **Category A axes:** derive from idea004/design002 (best LLRD schedule) + standard multi-layer ViT feature usage
- **Category B axes:** feature pyramid, cross-scale gating, alternating average — all novel
- **idea_overview.csv status:** Updated to 'Not Designed'.

### idea011 — LLRD with Continuous Depth PE
- **Status:** Defined (2026-04-10). Design phase NOT yet started.
- **Designs needed:** 4 novel designs.
- **Baseline starting point:** `runs/idea008/design003/code/` (best completed: 112.0 mm val_mpjpe_weighted, 93.7 mm pelvis)
- **Key motivation:** Combine the two best orthogonal improvements: LLRD from idea004/design002 (best body MPJPE 112.3) and continuous depth PE from idea008/design003 (best weighted MPJPE 112.0). Never combined before. idea007 tried a weaker depth PE + LLRD combo and failed, but used bucketed (not continuous) depth PE.
- **Designs:**
  - design001: LLRD gamma=0.90, unfreeze=5 + sqrt continuous depth PE
  - design002: LLRD gamma=0.85, unfreeze=5 + sqrt continuous depth PE
  - design003: LLRD gamma=0.90, unfreeze=10 + sqrt continuous depth PE
  - design004: LLRD gamma=0.90, unfreeze=5 + gated continuous depth PE (from idea008/design002)
- **Category A axes:** all 4 designs combine independently strong ideas (exploit & extend)
- **idea_overview.csv status:** Updated to 'Not Designed'.

### idea012 — Regularization for Generalization
- **Status:** Defined (2026-04-10). Design phase NOT yet started.
- **Designs needed:** 5 novel designs.
- **Baseline starting point:** `runs/idea004/design002/train.py` (best completed: 112.3 mm val_mpjpe_body)
- **Key motivation:** Persistent 22-30 mm train-val gap across ALL completed experiments indicates overfitting. No prior idea has varied dropout, weight decay, stochastic depth, or added explicit regularization. idea006 tried augmentation (input-side) with mixed results; this targets model-side regularization.
- **Designs:**
  - design001: head_dropout=0.2 (up from 0.1)
  - design002: weight_decay=0.3 (up from 0.1)
  - design003: drop_path_rate=0.2 (up from 0.1)
  - design004: R-Drop consistency loss (MSE between two stochastic forward passes, alpha=1.0)
  - design005: Combined regularization (dropout=0.2 + weight_decay=0.2 + drop_path=0.2)
- **Category A axes:** dropout and weight decay scaling (existing knobs never tuned)
- **Category B axes:** stochastic depth increase, R-Drop, combined regularization — all novel
- **idea_overview.csv status:** Updated to 'Not Designed'.

### idea013 — Joint Prediction Loss Reformulation
- **Status:** Defined (2026-04-10). Design phase NOT yet started.
- **Designs needed:** 4 novel designs.
- **Baseline starting point:** `runs/idea004/design002/train.py` (best completed: 112.3 mm val_mpjpe_body)
- **Key motivation:** Every experiment uses Smooth L1 (beta=0.05) with equal joint weights. The loss function itself has never been varied. This is a fundamental and unexplored axis.
- **Designs:**
  - design001: Small-beta Smooth L1 (beta=0.01) — more L1-like, stronger gradients for medium errors
  - design002: Large-beta Smooth L1 (beta=0.1) — more L2-like, stronger early convergence
  - design003: Bone-length auxiliary loss (lambda_bone=0.1) — soft anatomical consistency from SMPLX_SKELETON edges
  - design004: Hard-joint-weighted loss — one-shot per-joint reweighting after epoch 0 based on error magnitude
- **Category A axes:** beta tuning (refining existing Smooth L1 parameter)
- **Category B axes:** bone-length loss, hard-joint weighting — both novel
- **idea_overview.csv status:** Updated to 'Not Designed'.

### idea014 — Best-of-Breed Combination: LLRD + Depth PE + Wide Head
- **Status:** Defined (2026-04-10). Design phase NOT yet started.
- **Designs needed:** 3 novel designs.
- **Baseline starting point:** `runs/idea008/design003/code/` (best completed: 112.0 mm val_mpjpe_weighted, 93.7 mm pelvis)
- **Key motivation:** Three independently strong improvements (LLRD, continuous depth PE, wide head) have never been combined. Each modifies a different model component. idea011 tests LLRD+depth PE; this adds wide head to the mix and tests pairwise/triple combinations.
- **Designs:**
  - design001: Depth PE + Wide Head (no LLRD) — pairwise combination
  - design002: LLRD + Depth PE + Wide Head — full triple combination
  - design003: LLRD + Depth PE + Wide Head + weight_decay=0.3 — triple with regularization
- **Category A axes:** all 3 designs combine independently proven winners (exploit & extend)
- **idea_overview.csv status:** Updated to 'Not Designed'.

## Review Principles Applied

- Designs must specify the exact Python API hook (argument names, tensor shapes, dtypes).
- Kinematic graph must be traced to `SMPLX_SKELETON` in `infra.py` for consistency.
- Any mask that can produce a fully `-inf` row in softmax is a NaN risk and must be flagged.
- Precomputed static tensors must be registered as buffers, not recomputed each forward.
- "Dense" control must pass `tgt_mask=None` explicitly to confirm identical behavior to baseline.
