# Head Architecture Refinement: Query Initialization, Depth of Decoder, and Output Normalization

**Expected Designs:** 5

## Starting Point

The baseline starting point for this idea is:

`runs/idea004/design002/train.py`

That design produced the best completed validation score so far at **112.3 mm val_mpjpe_body** using LLRD (gamma=0.90, unfreeze_epoch=5) with the standard `Pose3DHead`. The optimization schedule is now well-characterized and will be kept fixed in all designs here so that the gains (or losses) come purely from head architecture changes.

## Concept

All previous ideas focused on: (a) how to fuse depth as a backbone input modality, (b) how the backbone is optimized (LLRD schedules), (c) positional encoding of depth, or (d) data augmentation. None of them has systematically varied the transformer **decoder head** itself — its depth (number of layers), width (hidden_dim), the initialization of joint queries, or the normalization applied before the final joint regression.

The head is only 5.8M parameters (small relative to the 293M backbone). This means:
- Head-only changes fit comfortably in the 1080ti 11GB budget.
- The search space is unexplored.
- Each variation trains at the same speed since the backbone forward pass dominates.

## Broader Reflection

### Strong results to build on

- **idea004/design002** (val_mpjpe_body = **112.3 mm**) is the overall best. The LLRD + progressive-unfreeze schedule is kept intact across all designs so its gains are baseline for this idea.
- **idea001/design003** (late cross-attention fusion) achieved 121.1 mm. The cross-attention architecture in the head is the mechanism through which joint queries attend to backbone features — refining it was never directly targeted.
- **idea005/design001** (depth-bucket PE) achieved 123.8 mm with the vanilla head. Head improvements should stack on top of depth-aware inputs since they address orthogonal aspects.

### Patterns to avoid

- **idea002** (kinematic attention masking) showed that modifying the self-attention mask among query tokens hurts more than it helps within 20 epochs. We should not revisit joint-level masking.
- **idea003** (curriculum loss) showed that dynamic loss weighting requires more than 20 epochs to stabilize. Avoid loss-side changes in this idea.
- **idea005/design002** and **idea005/design003** diverged significantly from the winning design, showing that large structural departures from established baselines carry high risk. Designs here should be incremental changes to the head only.
- OOM: The backbone occupies ~9 GB of VRAM at batch=4. Head changes must not significantly increase memory. Widening `hidden_dim` from 256 to 512 doubles head memory but the head is tiny relative to the backbone; a quick estimate shows this is safe. Do not add more than ~10M extra parameters to the head.

## Design Axes

### Category A — Exploit & Extend

**Axis A1: Decoder depth.** The baseline uses 4 decoder layers. We know from DETR-family literature that 6 layers often outperforms 4, but 8 can overfit in a proxy run. Directly test 6 layers to see if additional cross-attention refinement improves joint localization.

*Derives from:* idea004/design002 (best schedule baseline); idea001/design003 (showed cross-attention depth matters for modality fusion).

**Axis A2: Hidden dimension widening.** Baseline head_hidden=256. Widening to 384 gives the joint queries more representational capacity without exceeding memory limits. Keep layers=4.

*Derives from:* idea004/design002 (best completed result); architecture search principle that head width can bottleneck a powerful backbone.

### Category B — Novel Exploration

**Axis B1: Sine-cosine positional priors for joint queries.** Instead of random-initialized learnable embeddings (`nn.Embedding(70, 256)`), initialize each of the 70 joint query vectors using sine-cosine encodings derived from the joint index. This gives the decoder a structured starting point aligned with joint identity ordering rather than relying on random initialization to discover structure from scratch. The weights remain trainable; only the initialization changes.

**Axis B2: Shared input projection + per-layer feature gating.** The baseline uses a single `Linear(1024→256)` projection applied once before the decoder. Replace it with a gated projection where a learned scalar gate (sigmoid-activated) controls how much of the projected backbone feature passes into each cross-attention layer. This adds 4 scalar gates (one per decoder layer), making the decoder more robust to the depth-channel distributional shift.

**Axis B3: Layer normalization on head output before final linear.** Add a `LayerNorm(256)` immediately before the final `Linear(256→3)` regression layer. This stabilizes the activation scale entering the regressor and has been shown in transformer regression heads to reduce final-layer instability. Extremely cheap: no new parameters beyond the LN scale/bias.

## Expected Designs

The Designer should generate **5** novel designs:

1. **6-layer decoder** — head_num_layers=6, head_hidden=256, head_num_heads=8; everything else fixed to idea004/design002 schedule.
2. **Wide head** — head_num_layers=4, head_hidden=384, head_num_heads=8 (must remain divisible); update `input_proj` to Linear(1024→384) and query embed to Embedding(70, 384).
3. **Sine-cosine joint query init** — head_num_layers=4, head_hidden=256; replace `nn.Embedding` with fixed-init sinusoidal encodings for each joint index (0..69), weights remain learnable post-init.
4. **Per-layer input feature gate** — head_num_layers=4, head_hidden=256; add learned scalar gate before each decoder layer's cross-attention key/value projection; gates initialized to 1.0 so behavior at epoch 0 matches the baseline.
5. **Output LayerNorm** — head_num_layers=4, head_hidden=256; add `LayerNorm(256)` before the final `Linear(256→3)`; this is the simplest possible change and acts as a sanity-check lower bound for head refinement.

## Design Constraints

- Keep the LLRD schedule from idea004/design002 (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4).
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- All designs must use the standard 4-channel RGBD input with baseline depth normalization.
- For design 2 (wide head), verify `head_num_heads` divides `head_hidden` evenly: 384/8=48 — valid.
- Do not modify `infra.py` or the backbone architecture.
