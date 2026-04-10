# idea012 — Regularization for Generalization

**Expected Designs:** 5

## Starting Point

The baseline starting point for this idea is:

`runs/idea004/design002/train.py`

That design produced the best completed validation body MPJPE at **112.3 mm** using LLRD (gamma=0.90, unfreeze_epoch=5). It also exhibits a characteristic train-val gap: train_mpjpe_body = 83.7 mm vs val_mpjpe_body = 112.3 mm, a gap of ~29 mm. This idea systematically explores regularization techniques to close that gap.

## Concept

A persistent and large train-val MPJPE gap exists across **every** completed experiment in the pipeline:

| Idea | Best Design | Train Body | Val Body | Gap |
|------|------------|------------|----------|-----|
| idea001 | design003 | 91.3 | 121.1 | 29.8 |
| idea004 | design002 | 83.7 | 112.3 | 28.6 |
| idea008 | design003 | 98.3 | 121.0 | 22.7 |
| idea008 | design004 | 98.6 | 120.8 | 22.2 |

The gap ranges from 22-30 mm and has not been addressed directly. idea006 explored data augmentation but found that geometric transforms (especially horizontal flip) hurt pelvis localization. This idea takes the complementary approach: **model-side and optimization-side regularization** that does not alter the input data distribution.

No prior idea has varied dropout rates, stochastic depth, weight decay, or added explicit regularization terms to the training objective. These are standard and well-understood techniques that fit within the 20-epoch proxy budget.

## Broader Reflection

### Strong prior results to build on

- **idea004/design002** (val_mpjpe_body = **112.3 mm**) is the best body MPJPE baseline. Its LLRD schedule is well-understood and kept fixed here.
- **idea006** (augmentation) showed that input-side regularization is tricky — horizontal flip with joint swap degraded pelvis localization to 270 mm. The safest augmentation was scale/crop jitter (design002: 126.0 mm body), which is modest. Model-side regularization is the unexplored complementary path.
- **idea008/design003** showed that better positional encoding reduced the gap to ~23 mm — depth PE helps generalization. But a 23 mm gap still indicates overfitting.

### Patterns to avoid

- **idea006/design001** and **design005/006** showed that flip augmentation is harmful for pelvis. Do not add data augmentation in this idea.
- **idea003** (curriculum loss) showed that dynamic loss manipulation needs more than 20 epochs. Avoid complex loss scheduling.
- Large architectural changes (idea002, idea005/design002-003) tend to underperform relative to incremental changes. Keep the model architecture identical to idea004/design002 and only vary regularization hyperparameters.

## Design Axes

### Category A — Exploit & Extend

**Axis A1: Increased head dropout.**
The baseline uses `head_dropout=0.1` in the transformer decoder layers. Increasing dropout is the simplest regularization change. Test `head_dropout=0.2` to reduce overfitting in the small decoder head. The LLRD schedule from idea004/design002 is kept fixed.

*Derives from:* idea004/design002 (best body MPJPE baseline), varying the only existing regularization knob that has never been tuned.

**Axis A2: Increased weight decay.**
The baseline uses `weight_decay=0.1` (in the original baseline) or `weight_decay=0.03` (in idea008). idea004/design002 uses the default 0.1. Increasing to `weight_decay=0.3` adds stronger L2 regularization to all parameters, which is a standard approach to reducing overfitting in fine-tuning scenarios.

*Derives from:* idea004/design002, scaling the existing weight decay parameter more aggressively.

### Category B — Novel Exploration

**Axis B1: Stochastic depth (drop path) increase.**
The baseline uses `drop_path_rate=0.1` applied to the ViT backbone. Increasing stochastic depth to `0.2` randomly drops entire transformer blocks during training, forcing the model to be robust to missing intermediate representations. This is a well-established regularizer for ViTs (Huang et al., 2016) but has never been varied in this pipeline.

**Axis B2: R-Drop consistency regularization.**
Add a KL-divergence consistency loss between two forward passes of the same input through the model (with different dropout masks). The total loss becomes `L_task + alpha * KL(p1 || p2)`. For regression, this is implemented as an MSE penalty between two stochastic forward pass outputs: `alpha * MSE(pred1, pred2)` with `alpha=1.0`. This requires two forward passes per step but at batch_size=4 the memory overhead is manageable (store two sets of predictions, not two full forward graphs — use `torch.no_grad()` on the second pass and detach).

**Axis B3: Combined dropout + weight decay + drop path.**
Apply all three orthogonal regularization knobs simultaneously: `head_dropout=0.2`, `weight_decay=0.2`, `drop_path_rate=0.2`. This tests whether the regularization effects stack beneficially or interfere. It is the full-strength regularization cocktail.

## Expected Designs

The Designer should generate **5** novel designs:

1. **Head dropout 0.2** — Change `head_dropout` from 0.1 to 0.2. Everything else identical to idea004/design002.
2. **Weight decay 0.3** — Change `weight_decay` from 0.1 to 0.3 for all optimizer param groups. Everything else identical.
3. **Stochastic depth 0.2** — Change `drop_path_rate` from 0.1 to 0.2 in the backbone. Everything else identical.
4. **R-Drop consistency** — Add MSE consistency loss between two stochastic forward passes with `alpha=1.0`. Second forward pass uses `torch.no_grad()` and is detached. Compute `consistency = MSE(pred1_joints[:, BODY_IDX], pred2_joints[:, BODY_IDX])`. Total loss = `L_task + alpha * consistency`. All other hyperparameters identical to idea004/design002.
5. **Combined regularization** — `head_dropout=0.2`, `weight_decay=0.2`, `drop_path_rate=0.2` applied simultaneously. No R-Drop (to isolate the effect of the three simpler knobs).

## Design Constraints

- Keep the LLRD schedule from idea004/design002 (gamma=0.90, unfreeze_epoch=5, base_lr_backbone=1e-4, lr_head=1e-4) fixed across all designs.
- `BATCH_SIZE=4`, `ACCUM_STEPS=8` fixed (infra.py).
- `epochs=20`, `warmup_epochs=3` fixed.
- `grad_clip=1.0`.
- `lambda_depth=0.1`, `lambda_uv=0.2`.
- For design 4 (R-Drop): the second forward pass must use `torch.no_grad()` and `.detach()` to avoid doubling the backward graph memory. The consistency loss is computed only on body joint predictions, not pelvis depth/uv.
- For design 4: two forward passes per training step will roughly double iteration time. This is acceptable since the 20-epoch budget is wall-time-flexible.
- Do not modify `infra.py`, transforms, or the model architecture (except dropout/drop_path rates which are constructor arguments).
- Do not add any data augmentation. This idea focuses purely on model-side and optimization-side regularization.
