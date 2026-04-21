# Detailed Experiments Guide

This file expands the short summary in `experiments.md` into a per-idea narrative. It is intended as a readable map of what each experiment family tried, why it was introduced, and what each concrete design changed.

Metric shorthand: `W` = validation weighted MPJPE in millimeters, and `@eN` means the metric was recorded at epoch `N`. Partial runs keep their last recorded epoch instead of being treated as full 20-epoch results.

## idea001 - RGB-D Modality Fusion Strategy

The Sapiens baseline uses 'early fusion' by modifying the first patch embedding layer for 4-channel input. This potentially disrupts learned RGB representations. We want to test different fusion points.

### design001

- Name: Design 001 - Early 4-Channel Fusion (early_4ch)
- Result: <span><strong>W 142.5 @e20 (baseline)</strong></span>
- Explanation: Keep the model architecture exactly as in baseline.py. The depth map is concatenated with the RGB tensor before the patch embedding (i.e., x = torch.cat([rgb, depth], dim=1) → shape (B, 4, 640, 384)). The 4th channel's patch embedding weights are initialized from the mean of the 3 RGB channel weights in the pretrained checkpoint. All ViT layers see a fully fused 4-channel representation from the very first token projection.

### design002

- Name: Design 002 - Mid-Layer Depth Fusion (mid_fusion)
- Result: <span style="color: green;">W 139.6 @e20</span>
- Explanation: Process only the 3-channel RGB image through the standard (unchanged) patch embedding so the first 12 transformer blocks run on clean RGB tokens. After block 11 (zero-indexed, i.e., the 12th block), add a learned per-spatial-position depth bias to the token sequence. The depth bias is produced by a lightweight depth projection network (DepthProjector) that maps the raw depth map to a (B, N_tokens, embed_dim) additive signal, which is added to the RGB token sequence before block 12 runs.

### design003

- Name: Design 003 - Late Cross-Attention Depth Fusion (late_cross_attention)
- Result: <span style="color: red;">W 142.8 @e20</span>
- Explanation: Run the full 24-block Sapiens ViT on RGB only (3-channel input, no modification). After the ViT produces its spatial feature map (B, 1024, 40, 24), flatten it to a token sequence (B, 960, 1024). In parallel, encode the raw depth map through a lightweight DepthTokenizer to produce a depth token sequence of the same spatial resolution: (B, 960, 64). A single-layer DepthCrossAttention transformer block then refines the RGB token sequence using the depth tokens: the RGB tokens act as queries, the depth tokens act as keys and values. The resulting updated RGB token sequence is reshaped back to (B, 1024, 40, 24) and passed to the existing Pose3DHead unchanged.

## idea002 - Kinematic Attention Masking

Human pose estimation models often predict physically impossible poses because the attention mechanism lacks explicit structural constraints. Kinematic Attention Masking restricts self-attention weights between structurally distant joints in the final decoding layers, enforcing anatomical plausibility.

### design001

- Name: Kinematic Attention Masking - Design 1
- Result: <span><strong>W 142.5 @e20 (baseline)</strong></span>
- Explanation: This is the unconstrained dense-attention control run. It uses the exact same Pose3DHead architecture as baseline.py with no kinematic masking applied. It establishes a reference MPJPE to compare against Designs 2 and 3.

### design002

- Name: Kinematic Attention Masking - Design 2
- Result: <span style="color: red;">W 144.5 @e20</span>
- Explanation: This design applies a soft additive bias to the self-attention logits of the joint query decoder, penalizing attention between structurally distant joints. Joints that are close in the kinematic tree (low hop count) receive little or no penalty; joints that are far apart receive a larger negative bias. The bias is applied at every decoder layer.

### design003

- Name: Kinematic Attention Masking - Design 3
- Result: <span style="color: green;">W 139.5 @e20</span>
- Explanation: This design applies a hard binary mask to the self-attention logits of the joint query decoder. Joints within a 2-hop neighborhood in the kinematic tree (including self) are allowed to attend freely (bias = 0.0); joints outside this neighborhood are completely blocked (bias = -inf). The mask is precomputed once, registered as a buffer, and applied at every decoder layer.

## idea003 - Curriculum-Based Loss Weighting

Absolute depth is hard and creates noisy gradients early on. It needs a learned or structured loss weighting over epochs to stabilize relative pose training.

### design001

- Name: Design 001: Homoscedastic Uncertainty Loss Weighting (Learnable uncertainty weight)
- Result: <span style="color: red;">W 163.0 @e20</span>
- Explanation: Apply homoscedastic task-uncertainty weighting so the model learns how strongly to weight pose, pelvis-depth, and UV supervision instead of keeping those weights fixed by hand. In practice this adds learnable log-variance terms that rescale each task loss during training.

### design002

- Name: Design 002: Linear Warmup for Depth Loss
- Result: <span style="color: red;">W 143.2 @e20</span>
- Explanation: Implement a linear warmup schedule for lambda_depth only. The UV loss weight remains fixed throughout training (matching the baseline).

## idea004 - Layer-Wise Learning Rate Decay & Progressive Unfreezing

Use structure in how the backbone learns to prevent catastrophic forgetting of the ViT's rich pre-trained RGB features while adapting it to depth. Give different learning rates to different depths.

### design001

- Name: Design 001 - Constant Decay LLRD (gamma=0.95, unfreeze_epoch=5)
- Result: <span style="color: green;">W 131.9 @e20</span>
- Explanation: This design introduces Layer-Wise Learning Rate Decay (LLRD) to the Sapiens ViT-B backbone. Instead of a single flat learning rate for the entire backbone (as in the baseline), each transformer block receives a distinct learning rate that decays exponentially from the deepest (highest-index) block to the shallowest (index 0). The head retains its own independent learning rate. Progressive unfreezing freezes the lower (shallower) backbone blocks at the start of training and unfreezes them at a specified epoch.

### design002

- Name: Design 002 - Constant Decay LLRD (gamma=0.90, unfreeze_epoch=5)
- Result: <span style="color: green;">W 130.7 @e20</span>
- Explanation: This design applies Layer-Wise Learning Rate Decay (LLRD) to the Sapiens ViT-B backbone with a steeper decay factor gamma=0.90 and progressive unfreezing at epoch 5. Compared to design001 (gamma=0.95), this variant creates a stronger gradient between deep and shallow layer learning rates (~8x ratio vs ~3x), enforcing stronger preservation of shallow pre-trained features.

### design003

- Name: Design 003 - Constant Decay LLRD (gamma=0.85, unfreeze_epoch=5)
- Result: <span style="color: green;">W 131.7 @e20</span>
- Explanation: This design applies Layer-Wise Learning Rate Decay (LLRD) to the Sapiens ViT-B backbone with the steepest decay factor in the unfreeze_epoch=5 group: gamma=0.85. Progressive unfreezing occurs at epoch 5, same as designs 001 and 002. This variant creates the strongest LR gradient (~21x ratio from deepest to shallowest block) among the early-unfreeze designs, maximally suppressing updates to shallow pre-trained features.

### design004

- Name: Design 004 - Constant Decay LLRD (gamma=0.95, unfreeze_epoch=10)
- Result: <span style="color: green;">W 132.3 @e20</span>
- Explanation: This design applies Layer-Wise Learning Rate Decay (LLRD) to the Sapiens ViT-B backbone with gamma=0.95 and a later progressive unfreezing at epoch 10. Compared to design001 (same gamma, unfreeze at epoch 5), the frozen shallow blocks are held frozen for twice as long, giving the head and deep backbone blocks a full 10 warm-up epochs to stabilize before shallow layers are unlocked. This tests whether later unfreezing leads to better final performance by reducing interference during the critical early adaptation phase.

### design005

- Name: Design 005 - Constant Decay LLRD (gamma=0.90, unfreeze_epoch=10)
- Result: <span style="color: green;">W 131.6 @e20</span>
- Explanation: This design applies Layer-Wise Learning Rate Decay (LLRD) to the Sapiens ViT-B backbone with gamma=0.90 and progressive unfreezing at epoch 10. It is the intersection of the steeper decay axis (gamma=0.90 from design002) and the later unfreeze axis (epoch 10 from design004). This combination maximally delays shallow layer adaptation while also applying a stronger LR gradient across blocks, testing whether their combined effect produces better retention of pre-trained features than either factor alone.

### design006

- Name: Design 006 - Constant Decay LLRD (gamma=0.85, unfreeze_epoch=10)
- Result: <span style="color: green;">W 131.9 @e20</span>
- Explanation: This design applies Layer-Wise Learning Rate Decay (LLRD) to the Sapiens ViT-B backbone with the steepest decay factor in the grid, gamma=0.85, and the latest unfreezing point, unfreeze_epoch=10. It tests the most conservative shallow-layer update schedule in idea004.

## idea005 - Depth-Aware Positional Embeddings

Provide explicit geometric awareness to the Transformer, which currently only uses standard 2D positional encodings. Inject depth values or structure into the positional embeddings.

### design001

- Name: Design 001 - discretized_depth_pe
- Result: <span style="color: green;">W 121.4 @e20</span>
- Explanation: Module: DepthBucketPE

### design002

- Name: Design 002 - relative_depth_bias
- Result: <span style="color: green;">W 142.1 @e20</span>
- Explanation: Module: DepthAttentionBias

### design003

- Name: Design 003 - depth_conditioned_pe
- Result: <span style="color: red;">W 146.1 @e20</span>
- Explanation: Module: DepthConditionedPE

@Todo: use min pool instead of avg_pool2d when obtain depth

## idea006 - Training Data Augmentation for Generalization

This idea targets the persistent train/validation gap seen across earlier runs. The baseline used effectively no train-time augmentation because the training transform matched validation, so this experiment family tests whether geometric, photometric, and depth-specific perturbations can improve generalization.

### design001

- Name: Horizontal Flip Augmentation
- Result: <span style="color: red;">W 175.5 @e20</span>
- Explanation: Add a RandomHorizontalFlip augmentation (p=0.5) into build_train_transform() only. Horizontal flipping doubles the effective viewpoint diversity at near-zero computational cost. Joint coordinates, pelvis UV, and left-right joint assignments must all be updated consistently with the image flip.

### design002

- Name: Scale/Crop Jitter Augmentation
- Result: <span style="color: red;">W 142.8 @e20</span>
- Explanation: Add a RandomScaleJitter augmentation that randomly scales the bounding box by a factor sampled from Uniform(0.8, 1.2) before CropPerson. This makes the person appear at varying scales within the fixed output crop, teaching the model to be scale-invariant. Joint root-relative coordinates (3D camera-space metric values) do not require adjustment since they are independent of pixel scale.

### design003

- Name: Color Jitter Augmentation (RGB Only)
- Result: <span style="color: green;">W 140.9 @e20</span>
- Explanation: Add torchvision.transforms.ColorJitter to build_train_transform() only, applied to the RGB channels after ToTensor. The depth channel is left entirely unchanged. Color jitter randomly perturbs brightness, contrast, saturation, and hue, making the backbone features invariant to photometric variation - which is irrelevant to 3D pose geometry. No joint coordinates or intrinsics are affected.

### design004

- Name: Depth Channel Augmentation (Gaussian Noise + Pixel Dropout)
- Result: <span style="color: red;">W 143.1 @e20</span>
- Explanation: Add two stochastic depth-channel perturbations to build_train_transform() only, applied after ToTensor on the normalized depth tensor

### design005

- Name: Combined Geometric Augmentation (Horizontal Flip + Scale Jitter)
- Result: <span style="color: red;">W 176.4 @e20</span>
- Explanation: Apply both RandomScaleJitter (±20%, before CropPerson) and RandomHorizontalFlip (p=0.5, after CropPerson, before SubtractRoot) in build_train_transform(). The ordering is critical: scale jitter must precede CropPerson (it modifies the bbox), and flip must follow CropPerson (it operates on the already-cropped image array) but precede SubtractRoot (so SubtractRoot sees the flipped coordinate system and computes pelvis_uv correctly from the negated-Y joints).

### design006

- Name: Full Augmentation Stack (Horizontal Flip + Color Jitter + Depth Noise)
- Result: <span style="color: red;">W 170.8 @e20</span>
- Explanation: Add RandomHorizontalFlip, RGBColorJitter, and DepthAugmentation to build_train_transform() only. Each augmentation is applied independently at its appropriate stage in the pipeline

## idea007 - Depth-Bucket Positional Embeddings with Layer-Wise Fine-Tuning

Exploit the best completed architecture from idea005/design001 and combine it with the best completed backbone adaptation strategy from idea004: layer-wise learning-rate decay (LLRD) and progressive unfreezing. The central question is whether the explicit depth-aware positional signal benefits even more when shallow pretrained ViT layers are protected early in training.

### design001

- Name: Design 001 - Gentle LLRD on Depth-Bucket PE (gamma=0.95, unfreeze_epoch=5)
- Result: <span style="color: green;">W 132.6 @e20</span>
- Explanation: Keep the depth-bucket positional embedding architecture unchanged from runs/idea005/design001, but replace the optimizer schedule with layer-wise learning-rate decay (LLRD) and progressive unfreezing.

### design002

- Name: Design 002 - Strong LLRD on Depth-Bucket PE (gamma=0.90, unfreeze_epoch=5)
- Result: <span style="color: green;">W 135.0 @e20</span>
- Explanation: Use the same architecture as runs/idea005/design001, but adopt the stronger LLRD schedule from idea004/design002.

### design003

- Name: Design 003 - Strong LLRD with Earlier Unfreeze (gamma=0.90, unfreeze_epoch=3)
- Result: <span style="color: green;">W 134.9 @e20</span>
- Explanation: Keep the architecture from runs/idea005/design001 unchanged and reuse the strong LLRD schedule from Design 002, but move the full-backbone unfreeze earlier.

## idea008 - Continuous Depth Positional Encoding

The completed experiments suggest that explicit depth-aware positional structure is the strongest architectural improvement discovered so far, while more disruptive attention or curriculum changes did not help. The next promising step is to make the depth positional signal smoother and better calibrated rather than replacing it outright.

### design001

- Name: Design 001 - Continuous Interpolated Depth Positional Encoding
- Result: <span style="color: green;">W 116.4 @e20</span>
- Explanation: Keep the successful row + column decomposition from runs/idea005/design001, but replace the hard 16-bin depth lookup with continuous linear interpolation between neighboring depth embeddings. The model still learns 16 depth-anchor embeddings, but each patch receives a weighted mixture of the two nearest anchors instead of a one-hot bucket.

### design002

- Name: Design 002 - Interpolated Depth PE with Residual Gate
- Result: <span style="color: green;">W 118.8 @e20</span>
- Explanation: Start from the same successful row + column + depth positional decomposition as runs/idea005/design001, replace the hard depth bucket lookup with the same continuous interpolation scheme used in idea008/design001, and add a lightweight learned residual gate on the depth term.

### design003

- Name: Design 003 - Interpolated Depth PE with Near-Emphasized Spacing
- Result: <span style="color: green;">W 112.0 @e20</span>
- Explanation: Keep the successful row + column decomposition from runs/idea005/design001 and the continuous interpolation strategy from idea008/design001, but change the anchor spacing so more effective resolution is allocated to nearer depths.

### design004

- Name: Design 004 - Hybrid Two-Resolution Depth PE
- Result: <span style="color: green;">W 112.1 @e20</span>
- Explanation: Keep the successful row + column positional decomposition from runs/idea005/design001, replace the hard local bucket lookup with the continuous interpolation scheme from idea008/design001, and add a second lightweight coarse global depth positional term shared across all tokens in an image.

## idea009 - Head Architecture Refinement

All previous ideas focused on: (a) how to fuse depth as a backbone input modality, (b) how the backbone is optimized (LLRD schedules), (c) positional encoding of depth, or (d) data augmentation. None of them has systematically varied the transformer decoder head itself - its depth (number of layers), width (hidden_dim), the initialization of joint queries, or the normalization applied before the final joint regression.

### design001

- Name: Design 001 - 6-Layer Decoder Head
- Result: <span style="color: green;">W 131.4 @e20</span>
- Explanation: The baseline Pose3DHead uses 4 transformer decoder layers. In DETR-family models, additional cross-attention layers allow the joint queries to iteratively refine their positions against the backbone feature map. The baseline has never been tested beyond 4 layers. There is headroom to add 2 more decoder layers cheaply since the head is only ~5.8M parameters - two extra layers add roughly 2.9M parameters, well within the VRAM budget.

### design002

- Name: Design 002 - Wide Head (hidden_dim=384)
- Result: <span style="color: green;">W 130.4 @e20</span>
- Explanation: The baseline Pose3DHead uses hidden_dim=256 for all internal representations: the input projection, joint query embeddings, and the transformer decoder's d_model. With a 293M-parameter backbone producing 1024-dim feature tokens, the 256-dim bottleneck may limit how much discriminative information the joint queries can extract per cross-attention step. Widening to 384 gives each query token 50% more capacity per dimension, potentially improving localization accuracy without adding decoder depth.

### design003

- Name: Design 003 - Sine-Cosine Joint Query Initialization
- Result: <span style="color: green;">W 132.2 @e20</span>
- Explanation: The baseline Pose3DHead initializes its 70 joint query embeddings with nn.Embedding(70, 256) and applies trunc_normal_(std=0.02) - a near-zero random initialization. This gives the decoder no structural information about the joint ordering at the start of training. Each query must learn from scratch that it corresponds to a particular anatomical location, with no inductive bias toward the sequential structure of the joint index. The queries are fully learnable after initialization, so any starting point is valid, but a structured starting point could accelerate early convergence and reduce the variance of the final solution.

### design004

- Name: Design 004 - Per-Layer Input Feature Gate
- Result: <span style="color: green;">W 131.6 @e20</span>
- Explanation: The baseline Pose3DHead computes a single projected memory tensor

### design005

- Name: Design 005 - Output LayerNorm Before Final Regression
- Result: <span style="color: green;">W 133.0 @e20</span>
- Explanation: The baseline Pose3DHead.forward passes the transformer decoder's output tensor out (shape B x J x 256) directly into the final linear heads (joints_out, depth_out, uv_out) without any normalization. The final decoder layer (with norm_first=True / pre-norm) applies layer normalization *before* its self-attention and cross-attention sub-layers, but the residual outputs that emerge from the decoder's final layer are not re-normalized before regression. This means the activation scale entering the linear regressors can vary across training steps, particularly in the early warm-up epochs when the head's weights change rapidly. Unstable input scales to the output linear layers translate to unstable gradient magnitudes for those layers and can delay convergence.

## idea010 - Multi-Scale Backbone Feature Aggregation

Every prior idea has consumed only the final ViT layer output (B, 1024, 40, 24) as input to the decoder head. In ViT architectures, intermediate layers capture qualitatively different information: early layers encode local spatial/texture patterns (useful for precise joint localization), middle layers encode part-level structure, and late layers encode global semantic context. By aggregating features from multiple ViT layers before feeding them to the cross-attention decoder, the head gains access to a richer, multi-resolution representation without any additional backbone parameters.

### design001

- Name: - Last-4-Layer Concatenation + Linear Projection
- Result: <span style="color: green;">W 128.9 @e20</span>
- Explanation: Extract the outputs of the last 4 ViT transformer blocks (0-indexed layers 20, 21, 22, 23 out of the 24-block Sapiens 0.3B), concatenate along the channel dimension to form a (B, 4096, 40, 24) tensor, then project back to (B, 1024, 40, 24) via a single linear layer. This is the simplest multi-scale aggregation baseline: it gives the decoder head access to features from multiple depths of the backbone without changing the head architecture at all.

### design002

- Name: - Learned Layer Weights (Softmax-Weighted Sum)
- Result: <span style="color: green;">W 132.3 @e20</span>
- Explanation: Extract the outputs of the last 4 ViT transformer blocks (0-indexed layers 20, 21, 22, 23) and compute a learned weighted average: output = sum(softmax(w)[i] * layer_i) where w is a vector of 4 learnable scalars initialized to equal values (so initial weights are uniform 0.25 each). This preserves the (B, 1024, 40, 24) shape identically to the baseline -- no projection layer needed, only 4 new scalar parameters. Inspired by ELMo-style layer mixing.

### design003

- Name: - Feature Pyramid with 3 Scales
- Result: <span style="color: green;">W 130.0 @e20</span>
- Explanation: Extract features from 3 evenly-spaced ViT layers spanning the full backbone depth: layer 7 (early), layer 15 (middle), and layer 23 (final) (0-indexed, out of 24 blocks). Project each from 1024 to 256 channels via separate linear layers, concatenate to get (B, 768, 40, 24), then project to (B, 1024, 40, 24) via a final linear. This gives the head a true multi-scale feature pyramid combining low-level spatial detail, mid-level part structure, and high-level semantics.

### design004

- Name: - Cross-Scale Attention Gate
- Result: <span style="color: green;">W 130.0 @e20</span>
- Explanation: Extract features from two ViT layers: layer 11 (mid-level, roughly halfway through the 24-block backbone) and layer 23 (final). Compute a spatial attention gate from the mid-layer features using a single linear projection to a scalar, apply sigmoid, and use it to modulate the final-layer features via a residual multiplicative gate: output = layer_23 * (1 + gate). The gate learns to up-weight spatial locations where mid-level features indicate important local structure (e.g., limb boundaries, joint neighborhoods).

### design005

- Name: - Alternating Layer Average
- Result: <span style="color: green;">W 138.5 @e20</span>
- Explanation: Extract features from all even-indexed ViT blocks (0-indexed: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 -- i.e., every other block, 12 total) and compute a simple channel-wise mean. This uniformly samples the full depth of the backbone with no additional parameters. The head receives a smoothed representation that combines low-level spatial detail with high-level semantics.

## idea011 - LLRD with Continuous Depth PE

The two strongest completed results in the pipeline target orthogonal aspects of the model

### design001

- Name: Design 001 - LLRD (gamma=0.90, unfreeze=5) + Sqrt-Spaced Continuous Depth PE
- Result: <span style="color: green;">W 104.7 @e20</span>
- Explanation: Take the architecture from idea008/design003 (row + column + continuous interpolated depth PE with sqrt anchor spacing) and add the LLRD optimization schedule from idea004/design002 (gamma=0.90, progressive unfreezing at epoch 5). The depth PE architecture is not modified at all -- only the optimizer is changed to use per-block learning rates and progressive unfreezing.

### design002

- Name: Design 002 - LLRD (gamma=0.85, unfreeze=5) + Sqrt-Spaced Continuous Depth PE
- Result: <span style="color: green;">W 107.0 @e20</span>
- Explanation: Take the architecture from idea008/design003 (row + column + continuous interpolated depth PE with sqrt anchor spacing) and add an LLRD optimization schedule with gamma=0.85 and progressive unfreezing at epoch 5. This is identical to design001 except for the steeper decay factor.

### design003

- Name: Design 003 - LLRD (gamma=0.90, unfreeze=10) + Sqrt-Spaced Continuous Depth PE
- Result: <span style="color: green;">W 109.5 @e20</span>
- Explanation: Take the architecture from idea008/design003 (row + column + continuous interpolated depth PE with sqrt anchor spacing) and add an LLRD optimization schedule with gamma=0.90 and progressive unfreezing delayed to epoch 10. During the longer frozen phase, only the upper 12 backbone blocks, depth PE parameters, and head are trained; the lower 12 blocks and embeddings remain frozen for half the total training duration.

### design004

- Name: Design 004 - LLRD (gamma=0.90, unfreeze=5) + Gated Continuous Depth PE
- Result: <span style="color: green;">W 110.3 @e20</span>
- Explanation: Take the architecture from idea008/design002 (row + column + gated continuous interpolated depth PE with linear anchor spacing and scalar sigmoid gate) and add the LLRD optimization schedule from idea004/design002 (gamma=0.90, progressive unfreezing at epoch 5). The depth PE architecture including the learned gate is not modified -- only the optimizer is changed.

## idea012 - Regularization for Generalization

A persistent and large train-val MPJPE gap exists across every completed experiment in the pipeline

### design001

- Name: Design 001 - Head Dropout 0.2
- Result: <span style="color: green;">W 132.8 @e20</span>
- Explanation: Increase the transformer decoder head dropout from 0.1 to 0.2. This is the simplest single-knob regularization change targeting the decoder head where overfitting is most likely concentrated. All other hyperparameters remain identical to idea004/design002.

### design002

- Name: Design 002 - Weight Decay 0.3
- Result: <span style="color: green;">W 131.3 @e20</span>
- Explanation: Increase weight decay from 0.03 (idea004/design002 value) to 0.3 for all optimizer parameter groups. This is a 10x increase in L2 regularization strength, testing whether stronger weight penalty reduces the train-val gap. All other hyperparameters remain identical.

### design003

- Name: Design 003 - Stochastic Depth 0.2
- Result: <span style="color: green;">W 136.7 @e20</span>
- Explanation: Increase the stochastic depth (drop path) rate in the ViT backbone from 0.1 to 0.2. This randomly drops entire transformer blocks during training, forcing the network to be robust to missing intermediate representations. All other hyperparameters remain identical.

### design004

- Name: Design 004 - R-Drop Consistency Regularization
- Result: <span style="color: green;">W 132.5 @e20</span>
- Explanation: Add an R-Drop-style consistency regularization loss that penalizes the difference between two stochastic forward passes of the same input. For regression, this is implemented as MSE between two sets of body joint predictions obtained with different dropout/drop-path masks. The total training loss becomes L_task + alpha * MSE(pred1_body, pred2_body) with alpha=1.0. All other hyperparameters remain identical to idea004/design002.

### design005

- Name: Design 005 - Combined Regularization (Dropout + Weight Decay + Drop Path)
- Result: <span style="color: green;">W 137.7 @e20</span>
- Explanation: Apply three orthogonal regularization knobs simultaneously: head dropout 0.2, weight decay 0.2, and stochastic depth (drop path) 0.2. This tests whether combining multiple moderate regularizers produces a better result than any single strong regularizer. No R-Drop is included, to isolate the effect of these three simpler knobs.

## idea013 - Joint Prediction Loss Reformulation

Every experiment in the pipeline uses the same loss function: Smooth L1 (beta=0.05) applied independently to each body joint's 3D coordinates. This formulation treats all joints equally, ignores inter-joint relationships, and uses a fixed robustness threshold. No prior idea has varied the loss function itself (idea003 varied loss *weights* between tasks, not the loss *formulation* for pose).

### design001

- Name: Design 001 - Small-Beta Smooth L1 (beta=0.01)
- Result: <span style="color: green;">W 130.6 @e20</span>
- Explanation: Reduce the Smooth L1 loss beta from the baseline 0.05 m (50 mm) to 0.01 m (10 mm). This makes the loss behave more like L1 for errors in the 10-50 mm range, which are currently treated quadratically. The stronger constant gradient for medium-sized errors may help the model push past plateaus where per-joint errors cluster around 30-50 mm.

### design002

- Name: Design 002 - Large-Beta Smooth L1 (beta=0.1)
- Result: <span style="color: green;">W 134.0 @e20</span>
- Explanation: Increase the Smooth L1 loss beta from 0.05 m (50 mm) to 0.1 m (100 mm). This makes the loss more L2-like for a wider range of errors, giving stronger gradients for large errors (50-100 mm range) which are now in the quadratic regime rather than the linear one. The quadratic regime also naturally down-weights outlier frames where a joint has an unusually large error, reducing gradient variance.

### design003

- Name: Design 003 - Bone-Length Auxiliary Loss
- Result: <span style="color: green;">W 130.9 @e20</span>
- Explanation: Add a soft bone-length consistency penalty alongside the standard Smooth L1 pose loss. For each anatomical bone (edge in the body skeleton), compute the L2 distance between connected joints for both prediction and ground truth, then **penalize the absolute difference in bone lengths**. This encourages anatomically plausible poses without modifying the model architecture.

### design004

- Name: Design 004 - Hard-Joint-Weighted Loss
- Result: <span style="color: green;">W 131.0 @e20</span>
- Explanation: After epoch 0, compute per-joint mean training L1 error across all body joints (0-21). Derive fixed per-joint weights inversely proportional to accuracy: **joints with higher error get more loss weight**. Apply these fixed weights element-wise to the per-joint Smooth L1 loss for epochs 1-19. This is a one-shot reweighting that allocates more gradient signal to the hardest joints (typically extremities like wrists and ankles) without the instability of dynamic per-epoch scheduling.

## idea014 - Best-of-Breed Combination: LLRD + Depth PE + Wide Head

The pipeline has now identified three independently strong improvements that target non-overlapping components of the model

### design001

- Name: Design 001 - Depth PE + Wide Head (No LLRD)
- Result: <span style="color: green;">W 121.5 @e20</span>
- Explanation: Replace the standard head (hidden_dim=256) in idea008/design003 with the wide head from idea009/design002 (hidden_dim=384, num_heads=8, num_layers=4). Keep the continuous depth PE with sqrt spacing and the flat optimizer (lr_backbone=1e-5, lr_head=1e-4, lr_depth_pe=1e-4) unchanged.

### design002

- Name: Design 002 - LLRD + Depth PE + Wide Head (Triple Combination)
- Result: <span style="color: green;">W 104.4 @e20</span>
- Explanation: Apply both the LLRD schedule and the wide head to the idea008/design003 codebase

### design003

- Name: Design 003 - LLRD + Depth PE + Wide Head + Weight Decay 0.3
- Result: <span style="color: green;">W 103.5 @e20</span>
- Explanation: Identical to design002 (LLRD + depth PE + wide head) with a single change: weight_decay=0.3 (10x the baseline 0.03) applied to all optimizer param groups.

## idea015 - Iterative Refinement Decoder

No prior idea has varied the *prediction protocol* of the head. All prior designs run a single forward pass of the TransformerDecoder over 70 joint queries and emit a single 3D joint prediction. In pose estimation (see DETR-style pose heads, PRTR, PoseFormer), **iterative refinement of query features - re-feeding a coarse prediction back into the query embedding - is a standard and well-motivated way to improve localization without adding width or loss terms.**

### design001

- Name: Two-Pass Shared-Decoder Refinement (Query Injection)
- Result: <span style="color: green;">W 101.9 @e20</span>
- Explanation: Two-pass iterative refinement with query injection (shared decoder weights). **Take output 3*70 to refine_mlp and output a small refinement. Then do again.**

### design002

- Name: Two-Pass Shared-Decoder Refinement (Cross-Attention Gaussian Bias from J1)
- Result: <span style="color: red;">W 143.7 @e6</span>
- Explanation: Two-pass shared-decoder refinement with Gaussian cross-attention bias (memory-gated attention).

### design003

- Name: Three-Pass Shared-Decoder Refinement (Deep Supervision 0.25/0.5/1.0)
- Result: <span style="color: red;">W 229.6 @e5</span>
- Explanation: Three-pass iterative refinement with shared decoder and query injection (progressive deep supervision).

### design004

- Name: Two-Pass Two-Decoder Refinement (Independent 2-Layer Refine Decoder)
- Result: <span style="color: green;">W 98.9 @e20</span>
- Explanation: Two-pass refinement with a separate independent 2-layer decoder.

## idea016 - 2.5D Heatmap Soft-Argmax

Every prior idea in this pipeline (idea001-014) uses the same output protocol: the decoder produces 70 hidden vectors (B, 70, 256 or 384) and a final Linear(hidden->3) directly regresses (x, y, z) for each joint. No idea has varied the output representation.

### design001

- Name: 2D Heatmap + Scalar Depth (40x24 native grid)
- Result: <span style="color: red;">W 180.1 @e20</span>
- Explanation: Replace the final joints_out = Linear(384, 3) in Pose3DHead with a two-branch output

### design002

- Name: 2D Heatmap + Scalar Depth (80x48 upsampled grid)
- Result: <span style="color: red;">W 178.7 @e20</span>
- Explanation: Same two-branch heatmap architecture as design001, but the 40x24 spatial heatmap is bilinearly upsampled to 80x48 before softmax + soft-argmax. This tests whether the coarser native grid (40x24) limits sub-pixel localization accuracy, at the cost of a 4x larger softmax (3840 bins vs 960).

### design003

- Name: Full 3D Volumetric Heatmap (40x24x16, Integral Pose Regression)
- Result: <span style="color: red;">W 335.7 @e6</span>
- Explanation: Replace the final output with a full 3D volumetric heatmap: Linear(384, 40*24*16=15360) → reshape (B, 70, 40, 24, 16) → softmax over the full 3D volume → soft-argmax on all three axes simultaneously. The 16-bin depth axis reuses the same sqrt-spaced depth bin centres already defined in DepthBucketPE (coherent with num_depth_bins=16). This is the canonical Integral Pose Regression (Sun et al. 2018) applied to the 3D case.

### design004

- Name: 2D Heatmap + Scalar Depth + Auxiliary Gaussian MSE Supervision
- Result: <span style="color: red;">W 179.4 @e20</span>
- Explanation: Same as design001 (2D heatmap + scalar depth at native 40x24 resolution) with one addition: an auxiliary MSE loss on the predicted heatmap against a Gaussian target centred at the ground-truth (u, v) for each joint. Weight lambda_hm = 0.1. This provides dense spatial supervision during early epochs - a known technique for stabilising soft-argmax training when the coordinate loss alone may not generate enough gradient to shape the heatmap.

## idea017 - Temporal Adjacent-Frame Fusion

A fundamental limitation of the current pipeline: every prior experiment (idea001-014) processes a single frame in isolation, even though the dataset is sequential video. Temporal smoothness of pose is one of the strongest priors in 3D human pose estimation - methods like VideoPose3D, PoseFormer, MotionBERT all exploit multi-frame context to dramatically reduce per-frame error, especially on ambiguous depth and occluded joints.

### design001

- Name: Delta-Input Channel Stacking (8-channel, single backbone pass)
- Result: no consolidated result
- Explanation: Concatenate [RGB_t, D_t, RGB_{t-5}, D_{t-5}] as an 8-channel input. The backbone patch embed is widened; all other backbone weights (all 24 ViT layers, DepthBucketPE) are unchanged. The model remains a single forward pass, preserving the full LLRD optimizer structure. The intuition: the backbone's attention layers can learn to extract temporal motion cues from the combined channel signal (optical-flow-like deltas emerge via learned filters).

### design002

- Name: Cross-Frame Memory Attention (2-frame, both trainable, gradient checkpointing)
- Result: <span style="color: red;">W 154.6 @e5</span>
- Explanation: Two-frame dataloader

### design003

- Name: Cross-Frame Memory Attention (2-frame, past frozen no_grad, centre trainable)
- Result: <span style="color: red;">W 167.1 @e5</span>
- Explanation: Dataloader

### design004

- Name: Three-Frame Symmetric Temporal Fusion (t-5, t, t+5; past/future frozen, centre trainable)
- Result: <span style="color: red;">W 167.7 @e5</span>
- Explanation: Dataloader: three-frame fetch

## idea018 - Weight Averaging: EMA and SWA on SOTA Triple-Combo

Expected Designs: 4

### design001

- Name: EMA of Full Model Weights (decay=0.999)
- Result: <span style="color: red;">W 212.6 @e5</span>
- Explanation: Maintain an exponential moving average (EMA) shadow copy of the live model. After each effective optimizer step (i.e. every accum_steps micro-batches), update the EMA copy with ema_param = decay * ema_param + (1 - decay) * live_param. Validate exclusively on the EMA copy. No architecture, loss, or optimizer changes.

### design002

- Name: EMA with Warmup (decay ramps from 0 to 0.9995)
- Result: <span style="color: red;">W 148.5 @e5</span>
- Explanation: Use a momentum-style EMA warmup formula: decay_t = min(target_decay, (1 + step) / (10 + step)) where step is the count of effective optimizer steps. At step=0: decay=0.091 (nearly no memory). At step=100: decay=0.917. At step=990: decay=0.9990. At step ≥ 2000: saturates at target_decay=0.9995. Validation is run on the EMA model exactly as in design001.

### design003

- Name: Stochastic Weight Averaging (SWA) over Last 5 Epochs
- Result: <span style="color: red;">W 169.7 @e5</span>
- Explanation: Run epochs 0-14 exactly as baseline (cosine LR + LLRD). At epoch 15, switch the optimizer's LR groups to a constant LR = 0.5 x cosine_LR_at_epoch15 and begin accumulating a uniform running average of the weights at the end of each epoch. After 5 SWA snapshots (epochs 15-19), report the averaged weights.

### design004

- Name: EMA (decay=0.999) + Last-Epoch Polish Pass
- Result: <span style="color: red;">W 212.6 @e5</span>
- Explanation: Train 20 epochs with EMA (decay=0.999) exactly as in design001. At the end of epoch 20: 1. Load EMA weights into the live model. 2. Run 1 extra epoch (epoch 21) with a constant flat LR = 1e-6 for all parameter groups. 3. Report val metrics on the polished live model at the end of epoch 21.

## idea019 - Anatomical Structure Priors for Iterative Refinement

Every design so far treats the 70 joint queries as independent, unstructured tokens. The model must learn all anatomical relationships (bone connectivity, symmetry, proportional constraints) purely from data within 20 epochs. Injecting lightweight structural priors -- especially into the refinement pass where coarse predictions are already available -- should help the model correct anatomically implausible configurations. This is orthogonal to all components already in the SOTA stack (LLRD, depth PE, wide head, iterative refinement) and modifies only how the decoder processes joint queries in the second refinement pass.

### design001

- Name: Bone-Length Auxiliary Loss on Refinement Output (Axis A1)
- Result: <span style="color: green;">W 103.1 @e20</span>
- Explanation: Bone-length consistency auxiliary loss applied to the refined prediction J2.

### design002

- Name: Kinematic-Chain Soft Self-Attention Bias in Refinement Pass (Axis A2)
- Result: <span style="color: green;">W 102.9 @e20</span>
- Explanation: Soft additive kinematic self-attention bias in the second decoder pass only, with a learnable scalar magnitude initialized to 0.0.

### design003

- Name: Left-Right Symmetry Loss (Axis B1)
- Result: <span style="color: green;">W 103.5 @e20</span>
- Explanation: Left-right symmetry auxiliary loss on limb bone lengths from the refined prediction J2.

### design004

- Name: Joint-Group Query Initialization in Refinement Pass (Axis B2)
- Result: <span style="color: green;">W 102.2 @e20</span>
- Explanation: Learnable group embeddings added to joint queries before the second decoder pass, zero-initialized so training starts identical to baseline.

### design005

- Name: Combined Anatomical Priors: Bone-Length Loss + Symmetry Loss + Kinematic Bias (Axis B3)
- Result: <span style="color: green;">W 106.5 @e20</span>
- Explanation: All three anatomical priors combined: bone-length loss (lambda_bone=0.1) + symmetry loss (lambda_sym=0.05) + kinematic self-attention bias (learnable scalar, init=0.0) in the refinement decoder pass.

## idea020 - Refinement-Specific Loss and Gradient Strategy

The two-decoder refinement (idea015/design004) outperforms the shared-decoder approach (idea015/design001) by 3 mm on weighted MPJPE. However, the loss function, deep supervision weighting, and gradient flow between the two passes have never been varied. The coarse decoder and refine decoder currently share the same loss type (Smooth L1 beta=0.05), the same deep supervision ratio (0.5:1.0), and the same optimizer group (head LR=1e-4). There are several promising axes

### design001

- Name: Stop-Gradient on Coarse J1 Before Refinement (Axis A1)
- Result: <span style="color: green;">W 101.9 @e20</span>
- Explanation: Detach J1 before passing it to self.refine_mlp(). This stops gradient flow from the refinement branch back through the coarse decoder via J1. The two decoders now optimize fully independent objectives: - Coarse decoder: optimized only via 0.5 * L(J1). - Refine decoder: optimized via 1.0 * L(J2), receiving a stable, detached J1 input.

### design002

- Name: Reduced Coarse Supervision Weight 0.1 (Axis A2)
- Result: <span style="color: green;">W 101.2 @e20</span>
- Explanation: Reduce the coarse supervision weight from 0.5 to 0.1

### design003

- Name: L1 Loss on Refinement Pass Only (Axis B1)
- Result: <span style="color: green;">W 99.4 @e20</span>
- Explanation: Replace the Smooth L1 loss for J2 with pure F.l1_loss. The coarse pass retains Smooth L1 (beta=0.05) via pose_loss()

### design004

- Name: Higher LR for Refine Decoder (2x Head LR, Axis B2)
- Result: <span style="color: green;">W 100.0 @e20</span>
- Explanation: Split model.head.parameters() into two separate optimizer groups: 1. Coarse-head group (LR = 1e-4): input_proj, joint_queries, decoder, joints_out, depth_out, uv_out. 2. Refine-head group (LR = 2e-4): refine_decoder, refine_mlp, joints_out2.

### design005

- Name: Residual Refinement Formulation (Axis B3)
- Result: <span style="color: red;">W 168.3 @e5</span>
- Explanation: Change the refinement formulation from absolute prediction to residual correction

## idea021 - Anatomical Priors on Two-Decoder SOTA

idea019 tested anatomical priors (bone-length loss, kinematic attention bias, symmetry loss, joint-group embeddings) on idea015/design001 (shared-decoder, val_weighted=101.94). Results were marginal: best body MPJPE of 105.77 (design002, kinematic bias) and best weighted of 102.22 (design004, joint-group init). However, idea019 was built on the wrong baseline -- it used idea015/design001 (shared-decoder) instead of idea015/design004 (two-decoder), which is 3 mm better on weighted MPJPE.

### design001

- Name: Kinematic Soft-Attention Bias in Refine Decoder Only (Axis A1)
- Result: <span style="color: green;">W 98.6 @e20</span>
- Explanation: Precompute a (70, 70) hop-distance matrix from SMPLX_SKELETON using BFS and convert to an additive attention bias: - 1-hop neighbors: +1.0 - 2-hop neighbors: +0.5 - 3-hop neighbors: +0.25 - Beyond 3 hops: 0.0

### design002

- Name: Joint-Group Query Injection Before Refine Decoder (Axis A2)
- Result: <span style="color: green;">W 100.7 @e20</span>
- Explanation: Define 4 anatomical groups and assign each of the 70 joints to one group: - Group 0 - Torso: pelvis, spine, neck, head (joints 0-3, approximately) - Group 1 - Arms: shoulders, elbows, wrists (joints ~4-9) - Group 2 - Legs: hips, knees, ankles (joints ~10-15) - Group 3 - Extremities: fingers, toes, face joints (joints 16-69)

### design003

- Name: Bone-Length Loss on J2 with lambda=0.05 (Axis B1)
- Result: <span style="color: green;">W 101.4 @e20</span>
- Explanation: Add a bone-length auxiliary loss on J2 only (not J1), with lambda_bone = 0.05 (half the weight tested in idea019 where 0.1 was neutral). The loss penalizes the mean absolute deviation between predicted and ground-truth bone lengths for body joints only (BODY_IDX, joints 0-21).

### design004

- Name: Kinematic Bias + Joint-Group Injection Combined (Axis B2)
- Result: <span style="color: green;">W 99.5 @e20</span>
- Explanation: Combine design001 (Axis A1) and design002 (Axis A2) of this idea: 1. Kinematic soft-attention bias (kin_bias_scale * kin_bias) passed as tgt_mask to self.refine_decoder only. 2. Joint-group query injection (group_emb(joint_group_ids)) added to queries2 before the refine decoder.
