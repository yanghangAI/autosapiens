# Design 003 — Sine-Cosine Joint Query Initialization

## Starting Point

`runs/idea004/design002`

This design builds directly on idea004/design002 (val_mpjpe_body = 112.3 mm), the current best result, which uses the LLRD schedule (gamma=0.90, unfreeze_epoch=5). Only the initialization of the joint query embeddings is changed — the head architecture (4 layers, hidden_dim=256), the LLRD schedule, and all other hyperparameters remain identical to the baseline.

## Problem

The baseline `Pose3DHead` initializes its 70 joint query embeddings with `nn.Embedding(70, 256)` and applies `trunc_normal_(std=0.02)` — a near-zero random initialization. This gives the decoder no structural information about the joint ordering at the start of training. Each query must learn from scratch that it corresponds to a particular anatomical location, with no inductive bias toward the sequential structure of the joint index. The queries are fully learnable after initialization, so any starting point is valid, but a structured starting point could accelerate early convergence and reduce the variance of the final solution.

## Proposed Change

Replace the random initialization of `self.joint_queries.weight` with a sine-cosine positional encoding derived from each joint's index (0..69). The `nn.Embedding` weight tensor remains a standard `nn.Parameter` and continues to be updated by the optimizer — only the *starting values* change.

### Sine-cosine encoding formula

For joint index `j` (0-indexed, j = 0..69) and embedding dimension `d` (d_model = 256):

```
PE[j, 2k]   = sin(j / 10000^(2k / d_model))
PE[j, 2k+1] = cos(j / 10000^(2k / d_model))
```

for k = 0, 1, ..., d_model/2 − 1. This is the standard Vaswani et al. (2017) sinusoidal positional encoding applied to the joint index space rather than a sequence position. The result is a 70 × 256 matrix of smooth, structured values in [−1, 1] that vary continuously with joint index and with embedding dimension.

### Why this is a sound initialization

1. **Non-random structure:** Adjacent joint indices (e.g., left knee = 4, left ankle = 7 in standard COCO ordering) produce nearby but distinct encoding vectors, so the decoder starts with some notion that joint 4 and joint 7 are different but related queries.
2. **Full-rank:** The sinusoidal matrix is full-rank for typical d_model values (256 >> 70), so no information is collapsed at initialization.
3. **Bounded magnitude:** All values are in [−1, 1], similar in scale to a `trunc_normal_(std=0.02)` but with more structured variance. The optimizer will adjust the scale through the first warmup epochs.
4. **No code-path change at inference:** The weights are still a plain `nn.Embedding` parameter — the change is entirely in `_init_weights`.

### Implementation (model.py only)

In `Pose3DHead._init_weights`, replace:

```python
nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
```

with:

```python
import math

def _sinusoidal_init(embedding: nn.Embedding) -> None:
    num_joints, d_model = embedding.weight.shape
    pe = torch.zeros(num_joints, d_model)
    position = torch.arange(num_joints, dtype=torch.float).unsqueeze(1)  # (J, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    with torch.no_grad():
        embedding.weight.copy_(pe)

_sinusoidal_init(self.joint_queries)
```

This helper can be defined as a module-level function or a static method of `Pose3DHead`. It must be called *after* `nn.Embedding` construction so it overwrites the default uniform init.

No other files change. The `nn.Embedding` is still gradient-enabled, so the sinusoidal values are only a starting point.

## Configuration (`config.py` changes)

```python
output_dir      = "/work/pi_nwycoff_umass_edu/hang/auto/runs/idea009/design003"

# Model — identical to baseline except query init (handled in model.py)
head_hidden     = 256
head_num_heads  = 8
head_num_layers = 4        # unchanged

# Schedule — identical to idea004/design002
lr_backbone     = 1e-4
lr_head         = 1e-4
gamma           = 0.90
unfreeze_epoch  = 5
warmup_epochs   = 3
epochs          = 20
weight_decay    = 0.03
grad_clip       = 1.0
lambda_depth    = 0.1
lambda_uv       = 0.2
head_dropout    = 0.1
drop_path       = 0.1
```

No new config fields are introduced — the sinusoidal initialization is hardcoded in `model.py` and does not expose a tunable hyperparameter.

## Parameter Count Estimate

Identical to the baseline head (~5.48M params). The sinusoidal initialization changes no tensor shapes or parameter counts — it only affects the starting values of the 70 × 256 = 17 920 joint query weights.

## Implementation Notes

- The only file to modify is `model.py`, specifically `Pose3DHead._init_weights`.
- The `_sinusoidal_init` helper must be called with the device-agnostic `torch.no_grad()` context to avoid polluting the autograd graph at init time.
- `train.py`, `infra.py`, `transforms.py`, and `config.py` are structurally unchanged versus the baseline (only `output_dir` in config changes).
- The Builder should verify via a quick `print(model.head.joint_queries.weight[0, :4])` at initialization that the first four values of joint 0 are approximately `[sin(0), cos(0), sin(0), cos(0)] = [0.0, 1.0, 0.0, 1.0]` (within floating-point precision), confirming the init ran.
- Because `nn.Embedding` weights are copied from a pre-computed tensor, there is no risk of the sinusoidal values being overwritten by a subsequent `reset_parameters()` call — `nn.Embedding` does not call `reset_parameters` after `__init__` unless explicitly invoked.

## Expected Outcome

Expecting val_mpjpe_body to match or improve on the 112.3 mm baseline by 0–2 mm. The effect of initialization is typically most visible in the first 3–5 epochs (faster early convergence) and may wash out by epoch 20 as the weights adapt. If the result is statistically comparable to Design 1 (6-layer) and Design 2 (wide head), that confirms this initialization is a neutral-to-positive change suitable for stacking with those designs. A modest improvement here motivates combining sinusoidal init with deeper or wider head variants.
