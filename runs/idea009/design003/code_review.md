# Code Review — idea009/design003

**Design:** Sine-Cosine Joint Query Initialization  
**Date:** 2026-04-09  
**Reviewer:** Reviewer agent

---

## Checklist

### 1. Scope of Change (model.py only)

The design specifies that **only `model.py`** is modified. Verified:
- `model.py` — modified (sinusoidal init added). CORRECT.
- `config.py` — `output_dir` updated to `runs/idea009/design003`. No other changes. CORRECT.
- `train.py` — no changes (confirmed via grep). CORRECT.
- `transforms.py`, `infra.py` — unchanged. CORRECT.

### 2. `_sinusoidal_init` Static Method

The implementation at lines 82–94 of `model.py`:

```python
@staticmethod
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
```

- Formula matches the design spec (`PE[j,2k] = sin(j / 10000^(2k/d_model))`, `PE[j,2k+1] = cos(...)`). CORRECT.
- Log-space numerics (`exp(arange * -log(10000)/d_model)`) match standard Vaswani 2017 implementation. CORRECT.
- `torch.no_grad()` wraps the `.copy_()` call — autograd graph not polluted at init. CORRECT.
- Implemented as a `@staticmethod` of `Pose3DHead` (the design permitted either module-level or static method). CORRECT.
- `math` is imported at the top of `model.py` (line 18). CORRECT.

### 3. `_init_weights` Caller

```python
def _init_weights(self):
    self._sinusoidal_init(self.joint_queries)
    print(f"[init] joint_queries[0, :4] = {self.joint_queries.weight[0, :4].tolist()}")
    for m in [self.joints_out, self.depth_out, self.uv_out]:
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)
```

- `trunc_normal_` on `joint_queries` has been replaced with `_sinusoidal_init`. CORRECT.
- `trunc_normal_` is preserved for the regression heads (`joints_out`, `depth_out`, `uv_out`) — design does not change these. CORRECT.
- Sanity print confirms joint 0 values. Reported result `[0.0, 1.0, 0.0, 1.0]` matches `[sin(0), cos(0), sin(0), cos(0)]` exactly. CORRECT.
- `_init_weights()` is called at the end of `__init__`, after `nn.Embedding` construction, so sinusoidal values overwrite the default uniform init. CORRECT.

### 4. No Double-Overwrite Risk

The design explicitly notes that `nn.Embedding` does not call `reset_parameters()` after `__init__` unless explicitly invoked. The builder has not added any such call. The sinusoidal init is the final write to `embedding.weight` before the optimizer takes over. CORRECT.

### 5. Gradient Flow

`joint_queries` is a plain `nn.Embedding` (a learnable `nn.Parameter`). The `torch.no_grad()` block only suppresses gradient tracking during the init copy; it does not make the parameter non-trainable. All downstream gradient computation proceeds normally. CORRECT.

### 6. config.py — All Fields

All 16 required config fields verified against the design spec:

| Field | Design | Code | Match |
|---|---|---|---|
| output_dir | `runs/idea009/design003` | `runs/idea009/design003` | YES |
| head_hidden | 256 | 256 | YES |
| head_num_heads | 8 | 8 | YES |
| head_num_layers | 4 | 4 | YES |
| head_dropout | 0.1 | 0.1 | YES |
| drop_path | 0.1 | 0.1 | YES |
| lr_backbone | 1e-4 | 1e-4 | YES |
| lr_head | 1e-4 | 1e-4 | YES |
| gamma | 0.90 | 0.90 | YES |
| unfreeze_epoch | 5 | 5 | YES |
| warmup_epochs | 3 | 3 | YES |
| epochs | 20 | 20 | YES |
| weight_decay | 0.03 | 0.03 | YES |
| grad_clip | 1.0 | 1.0 | YES |
| lambda_depth | 0.1 | 0.1 | YES |
| lambda_uv | 0.2 | 0.2 | YES |

All 16 fields match. No new config fields introduced (sinusoidal init exposes no tunable hyperparameter). CORRECT.

### 7. Parameter Count

No change to `nn.Embedding` shape (70×256 = 17,920 weights). Parameter count is identical to the design002 baseline. CORRECT.

### 8. 2-Epoch Smoke Test

Passed. Sanity check `joint_queries[0, :4] = [0.0, 1.0, 0.0, 1.0]` confirmed, matching the exact values predicted by the design spec.

---

## Issues Found

None.

---

## Verdict

**APPROVED**

The implementation is a precise, minimal, and correct translation of the design spec. The sinusoidal PE formula matches Vaswani 2017, log-space numerics are correct, the `torch.no_grad()` wrapper is in place, `trunc_normal_` on the regression heads is preserved unchanged, and all 16 config fields match the spec exactly. The 2-epoch smoke test passed with the expected sanity values.
