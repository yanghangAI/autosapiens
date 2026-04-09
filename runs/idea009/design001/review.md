# Review — idea009/design001

**Design_ID:** idea009/design001  
**Date:** 2026-04-09  
**Verdict:** APPROVED

---

## Summary

Design 001 proposes increasing `head_num_layers` from 4 to 6 in the `Pose3DHead` transformer decoder. All other hyperparameters are frozen to the idea004/design002 baseline. This is Axis A1 from the idea.md spec.

---

## Evaluation

### 1. Fidelity to idea.md

The idea.md specifies for Design 1:

> **6-layer decoder** — head_num_layers=6, head_hidden=256, head_num_heads=8; everything else fixed to idea004/design002 schedule.

The design.md satisfies this exactly. No other axes are mixed in. The LLRD schedule (gamma=0.90, unfreeze_epoch=5, lr_backbone=1e-4, lr_head=1e-4) is preserved verbatim.

### 2. Completeness

All required config fields are explicitly specified:
- `output_dir`, `head_hidden`, `head_num_heads`, `head_num_layers` — present
- `lr_backbone`, `lr_head`, `gamma`, `unfreeze_epoch`, `warmup_epochs`, `epochs` — present
- `weight_decay`, `grad_clip`, `lambda_depth`, `lambda_uv`, `head_dropout`, `drop_path` — present

The Builder is given no ambiguity; there is nothing to guess.

### 3. VRAM / Parameter Budget

The parameter estimate is correct:
- Per layer: self-attn (262 144) + cross-attn (262 144) + FFN (~527 360) + layer norms (~2 048) ≈ 1.054M
- Two additional layers ≈ +2.1M params
- New head total ≈ 7.9M — well within headroom; backbone dominates VRAM at ~9 GB for batch=4

No VRAM concern.

### 4. Mathematical / Architectural Correctness

- `TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=1024)`: 256/8=32 head dim — valid.
- Increasing `num_layers` passes through to `nn.TransformerDecoder(decoder_layer, num_layers=6)` without any code change — the design correctly notes this.
- The LLRD optimizer groups all head parameters under `lr_head`; adding two extra layers means those parameters are automatically included in the head group. No custom optimizer logic is needed. Correct.

### 5. Implementation Notes

The note that `model.py` requires no code change (only `config.py` is updated) is correct and desirable — surgically minimal scope.

### 6. Expected Outcome

The claimed 1–3 mm improvement from DETR-family literature is plausible and appropriately hedged. The design also correctly anticipates the null/degradation hypothesis (signal of diminishing returns), which makes this a well-scoped ablation.

---

## Issues

None.

---

## Verdict: APPROVED
