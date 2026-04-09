# Review: idea009/design002 — Wide Head (hidden_dim=384)

**Date:** 2026-04-09
**Verdict:** APPROVED

---

## Checklist

### 1. Fidelity to idea.md (Axis A2)

The design implements exactly what idea.md specifies for Design 2:
- `head_num_layers = 4` (unchanged) ✓
- `head_hidden = 384` (was 256) ✓
- `head_num_heads = 8` (unchanged) ✓
- Divisibility: 384 / 8 = 48 — valid ✓
- LLRD schedule from idea004/design002 preserved exactly (gamma=0.90, unfreeze_epoch=5, lr_backbone=1e-4, lr_head=1e-4) ✓
- epochs=20, warmup_epochs=3, weight_decay=0.03, grad_clip=1.0, lambda_depth=0.1, lambda_uv=0.2, head_dropout=0.1, drop_path=0.1 — all match spec ✓

### 2. Mathematical Correctness

**Parameter count audit:**

Baseline head (hidden_dim=256, 4 layers):
- input_proj: 1024×256 + 256 = 262,400 ✓
- joint_queries: 70×256 = 17,920 ✓
- Per decoder layer: Self-attn (4×256²=262,144) + Cross-attn (4×256²=262,144) + FFN (2×256×1024 = 524,288 + biases ≈524,544) + LN (4×2×256=2,048) = ~1.051M ✓
- 4 layers ≈ 4.203M ✓
- Output heads: (256×3+3)+(256×1+1)+(256×2+2) = 768+3+256+1+512+2 = 1,542 (design says 1,539 — trivial rounding, inconsequential) ✓
- Total ≈ 5.48M ✓

Wide head (hidden_dim=384, 4 layers):
- input_proj: 1024×384 + 384 = 393,600 ✓
- joint_queries: 70×384 = 26,880 ✓
- Per decoder layer: Self-attn (4×384²=589,824) + Cross-attn (4×384²=589,824) + FFN (2×384×1536 = 1,179,648 + biases ~3,456 ≈ 1,183,104) + LN (4×2×384=3,072) = ~2.366M ✓
- 4 layers ≈ 9.462M ✓
- Output heads: (384×3+3)+(384×1+1)+(384×2+2) = 1,152+3+384+1+768+2 = 2,310 (design says 2,307 — trivial rounding) ✓
- Total ≈ 9.88M ✓

Delta: +4.4M parameters over baseline. Total model ~303M. Well within VRAM budget. ✓

### 3. VRAM / Feasibility

- Head adds ~4.4M parameters. At fp32 that is ~17.6 MB; at fp16/mixed ~8.8 MB. The backbone dominates at ~9 GB. This change is negligible relative to VRAM budget. ✓
- No new activations beyond what the decoder width change implies; no new decoder layers; no extra forward passes. Memory impact is minimal. ✓
- Runs within 20-epoch proxy limit on a single 1080ti (11 GB VRAM). ✓

### 4. Completeness — Config Coverage

All required config fields are present and unambiguous:
- `output_dir`, `head_hidden`, `head_num_heads`, `head_num_layers`
- `lr_backbone`, `lr_head`, `gamma`, `unfreeze_epoch`, `warmup_epochs`, `epochs`
- `weight_decay`, `grad_clip`, `lambda_depth`, `lambda_uv`, `head_dropout`, `drop_path`

No missing fields. Builder has no guesswork required. ✓

### 5. Architectural Feasibility

- The design correctly identifies that `Pose3DHead` already parameterizes all widths through a single `hidden_dim` argument. Setting `head_hidden=384` propagates automatically to `input_proj`, `joint_queries`, decoder d_model, dim_feedforward, and output linears. ✓
- No structural code changes needed. ✓
- LLRD optimizer grouping in train.py automatically covers the wider head layers — no manual updates required. ✓
- No changes to infra.py, transforms.py, or backbone. ✓

### 6. Scientific Rationale

- Widening the hidden dimension from 256 to 384 is a well-motivated change: the backbone produces 1024-dim features, projecting them down to 256 is an aggressive 4x compression. Relaxing to 384 (roughly 2.7x) allows joint queries to retain more signal per cross-attention step. ✓
- The expected 1–4 mm improvement is reasonable and conservative. The framing as an orthogonal test to Design 1 (depth vs. width) is scientifically sound. ✓

---

## Issues Found

None. The design is clean, self-consistent, and directly implements Axis A2 from idea.md.

---

## Verdict: APPROVED

This design is approved to proceed to the Builder. The single config field change (`head_hidden = 384`) is sufficient; no train.py or model.py modifications are needed.
