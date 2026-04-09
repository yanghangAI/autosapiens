# Review Log — idea009

---

## idea009/design001 — 2026-04-09 — APPROVED

**Design:** 6-Layer Decoder Head (head_num_layers=6, head_hidden=256, head_num_heads=8). All other parameters frozen to idea004/design002 baseline (LLRD gamma=0.90, unfreeze_epoch=5).

**Verdict:** APPROVED

**Reasoning:** Exactly matches Axis A1 spec. All config fields fully specified. Parameter math correct (+2.1M params, new head ~7.9M, VRAM safe). No code changes required beyond config.py. Optimizer LLRD grouping automatically includes new layers. Minimal, well-scoped change — no issues found.

---

## idea009/design002 — 2026-04-09 — APPROVED

**Design:** Wide Head (head_hidden=384, head_num_layers=4, head_num_heads=8). All other parameters frozen to idea004/design002 baseline (LLRD gamma=0.90, unfreeze_epoch=5).

**Verdict:** APPROVED

**Reasoning:** Exactly matches Axis A2 spec. Divisibility check passes (384/8=48). All 16 config fields fully specified. Parameter math correct (+4.4M params, wide head ~9.88M, total ~303M, VRAM safe). No structural code changes required — Pose3DHead already parameterizes all widths through a single hidden_dim argument; setting head_hidden=384 propagates automatically. Optimizer LLRD grouping automatically covers wider layers. No issues found.

---

## idea009/design003 — 2026-04-09 — APPROVED

**Design:** Sine-Cosine Joint Query Initialization (Axis B1). head_num_layers=4, head_hidden=256. Replaces `trunc_normal_(std=0.02)` init on `nn.Embedding(70, 256)` with Vaswani et al. sinusoidal PE by joint index 0..69. Weights remain fully trainable post-init.

**Verdict:** APPROVED

**Reasoning:** Faithfully implements Axis B1. PE formula and numerically stable log-space implementation are mathematically correct. Sanity check `weight[0, :4] ≈ [0, 1, 0, 1]` is valid. Parameter count is unchanged (initialization-only change). VRAM usage identical to baseline. Change is isolated to `Pose3DHead._init_weights` in `model.py` only. All 16 config fields are fully specified. Builder has unambiguous instructions with no guesswork required. No issues found.

---

## idea009/design004 — 2026-04-09 — APPROVED

**Design:** Per-Layer Input Feature Gate (Axis B2). head_num_layers=4, head_hidden=256. Adds `nn.ParameterList` of 4 scalar gates (one per decoder layer); each sigmoid-activated gate scales the projected memory tensor before cross-attention. Gates initialized to 4.6 (sigmoid(4.6) ≈ 0.99, effectively open at epoch 0). The single `self.decoder(queries, memory)` call is replaced with an explicit per-layer loop over `self.decoder.layers`.

**Verdict:** APPROVED

**Reasoning:** Exactly matches Axis B2 spec. Initialization to sigmoid(4.6) ≈ 0.99 is mathematically honest (sigmoid cannot reach 1.0 exactly) and the design explicitly acknowledges the small discrepancy — acceptable. Sigmoid bounding to (0,1) correctly prevents gate collapse. Forward pass loop over `self.decoder.layers` with optional norm guard is architecturally correct. 0-dim scalar gate broadcasts correctly over (B, S, hidden_dim) memory tensor. Gate parameters land in `lr_head` optimizer group automatically via `nn.ParameterList` inside `Pose3DHead` — no train.py changes needed. Parameter overhead: 4 scalars on ~5.48M head — completely negligible, no VRAM impact. All 16 config fields specified. Gate init value (4.6) correctly kept as implementation-internal, not a config field. Builder instructions are concrete and testable. No issues found.

---

## idea009/design005 — 2026-04-09 — APPROVED

**Design:** Output LayerNorm Before Final Regression (Axis B3). head_num_layers=4, head_hidden=256. Adds `nn.LayerNorm(256)` (`self.output_norm`) to `Pose3DHead`, applied to decoder output immediately before `pelvis_token` extraction and the three output projections. All other parameters unchanged from idea004/design002 baseline.

**Verdict:** APPROVED

**Reasoning:** Exactly matches Axis B3 / Design 5 spec. Positioning is architecturally correct — `nn.TransformerDecoder` with `norm_first=True` and no explicit `norm` argument leaves the decoder output unnormalized; `output_norm` fills this gap as expected in standard pre-norm transformers. Initialization (weight=1, bias=0) is benign and expected to be beneficial. Parameters (512 total: 256 weight + 256 bias) land in `lr_head` optimizer group automatically — no train.py changes needed. No VRAM impact. Builder instructions require exactly two one-line changes in model.py and an `output_dir` update in config.py. All 16 config fields explicitly specified. No issues found.
