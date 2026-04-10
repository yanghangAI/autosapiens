# idea010 — Code Review Log

---

## design001 — Last-4-Layer Concatenation + Linear Projection

**Date:** 2026-04-10
**Verdict:** APPROVED

Config: all 16 fields match design spec (output_dir, LLRD params, head params, multiscale_mode="concat4", multiscale_layers=[20,21,22,23]). Backbone extracts layers {20,21,22,23}, applies LN to each, concatenates to (B,960,4096), reshapes to (B,4096,H,W). MultiScaleConcat projects Linear(4096,1024) with Xavier init. Aggregator wired into both optimizer builders at lr=1e-4. Head in_channels=1024 unchanged. No bugs. 2-epoch test passed.

---

## design002 — Learned Layer Weights (Softmax-Weighted Sum)

**Date:** 2026-04-10
**Verdict:** APPROVED

Config: all 16 fields match design spec (multiscale_mode="learned_weights", multiscale_layers=[20,21,22,23]). Backbone returns list of 4 normed feature maps from layers {20,21,22,23}. LearnedLayerWeights has nn.Parameter(torch.zeros(4)) so softmax gives uniform 0.25 at init. Forward computes softmax-weighted sum correctly. Aggregator (4 scalar params) wired into both optimizer builders at lr=1e-4. No bugs. 2-epoch test passed.

---

## design003 — Feature Pyramid with 3 Scales

**Date:** 2026-04-10
**Verdict:** APPROVED

Config: all 16 fields match design spec (multiscale_mode="pyramid3", multiscale_layers=[7,15,23]). Backbone returns list of 3 normed feature maps from layers {7,15,23}. FeaturePyramid has 3x Linear(1024,256) + Linear(768,1024), all Xavier init. Param count ~1.57M matches design. Aggregator wired into both optimizer builders at lr=1e-4. No bugs. 2-epoch test passed.

---

## design004 — Cross-Scale Attention Gate

**Date:** 2026-04-10
**Verdict:** APPROVED

Config: all 17 fields match design spec (multiscale_mode="cross_gate", multiscale_layers=[11,23], gate_bias_init=-5.0). Backbone returns [layer_11, layer_23] both normed. CrossScaleGate has Linear(1024,1) with zero weight and bias=-5.0; sigmoid(-5)~0.007 so initial output ~ final_feat * 1.007 ~ baseline. Residual form `final_feat * (1 + gate)` matches design exactly. 1,025 params. Aggregator wired into both optimizer builders at lr=1e-4. No bugs. 2-epoch test passed.

---

## design005 — Alternating Layer Average

**Date:** 2026-04-10
**Verdict:** APPROVED

Config: all 16 fields match design spec (multiscale_mode="alt_avg", multiscale_layers=[1,3,5,...,23]). Backbone extracts 12 odd-indexed layers using memory-efficient running-sum approach (as recommended by design), applies LN to each before accumulation, divides by count=12. Returns single (B,1024,H,W) tensor. No aggregator module, zero new params. Optimizer unchanged from baseline (13 frozen / 26 full groups). No bugs. 2-epoch test passed.

---
