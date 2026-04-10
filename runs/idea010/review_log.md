# idea010 -- Design Review Log

---

## design001 -- Last-4-Layer Concatenation + Linear Projection
**Verdict: APPROVED** | Date: 2026-04-10

Extracts layers 20-23 (correct for 24-block model), concatenates to (B, 4096, 40, 24), projects back to (B, 1024, 40, 24) via Linear(4096, 1024) with Xavier init. ~4.2M new params. Aggregator optimizer group with lr=lr_head. All config fields specified. No issues found.

---

## design002 -- Learned Layer Weights (Softmax-Weighted Sum)
**Verdict: APPROVED** | Date: 2026-04-10

Extracts layers 20-23, softmax-weighted sum with 4 learnable scalars initialized to 0.0 (uniform 0.25 via softmax). Only 4 new params. Output shape unchanged. All config fields specified. No issues found.

---

## design003 -- Feature Pyramid with 3 Scales
**Verdict: APPROVED** | Date: 2026-04-10

Extracts layers 7, 15, 23 (early/mid/final, correctly adapted from idea.md's {4,8,12} for 12-block to 24-block). Three Linear(1024,256) per-scale projections + Linear(768,1024) fusion. Xavier init. ~1.57M new params. All config fields specified. No issues found.

---

## design004 -- Cross-Scale Attention Gate
**Verdict: APPROVED** | Date: 2026-04-10

Extracts layers 11 and 23 (mid/final, correctly adapted from idea.md's {6,12}). Spatial gate via Linear(1024,1) with zero-init weight and bias=-5.0 (sigmoid ~ 0 at init). Residual form output = layer_23 * (1 + gate). 1,025 new params. Designer correctly fixed idea.md's erroneous "zero-init bias" suggestion (which would give sigmoid(0)=0.5, not ~0). Config includes gate_bias_init field. No issues found.

---

## design005 -- Alternating Layer Average
**Verdict: APPROVED** | Date: 2026-04-10

Extracts 12 evenly-spaced layers (odd indices 1,3,...,23) and averages. Zero new parameters. Running-sum approach recommended for memory efficiency. No optimizer changes needed. All config fields specified. No issues found.
