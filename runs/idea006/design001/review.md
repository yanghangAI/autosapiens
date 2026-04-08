# Review — idea006 / design001

**Design_ID:** idea006/design001  
**Reviewer:** Reviewer agent  
**Date:** 2026-04-08  
**Verdict:** APPROVED

---

## Summary

design001 proposes adding a single `RandomHorizontalFlip` (p=0.5) transform into `build_train_transform()` after `CropPerson` and before `SubtractRoot`. The design is the first of six augmentation variants for idea006.

---

## Evaluation

### 1. Fidelity to idea.md
The design exactly matches the Axis 1 specification in idea.md ("Horizontal Flip only, p=0.5"). No scope creep; no deviation from the stated intent.

### 2. Mathematical Correctness
- The coordinate convention is verified against `baseline/transforms.py::SubtractRoot`: `u_px = K[0,0]*(-Y/X) + K[0,2]`. Negating Y (lateral) correctly mirrors the horizontal pixel location.
- `pelvis_uv` re-computation via `SubtractRoot` on the already-negated joint Y-coordinate is mathematically sound. The design correctly notes that explicit `pelvis_uv[0] *= -1` is NOT needed because `SubtractRoot` recomputes it from the flipped joints — this is correct, given the pipeline order `RandomHorizontalFlip → SubtractRoot`.
- `pelvis_depth` is flip-invariant (depth/distance unchanged by lateral flip) — correct.
- The K matrix does not require modification post-flip within a symmetric crop — correct and well-justified.

### 3. FLIP_PAIRS Accuracy
The `FLIP_PAIRS` constant in the design matches `infra.py` exactly (verified line-by-line). Import from `infra` is specified correctly.

### 4. Implementation Blueprint
The pseudocode `RandomHorizontalFlip.__call__` is complete and unambiguous:
- Image flip via `cv2.flip(..., 1)` — correct.
- `joints[:, 1] *= -1.0` before swap — correct order.
- Left-right pair swap using a loop — correct.
- `joints` is `.copy()`-ed before mutation — avoids aliasing bugs.

### 5. Pipeline Order
`CropPerson → RandomHorizontalFlip → SubtractRoot → ToTensor` is correct. Flipping after crop (fixed output resolution) ensures the crop dimensions are stable. Flipping before `SubtractRoot` lets SubtractRoot recompute `pelvis_uv` correctly on the flipped data.

### 6. Feasibility / Compute Constraint
This augmentation adds negligible cost (a conditional array flip and index swap). The config is identical to baseline (sapiens_0.3b, 20 epochs, batch 4, accum 8) — well within the 1080Ti 11GB VRAM constraint established by previous runs.

### 7. Config Completeness
All fields are explicitly specified in `config.py`. Only `output_dir` differs from baseline. No ambiguity for the Builder.

### 8. Files to Modify
Correctly identified: only `transforms.py` and `config.py`. `train.py` and `model.py` are unchanged, which is appropriate.

---

## Issues Found

None. The design is complete, mathematically sound, computationally feasible, and unambiguously specified.

---

## Verdict: APPROVED
