# Review — idea006/design005

**Design_ID:** idea006/design005  
**Reviewer:** Reviewer Agent  
**Date:** 2026-04-08  
**Verdict:** APPROVED

---

## Summary

design005 combines the two geometric augmentations from design001 (RandomHorizontalFlip) and design002 (RandomScaleJitter), implementing Axis 5 of idea006 exactly as specified.

---

## Evaluation

### 1. Alignment with idea.md
The design correctly implements Axis 5: "Combined Geometric Augmentation (Flip + Scale Jitter)". All six axes of idea006 are now covered across designs 001–005 (with design006 remaining). No deviation from the specification.

### 2. Transform Ordering
`RandomScaleJitter → CropPerson → RandomHorizontalFlip → SubtractRoot → ToTensor`

This ordering is correct and well-justified:
- `RandomScaleJitter` modifies `sample["bbox"]` before `CropPerson` consumes it — required.
- `RandomHorizontalFlip` operates on the fixed-resolution crop output — required.
- `SubtractRoot` follows the flip, so it recomputes `pelvis_uv` from the already-flipped joints — correct; no manual `pelvis_uv` negation needed.

### 3. Mathematical Correctness

**RandomScaleJitter:** Identical formula to the APPROVED design002. `s ~ Uniform(0.8, 1.2)` scales bbox half-extents around the center. Root-relative 3D joints in metric camera space are correctly left unchanged. Fallback for `[cx, cy, w, h]` bbox format is included.

**RandomHorizontalFlip:** Identical logic to the APPROVED design001. `joints[:, 1] *= -1.0` negates the lateral Y-axis; FLIP_PAIRS (23 pairs, remapped 0..69 index space, imported from infra) swap left-right joints. `pelvis_uv` is not manually negated — SubtractRoot recomputes it implicitly from the flipped joint set. This is the correct approach (consistent with Reviewer lesson learned from design001).

**FLIP_PAIRS:** Same 23-pair list vetted and approved in design001.

### 4. Config Fields
All fields explicitly specified and identical to baseline except `output_dir = "runs/idea006/design005"`. No guessing required for the Builder.

### 5. Compute Feasibility
Only `transforms.py` and `config.py` are modified. Model architecture and training loop are unchanged. Both augmentations are CPU-side, near-zero overhead. Runs comfortably within 20 epochs on a single 1080Ti.

### 6. Files to Modify
Clearly enumerated: `transforms.py` (add both classes, update `build_train_transform`) and `config.py` (update `output_dir`). `train.py` and `model.py` are unchanged. The Builder note on bbox format verification is appropriate due-diligence.

### 7. No Issues Found
No vagueness, no missing parameters, no mathematical errors, no ordering mistakes.

---

## Verdict: APPROVED

Design is complete, mathematically correct, architecturally feasible, and unambiguous for the Builder. Ready to proceed to implementation.
