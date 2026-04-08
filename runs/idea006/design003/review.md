# Review — idea006/design003

**Design_ID:** idea006/design003
**Date:** 2026-04-08
**Verdict:** APPROVED

---

## Summary

design003 implements Axis 3 from `idea006/idea.md`: Color Jitter augmentation applied
to the RGB tensor only, with the depth channel left unmodified. The design is minimal,
mathematically correct, and leaves no implementation details ambiguous for the Builder.

---

## Checklist

### Alignment with idea.md
- [x] Implements Axis 3 exactly: `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)`
- [x] Parameters match the specification in idea.md verbatim.

### Mathematical / Label Correctness
- [x] Joints are 3D metric root-relative coordinates — photometrically invariant by definition.
- [x] `sample["joints"]`, `sample["intrinsic"]`, `pelvis_uv`, `pelvis_depth` correctly identified as unaffected.
- [x] `sample["depth"]` correctly left unchanged.

### Implementation Correctness
- [x] `RGBColorJitter` wrapper is fully specified with exact class code.
- [x] Applied after `ToTensor` (float tensor [0,1]), which is the correct position for torchvision's tensor API (requires >= 0.8, which is standard).
- [x] `T.ColorJitter.__call__` applies a freshly sampled random transform per call — per-image randomization is correct without manual seed management.
- [x] `build_val_transform` is explicitly stated as unchanged.

### Config Completeness
- [x] All required config fields are listed (output_dir, arch, img dimensions, head dims, epochs, LR, weight_decay, warmup, grad_clip, lambda_depth, lambda_uv).
- [x] Only `output_dir` differs from baseline — appropriate for a single-axis augmentation experiment.

### Resource Constraints
- [x] No architectural changes; same model and epoch budget. Easily fits within 20-epoch / 1080Ti (11GB VRAM) constraints.

### Files to Modify
- [x] Only `transforms.py` and `config.py` — minimal and correct scope.
- [x] `train.py` and `model.py` unchanged — correctly noted.

### Builder Ambiguity
- [x] None. The `RGBColorJitter` class is provided verbatim; exact parameters, exact pipeline insertion point, and exact import are all specified.

---

## Conclusion

The design is complete, correct, and ready for implementation. No issues found.
