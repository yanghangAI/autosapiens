# Code Review — idea013/design003

**Design:** Bone-Length Auxiliary Loss (lambda_bone=0.1)
**Reviewer verdict:** APPROVED

## Checklist

1. **SMPLX_SKELETON import** — Line 44 imports `SMPLX_SKELETON` from infra. Correct.

2. **BODY_BONES definition** — Line 52: `BODY_BONES = [(a, b) for a, b in SMPLX_SKELETON if a < 22 and b < 22]`. Matches design exactly. Filters to body-only edges.

3. **bone_length_loss function** — Lines 55-66: Implementation matches design spec precisely. Per-bone L2 distance for pred and target, absolute difference of bone lengths, mean over batch, averaged over number of bones. Correct.

4. **Loss computation** — Lines 203-207:
   - `l_pose = pose_loss(...)` — standard Smooth L1 (beta=0.05), unchanged. Correct.
   - `l_bone = bone_length_loss(out["joints"][:, BODY_IDX], joints[:, BODY_IDX], BODY_BONES)` — correct arguments.
   - `loss = (l_pose + 0.1 * l_bone + args.lambda_depth * l_dep + args.lambda_uv * l_uv) / args.accum_steps` — lambda_bone=0.1 hardcoded. Matches design.

5. **del statement** — Line 237 includes `l_bone`. Correct.

6. **Config verification:**
   - `output_dir` = correct path (idea013/design003)
   - All other config fields match the design table exactly.
   - No config field for lambda_bone (hardcoded), as specified in design.

7. **LLRD optimizer logic** — Present and correct, identical to other idea013 designs.

8. **Note:** The iter_logger does not log `loss_bone` separately (design mentions this but the implementation only logs the standard fields). This is a minor cosmetic omission that does not affect correctness. The bone loss is included in the total `loss` value that is logged.

## Issues

None found (minor logging omission is cosmetic only).
