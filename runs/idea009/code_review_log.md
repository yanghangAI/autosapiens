# Code Review Log — idea009

---

## idea009/design001 — 2026-04-09 — APPROVED

**Change:** `head_num_layers = 6` (was 4) in `config.py` only.

All 16 config fields match the design spec exactly. `train.py` passes `head_num_layers=args.head_num_layers` directly to `SapiensPose3D`, which passes it to `Pose3DHead` as `num_layers`. The 2 additional decoder layers are automatically included in the `model.head.parameters()` group in the LLRD optimizer — no optimizer code change required. No hardcoded hyperparameters in `train.py`. No changes to `model.py`, `infra.py`, or `transforms.py`. 2-epoch smoke test passed.

**Verdict: APPROVED**

---

## idea009/design002 — 2026-04-09 — APPROVED

**Change:** `head_hidden = 384` (was 256) in `config.py` only.

All 16 config fields match the design spec exactly. `train.py` passes `head_hidden=args.head_hidden` to `SapiensPose3D`, which passes it to `Pose3DHead` as `hidden_dim`. The wider `input_proj`, `joint_queries`, and all 4 decoder layers are automatically included in the `model.head.parameters()` LLRD group — no optimizer code change required. No hardcoded hyperparameters in `train.py`. No changes to `model.py`, `infra.py`, or `transforms.py`. Minor: stale docstring reference to "idea004/design002" in `train.py` — cosmetic only. 2-epoch smoke test passed.

**Verdict: APPROVED**

---

## idea009/design003 — 2026-04-09 — APPROVED

**Change:** `_sinusoidal_init` static method added to `Pose3DHead` in `model.py`; `trunc_normal_` on `joint_queries` replaced with sinusoidal PE (Vaswani 2017). No other files changed beyond `output_dir` in `config.py`.

Formula and log-space numerics match the design spec exactly. `torch.no_grad()` wraps the `.copy_()` call — autograd graph clean. `trunc_normal_` preserved for regression heads. All 16 config fields match the spec. No new parameters introduced (70×256 embedding shape unchanged). 2-epoch smoke test passed; sanity check `joint_queries[0, :4] = [0.0, 1.0, 0.0, 1.0]` confirmed.

**Verdict: APPROVED**
