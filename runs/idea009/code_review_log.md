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

---

## idea009/design004 — 2026-04-09 — APPROVED

**Change:** `nn.ParameterList` of 4 scalar gates added to `Pose3DHead.__init__` in `model.py`; `forward` replaced with per-layer loop applying `sigmoid(gate) * memory` before each `TransformerDecoderLayer`. No other files changed beyond `output_dir` in `config.py`.

Gate initialization (`torch.tensor(4.6)`, sigmoid≈0.99) matches design spec. Per-layer loop replaces the single `self.decoder()` call correctly; 0-dim scalar broadcasts over `(B, S, hidden_dim)` correctly; `decoder.norm` guard preserved. `_init_weights` correctly does not touch gates. Gates in `nn.ParameterList` are auto-included in `lr_head` LLRD group — no optimizer changes needed. All 16 config fields match spec; gate init value (4.6) correctly kept internal to `model.py`. GPU mem 1.76/4.22GB, param count 308.8M, 2-epoch smoke test passed cleanly.

**Verdict: APPROVED**

---

## idea009/design005 — 2026-04-09 — APPROVED

**Change:** `self.output_norm = nn.LayerNorm(hidden_dim)` added to `Pose3DHead.__init__` in `model.py`; `out = self.output_norm(out)` inserted in `forward` immediately after the decoder and before `pelvis_token` extraction. No other files changed beyond `output_dir` in `config.py`.

All 16 config fields match the design spec exactly. `output_norm` is an attribute of `model.head` and therefore automatically included in the `lr_head` LLRD param group — no optimizer changes required. PyTorch default initialization (weight=1, bias=0) preserved; `_init_weights` correctly unchanged. Shape `(B, num_joints, hidden_dim)` is preserved through `LayerNorm(hidden_dim)`. Total parameter count 308.8M (512 new params, negligible). GPU mem 1.76/4.22 GB. 2-epoch smoke test passed cleanly with no errors.

**Verdict: APPROVED**
