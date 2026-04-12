# Review: idea018 / design004 — EMA (decay=0.999) + Last-Epoch Polish Pass

**Verdict: APPROVED (with flagged implementation risk)**

## Summary
Sound design concept. The EMA phase is identical to design001 and is correct. The polish pass correctly loads EMA weights, re-enables gradients, and runs one flat-LR epoch. One **implementation risk** is flagged regarding `iter_metrics.csv` append semantics — the Builder must handle this explicitly.

## Checklist

### Correctness
- [x] EMA phase (epochs 0–19) is identical to design001. All correctness points from design001 apply here.
- [x] Polish pass: `model.load_state_dict(ema_model.state_dict())` loads averaged weights into the live model. Correct.
- [x] Polish pass: `for p in model.parameters(): p.requires_grad_(True)` re-enables gradients after EMA copy. Necessary and correct.
- [x] New `polish_optimizer = AdamW(model.parameters(), lr=polish_lr, weight_decay=weight_decay)` — fresh optimizer state for the polish epoch. Correct; using the old LLRD optimizer state at LR=1e-6 would give incorrect per-group scales.
- [x] Validation at epoch 21 runs on `model` (post-polish live model). Correct.
- [x] Checkpoint `polished.pth` saves post-polish live model and `ema_pre_polish` key for the EMA weights. Correct.
- [x] `epochs = 20` stays unchanged in config; polish is a post-loop block. Correct.

### Configuration Completeness
- [x] `ema_decay = 0.999` added.
- [x] `polish_lr = 1e-6` added.
- [x] `polish_epochs = 1` added.
- [x] `output_dir` updated to `runs/idea018/design004`.
- [x] All other fields inherited from idea014/design003.

### Architecture / Loss / Optimizer
- [x] No architecture or loss changes.

### Memory Budget
- [x] EMA copy ≈ 345 MB during epochs 0–19.
- [x] Design recommends `del ema_model; torch.cuda.empty_cache()` after loading into live model — reduces peak VRAM during polish pass.
- [x] Polish AdamW state ≈ 2× params × 4 bytes ≈ 690 MB. Within 11 GB.

### 20-Epoch Proxy Limit
- [x] Main loop = 20 epochs. Polish adds 1 extra epoch (5% compute overhead). The idea.md explicitly sanctions this as "cheap". Acceptable.

### Sanity Checks
- [x] `polish_epochs = 0`: skip polish → identical to design001. Valid.
- [x] `ema_decay = 0`: EMA = live model always → polish starts from final live weights. Valid.
- [x] `polish_lr = 0` (effectively): no learning → polished weights ≈ EMA weights → recovers design001 result. Valid.

## Flagged Implementation Risk: `iter_metrics.csv` Append Mode

The design correctly identifies that `metrics.csv` must be opened in append mode for the polish epoch and provides a `csv.DictWriter` workaround. However, it does **not** address the same issue for `iter_metrics.csv`:

```python
iter_logger_polish = IterLogger(str(out_dir / "iter_metrics.csv"))
```

If `IterLogger` always writes a header row on open (which is typical), this will either:
1. Overwrite the existing `iter_metrics.csv` (losing all 20 epochs of per-iteration data), or
2. Append a duplicate header mid-file, corrupting the CSV that the pipeline reads.

**Builder instruction:** The Builder must check `IterLogger.__init__` in `infra.py`. If it writes headers unconditionally, use the same append+no-header pattern as the `metrics.csv` workaround:
- Either open `iter_metrics.csv` with `mode='a'` and skip header writing, or
- Write the polish iteration rows manually with `csv.writer` (appending to the existing file).

This is a fixable implementation detail, not a fundamental design flaw. The rest of the spec is unambiguous and correct.

## Decision
**APPROVED.** Builder must resolve the `iter_metrics.csv` append-mode issue as described above.
