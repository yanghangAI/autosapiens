# autoresearch — BEDLAM2 3D Pose

Autonomous research loop for the BEDLAM2 RGBD 3D pose project.
Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

---

## Setup

To set up a new research session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git -C /home/hangyang_umass_edu/MMC/sapiens checkout -b autoresearch/<tag>`
3. **Read the in-scope files** for full context (the repo uses mmengine):
   - `autoresearch/program.md` — you are reading it now.
   - `autoresearch/exp.py` — **the file you edit**. Full standalone MMEngine config.
   - `mmpose/models/heads/regression_heads/pose3d_transformer_head.py` — default head.
   - `mmpose/models/heads/regression_heads/pose3d_regression_head.py` — baseline head.
   - `mmpose/models/backbones/sapiens_rgbd.py` — 4-channel ViT backbone.
   - `pose/CLAUDE.md` — project conventions and gotchas (read before touching any code).
4. **Check data exists**: `ls /work/pi_nwycoff_umass_edu/hang/BEDLAM2subset/` — should show sequence directories. If missing, tell the human.
5. **Create the runs root**: `mkdir -p /work/pi_nwycoff_umass_edu/hang/MMC/autoresearch/runs/<tag>`
6. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row.
7. **Confirm and go**: confirm setup looks good, then start the loop.

---

## What the experiment does

Each experiment runs `exp.py` via SLURM on 1 × H100 GPU with a **fixed 5-epoch budget** on 100 training sequences / 20 val sequences. This makes every experiment directly comparable regardless of architecture or hyperparameter changes.

**The metric is `val_mpjpe_body` (mm) — lower is better.**
This is mean per-joint position error on 22 body joints in camera space (root-relative).

Typical baseline values at 5 epochs ≈ 110 mm (transformer head).

---

## What you CAN do

**Modify `exp.py`** — this is the primary file you edit. Fair game includes:
- Head architecture: `num_heads`, `dropout`, layer depth, attention type
- Switch head type: `Pose3dTransformerHead` ↔ `Pose3dRegressionHead`
- Optimizer: `lr`, `weight_decay`, `betas`, gradient clipping
- LR schedule: warmup length, cosine vs. linear, eta_min
- Loss weights: `loss_weight_depth`, `loss_weight_uv`, SmoothL1 `beta`
- Data augmentation: changes to `NoisyBBoxTransform` params, flip, etc.
- Batch size (be mindful of VRAM — H100 has 80 GB)
- Backbone `drop_path_rate`

**Modify model files** for architecture experiments:
- `mmpose/models/heads/regression_heads/pose3d_transformer_head.py`
- `mmpose/models/heads/regression_heads/pose3d_regression_head.py`
- (Backbone changes are riskier — proceed carefully)

## What you CANNOT do

- Change `num_epochs`, `train_max_seqs`, `val_max_seqs` — these are the fixed budget.
- Modify the evaluation metric (`mmpose/evaluation/metrics/bedlam_metric.py`).
- Modify the dataset or data loading infrastructure.
- Install new packages.
- Modify `autoresearch/submit_exp.sh` or `autoresearch/parse_result.sh`.

---

## Submitting and waiting for a job

```bash
# 1. Pick a work directory for this experiment (based on git hash)
COMMIT=$(git -C /home/hangyang_umass_edu/MMC/sapiens rev-parse --short HEAD)
WORK_DIR=/work/pi_nwycoff_umass_edu/hang/MMC/autoresearch/runs/<tag>/$COMMIT
mkdir -p "$WORK_DIR"

# 2. Submit
JOB_ID=$(sbatch \
    --output="$WORK_DIR/slurm.out" \
    --error="$WORK_DIR/slurm.err" \
    /home/hangyang_umass_edu/MMC/sapiens/pose/autoresearch/submit_exp.sh \
    "$WORK_DIR" \
    | awk '{print $NF}')
echo "Submitted job $JOB_ID"

# 3. Poll until done (check every 60 s)
while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q "$JOB_ID"; do
    echo "$(date): job $JOB_ID still running..."
    sleep 60
done
echo "Job $JOB_ID finished."
```

---

## Parsing results

```bash
bash /home/hangyang_umass_edu/MMC/sapiens/pose/autoresearch/parse_result.sh "$WORK_DIR"
# Prints: val_mpjpe_body: 109.42
```

If the output is empty or the script exits non-zero, the run crashed.
Inspect the log: `tail -n 80 "$WORK_DIR/slurm.out"` or
`find "$WORK_DIR" -name "*.log" | xargs tail -n 80`.

---

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv`
(tab-separated — do NOT use commas, they break descriptions).

```
commit	val_mpjpe_body	status	description
```

1. git commit hash (short, 7 chars)
2. `val_mpjpe_body` in mm, 2 decimal places — use `0.00` for crashes
3. status: `keep`, `discard`, or `crash`
4. short description of what this experiment tried

Example:
```
commit	val_mpjpe_body	status	description
a1b2c3d	112.34	keep	baseline (transformer head, 5ep, max_seqs=50)
b2c3d4e	108.91	keep	increase head num_heads from 8 to 16
c3d4e5f	115.02	discard	switch to regression head (worse than transformer)
d4e5f6g	0.00	crash	double depth of transformer decoder (OOM)
```

**Do not commit results.tsv** — leave it untracked.

---

## The experiment loop

The experiment runs on `autoresearch/<tag>` branch.

LOOP FOREVER:

1. Check git state: `git -C /home/hangyang_umass_edu/MMC/sapiens log --oneline -3`
2. Pick an experimental idea. Edit `exp.py` (and optionally a model file).
3. `git -C /home/hangyang_umass_edu/MMC/sapiens add -p && git commit -m "ar: <description>"`
4. Submit the job and poll until done (see "Submitting" above).
5. Parse results: `bash autoresearch/parse_result.sh "$WORK_DIR"`
6. If crash: inspect log, attempt quick fix and re-run. After 2 failed attempts, log as crash and revert.
7. Log to `results.tsv`.
8. **If val_mpjpe_body improved (lower)**: keep the git commit — the branch advances.
9. **If val_mpjpe_body is equal or worse**: `git -C /home/hangyang_umass_edu/MMC/sapiens reset --hard HEAD~1`

**Simplicity criterion**: all else equal, simpler is better. A 1 mm gain from 20 lines of
hacky code is not worth it. Removing code and matching or beating the baseline? Always keep.

**Crashes**: if the idea is fundamentally broken (OOM from a huge model, wrong tensor shape),
log as crash, revert, move on. If it's a trivial fix (typo, import), fix and re-run.

**Timeout**: if a SLURM job runs longer than 90 minutes (the 2-hour limit with headroom),
`scancel $JOB_ID`, treat as crash, revert.

**NEVER STOP**: once the loop begins, do NOT pause to ask the human whether to continue.
The human expects you to run indefinitely until manually stopped. If you run out of obvious
ideas, try combining near-misses, try more radical architectural changes, re-read the head
and backbone code for inspiration. The loop runs until you are interrupted.

---

## Key background (read before starting)

- Coordinate system: BEDLAM2 camera space is **X=forward (depth), Y=left, Z=up** —
  differs from OpenCV. Projection: `u = fx·(-Y/X) + cx`, `v = fy·(-Z/X) + cy`.
- 70 active joints: 22 body + 2 eyes + 30 hands + 16 surface. Defined in
  `mmpose/datasets/datasets/body3d/constants.py`.
- `mpjpe/body/val` = MPJPE on the 22 body joints only (most important metric).
- The transformer head (`Pose3dTransformerHead`) uses per-joint query tokens + cross-attention
  over the backbone feature map. It currently outperforms the regression head.
- `persistent_workers=False` is required — NPZ/mmap file descriptor issue; do NOT change.
- MMEngine gotcha: val metainfo fields are flattened to the top level dict in `metric.process()`.
  Read as `data_sample['K']`, not `data_sample['metainfo']['K']`.
