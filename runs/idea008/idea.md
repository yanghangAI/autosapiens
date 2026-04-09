# Continuous Depth Positional Encoding

**Expected Designs:** 4

## Starting Point

The baseline starting point for this idea is:

`runs/idea005/design001/code/`

That design delivered the best completed validation result so far in `results.csv`
(`val_mpjpe_weighted = 121.4 mm`) by replacing the standard 2D positional embedding with
row + column + depth-bucket positional embeddings. This idea keeps that core structure
but refines how depth is encoded so nearby depths do not collapse into the same hard bin.

## Concept

The completed experiments suggest that explicit depth-aware positional structure is the
strongest architectural improvement discovered so far, while more disruptive attention or
curriculum changes did not help. The next promising step is to make the depth positional
signal smoother and better calibrated rather than replacing it outright.

## Broader Reflection

### Strong prior results to build on

- `idea005/design001` (`121.4 mm`) is the best completed result overall. Explicit
  depth-aware positional information appears genuinely useful and low-risk.
- `idea004/design002` (`130.7 mm`) showed that careful, incremental refinement of a
  strong backbone often beats bigger architectural departures.

### Patterns to avoid

- `idea005/design002` and `idea005/design003` performed much worse than the winning
  bucketed-depth PE design, suggesting that large departures from the successful depth-PE
  formulation are risky.
- `idea002` and `idea003` indicate that heavy changes to attention structure or training
  dynamics can underperform within the 20-epoch proxy budget.
- `idea006` and `idea007` are still in flight, so this idea should not depend on their
  incomplete results.

## Search Axes

### Category A — Exploit & Extend

1. Replace hard depth bucket lookup from `idea005/design001` with **continuous linear
   interpolation** between neighboring depth embeddings, preserving the same row/column
   decomposition while reducing quantization error.
2. Add a **learned residual gate** on the depth positional term so the model can tune how
   strongly depth PE influences the pretrained row/column positional structure.
3. Change the depth discretization from uniform linear spacing to **near-emphasized
   spacing** (for example log-depth or square-root depth) while keeping the rest of the
   winning `idea005/design001` architecture intact.

Each of these axes derives directly from the strongest completed result,
`idea005/design001`.

### Category B — Novel Exploration

4. Use a **hybrid two-resolution depth PE**: a coarse global depth code plus a fine local
   interpolated depth code at each patch. This has not been tried in any previous run and
   remains lightweight enough for a 20-epoch 1080ti proxy run.

## Expected Designs

The Designer should generate **4** novel designs:

1. Continuous interpolated depth PE with the same 16-bin support as `idea005/design001`
2. Continuous interpolated depth PE plus a learned residual gate on the depth term
3. Continuous interpolated depth PE with near-emphasized depth spacing
4. Hybrid two-resolution depth PE (coarse + fine depth encodings)

## Design Constraints

- Keep the successful row + column positional decomposition from
  `runs/idea005/design001/code/`
- Keep the ViT backbone and decoder head size within the existing 1080ti budget
- Avoid introducing large attention-bias tensors or expensive pairwise token operations
- Specify all config values explicitly so the Builder can implement each variant without
  guessing
