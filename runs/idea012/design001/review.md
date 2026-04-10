# Design Review — idea012/design001

**Reviewer verdict: APPROVED**

## Summary

Single-knob regularization: increase `head_dropout` from 0.1 to 0.2. All other hyperparameters identical to idea004/design002.

## Evaluation

### Completeness
- Starting point: `runs/idea004/design002/`. Correct.
- Change is clearly isolated: only `head_dropout` from 0.1 to 0.2.
- Full table of all other unchanged parameters provided (gamma, LRs, weight_decay, drop_path, epochs, warmup, grad_clip, loss weights, batch size, accum steps). All values match idea004/design002 config.
- Implementation notes correctly state: only config.py change, no model/train/infra changes needed since dropout is a constructor argument.

### Mathematical Correctness
- No mathematical changes. Dropout rate is a hyperparameter passed to `nn.TransformerDecoderLayer`.

### Feasibility
- No VRAM impact from higher dropout. Trivially within 1080ti budget.

### Config Fields
- The design does not provide a standalone `config.py` fields block like the idea011 designs, but the config changes table is clear and unambiguous: change `head_dropout = 0.1` to `head_dropout = 0.2`, update `output_dir`. The Builder should have no ambiguity.

### Constraint Adherence
- LLRD schedule kept fixed (gamma=0.90, unfreeze_epoch=5). Correct per idea012 constraints.
- No data augmentation added. Correct.
- No model architecture changes. Correct.

## Issues Found
None. The design is minimal, well-specified, and leaves nothing for the Builder to guess.
