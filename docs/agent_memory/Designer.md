2026-04-09: Drafted idea008/design001 as continuous interpolated depth positional encoding from runs/idea005/design001. Stopped after single design draft for reviewer handoff.
2026-04-09: Fixed idea008/design001 review issue by moving interpolation/backbone-forward responsibilities explicitly to code/model.py and limiting code/train.py to optimizer wiring only if needed.
2026-04-09: Began idea009 (Head Architecture Refinement). 5 designs planned.
  - design001: 6-layer decoder (head_num_layers=6), all else from idea004/design002. APPROVED.
  - design002: Wide head (head_hidden=384, head_num_heads=8, head_num_layers=4). APPROVED.
  - design003: Sine-cosine joint query init (head_num_layers=4, head_hidden=256; sinusoidal init replaces trunc_normal_ in _init_weights). APPROVED.
  - design004: Per-layer input feature gate (head_num_layers=4, head_hidden=256; nn.ParameterList of 4 scalar sigmoid gates on memory, init=4.6 so sigmoid≈0.99; per-layer decoder loop in forward). Awaiting review.
