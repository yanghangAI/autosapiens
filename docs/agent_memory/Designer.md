2026-04-09: Drafted idea008/design001 as continuous interpolated depth positional encoding from runs/idea005/design001. Stopped after single design draft for reviewer handoff.
2026-04-09: Fixed idea008/design001 review issue by moving interpolation/backbone-forward responsibilities explicitly to code/model.py and limiting code/train.py to optimizer wiring only if needed.
2026-04-09: Began idea009 (Head Architecture Refinement). 5 designs planned.
  - design001: 6-layer decoder (head_num_layers=6), all else from idea004/design002. APPROVED.
  - design002: Wide head (head_hidden=384, head_num_heads=8, head_num_layers=4). APPROVED.
  - design003: Sine-cosine joint query init (head_num_layers=4, head_hidden=256; sinusoidal init replaces trunc_normal_ in _init_weights). APPROVED.
  - design004: Per-layer input feature gate (head_num_layers=4, head_hidden=256; nn.ParameterList of 4 scalar sigmoid gates on memory, init=4.6 so sigmoid≈0.99; per-layer decoder loop in forward). Awaiting review.
2026-04-10: Drafted idea010 (Multi-Scale Backbone Feature Aggregation). 5 designs, all from runs/idea004/design002.
  - design001: Last-4-layer concat + Linear(4096,1024). Xavier init. ~4.2M new params. LLRD unchanged.
  - design002: Learned layer weights -- 4 softmax scalars (init uniform). Only 4 new params. LLRD unchanged.
  - design003: Feature pyramid 3 scales (layers 7,15,23). Three Linear(1024,256) + Linear(768,1024). ~1.57M params.
  - design004: Cross-scale attention gate (layers 11,23). sigmoid(Linear(1024,1)) * (1+gate). bias_init=-5.0. ~1K params.
  - design005: Alternating layer average (12 even-spaced layers). Running-sum approach. 0 new params.
  Key note: Sapiens 0.3B has 24 blocks (not 12). Corrected all layer indices from idea.md accordingly.
2026-04-10: Drafted idea011 (LLRD + Continuous Depth PE Combination). 4 designs.
  - design001: LLRD gamma=0.90, unfreeze=5, from idea008/design003 (sqrt depth PE). Direct combo.
  - design002: LLRD gamma=0.85, unfreeze=5, from idea008/design003. More aggressive decay.
  - design003: LLRD gamma=0.90, unfreeze=10, from idea008/design003. Later unfreezing.
  - design004: LLRD gamma=0.90, unfreeze=5, from idea008/design002 (gated depth PE). Gate + LLRD.
  All designs: model.py unchanged, only train.py gets LLRD per-block groups + freeze/unfreeze.
2026-04-10: Drafted idea012 (Regularization for Generalization). 5 designs, all from runs/idea004/design002.
  - design001: Head dropout 0.2 (head_dropout=0.2). Config-only change.
  - design002: Weight decay 0.3 (weight_decay=0.3). Config-only change.
  - design003: Stochastic depth 0.2 (drop_path=0.2). Config-only change.
  - design004: R-Drop consistency (rdrop_alpha=1.0). New config field + train.py loop change (second no_grad forward pass, MSE on body joints).
  - design005: Combined regularization (head_dropout=0.2, weight_decay=0.2, drop_path=0.2). Config-only change.
