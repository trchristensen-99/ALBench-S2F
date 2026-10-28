# Model contract (CONTRACT.md)

Write `model.py` with exactly this interface and a `card.json` describing it.

```python
import torch, torch.nn as nn
HPARAMS = {"width": 256, "depth": 6}                # architecture defaults
TUNABLE = {"width": [128, 512], "depth": [4, 10]}   # optional, at most 4 keys, [low, high]
def build_model(out_dim: int, **hparams) -> nn.Module:
    """forward(x: FloatTensor[B, 4, 200]) -> FloatTensor[B, out_dim]"""
```

Rules checked automatically before any training:
- Pure model code. Imports are limited to: torch, math, typing, functools, collections, dataclasses. No file, network, process or environment
  access, no eval or exec, no weight loading.
- No training logic. No loss, optimizer, data loading, augmentation or reverse-complement averaging inside the model.
- `out_dim` is given, not chosen: 1 for regression, 2 for a mean and log-variance head, n_bins for binned classification.
- At most 40,000,000 trainable parameters.
- Device-agnostic and batch-safe. No hard-coded device. Works for any batch size in eval mode, including 1. In eval mode the output for one
  sequence must not depend on the other sequences in the batch.
- Uses the whole insert: the output must depend on at least 90% of the 200 input positions.
- Should run under torch.autocast; if it cannot, say so in the card (`"amp_safe": false`).

`card.json` keys:

```json
{
  "family_claimed": "transformer_hybrid",
  "summary": "one line describing the model",
  "hypothesis": "why this should work on this data",
  "what_changed": "for a refinement: the one structural change; otherwise empty",
  "parent_code_hash": "hash of the parent if refining, else empty",
  "suggested_recipe": {"lr": 0.0005, "warmup_frac": 0.1, "weight_decay": 0.05, "batch_size": 256},
  "recipe_rationale": "why this model needs these training settings",
  "expected_failure": "one way this model could fail",
  "amp_safe": true,
  "length_agnostic": false
}
```

`selfcheck.py` in the working directory imports `model.py`, builds it with out_dim 1 and 2, runs a forward pass on a random batch of shape
(2, 4, 200) and prints the output shape and the parameter count. Run it before you finish. It is a smoke test: the real checks are broader.
