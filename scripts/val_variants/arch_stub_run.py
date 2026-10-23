"""Drive albench/arch_search/arch_autoresearch.py main() with a canned proposal instead of an LLM call (wiring tests only).
usage: arch_stub_run.py PATH_TO_arch_autoresearch.py [main args...]"""

import importlib.util
import random
import sys

import numpy as np
import torch

random.seed(12345)
np.random.seed(12345)
torch.manual_seed(
    12345
)  # arch_autoresearch builds the model BEFORE it seeds torch, so seed here or two identical runs differ in their init
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("arch_under_test", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
CANNED = """### IDEA: tiny conv stub
### WHY: wiring test
```python
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(4, 16, 9, padding=4)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.c(x)).mean(-1)
        return self.fc(h).squeeze(-1)
```
### TRAIN: {"optimizer":"adamw","lr":0.002,"weight_decay":1e-5,"batch_size":128,"epochs":3,"lr_schedule":"cosine"}
"""
mod.call_model = lambda s, u, *a, **k: CANNED
sys.argv = [path] + sys.argv[2:]
mod.main()
