"""The validator's input-dependence probe is relative to the output scale (audit of Nov 2, F2): a network whose outputs are tiny at initialisation is not 'independent of its input'."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from albench.arch_search.validate import build_and_check  # noqa: E402

TINY_HEAD = """
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 8, 5, padding=2)
        self.head = nn.Linear(8, 1)
        nn.init.normal_(self.head.weight, 0, 1e-6)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(-1)).squeeze(-1)
"""

IGNORES_INPUT = """
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.b.expand(x.shape[0])
"""


def test_tiny_outputs_are_accepted():
    _, verdict = build_and_check(TINY_HEAD)
    assert verdict.ok, verdict.reason


def test_a_network_that_ignores_its_input_is_still_rejected():
    _, verdict = build_and_check(IGNORES_INPUT)
    assert not verdict.ok and "does not depend on input" in verdict.reason
