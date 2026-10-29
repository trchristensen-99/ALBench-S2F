"""Auxiliary-target training for any menu architecture: attach extra linear heads to the input of the architecture's final ``Linear(*, 1)`` readout.

Why: a student trained on the K562 oracle label can also be asked to predict the HepG2 oracle label of the same sequences, which shares regulatory grammar and may regularise the shared trunk. Menu architectures are black
boxes that return one scalar, so the heads are attached generically with a forward pre-hook on the last ``nn.Linear`` with one output. In training mode the wrapper returns ``(B, 1 + n_aux)`` (primary first); in
evaluation mode only the primary ``(B,)``, so validation, early stopping and every prediction are unchanged. ``weight`` scales the auxiliary squared error: the head output is multiplied by sqrt(weight) and the
auxiliary target must be scaled by the same factor by the caller.
"""

from __future__ import annotations


def _module():
    import torch
    import torch.nn as nn

    class MultiHead(nn.Module):
        def __init__(self, net, n_aux: int = 1, weight: float = 0.5):
            super().__init__()
            self.net = net
            lin = [m for m in net.modules() if isinstance(m, nn.Linear) and m.out_features == 1]
            if not lin:
                raise ValueError("no final Linear(*, 1) readout to attach an auxiliary head to")
            self._last = lin[-1]
            self.aux = nn.Linear(self._last.in_features, n_aux)
            self.scale = float(weight) ** 0.5
            self._feat = None
            self._last.register_forward_pre_hook(lambda _m, inp: setattr(self, "_feat", inp[0]))

        def forward(self, x):
            y = self.net(x)
            if not self.training:
                return y
            f = self._feat
            if f is None or f.dim() != 2:
                raise ValueError("the final readout does not take a (batch, features) input")
            return torch.cat([y.reshape(-1, 1), self.aux(f) * self.scale], dim=1)

    return MultiHead


def wrap(net, n_aux: int = 1, weight: float = 0.5):
    return _module()(net, n_aux, weight)
