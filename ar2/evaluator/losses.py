"""Loss registry (spec 4.2, 13.1). Losses live here, not in candidates. A loss maps the model outputs to a scalar, and ``predict`` maps outputs to the
z-scored point prediction (for the hetero and binned losses that is the mean / expected bin).

Multi-task: the model's ``out_dim`` is ``loss.out_dim * T``; outputs are reshaped to (B, T, out_dim) and the per-task losses are averaged. Fitness always
uses task 0.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class Loss(Protocol):
    out_dim: int

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

    def predict(self, outputs: torch.Tensor) -> torch.Tensor: ...


class MSELoss:
    out_dim = 1

    def __call__(self, outputs, targets):
        return F.mse_loss(outputs[:, 0], targets)

    def predict(self, outputs):
        return outputs[:, 0]


class HuberLoss:
    out_dim = 1

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, outputs, targets):
        return F.huber_loss(outputs[:, 0], targets, delta=self.delta)

    def predict(self, outputs):
        return outputs[:, 0]


class HeteroLoss:
    """Gaussian negative log-likelihood with a mean and a log-variance output (out_dim 2)."""

    out_dim = 2

    def __call__(self, outputs, targets):
        mu, logvar = outputs[:, 0], outputs[:, 1].clamp(-8.0, 8.0)
        return (0.5 * (logvar + (targets - mu) ** 2 * torch.exp(-logvar))).mean()

    def predict(self, outputs):
        return outputs[:, 0]


class CorrLoss:
    """MSE plus ``weight`` times (1 - Pearson r) inside the batch."""

    out_dim = 1

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def __call__(self, outputs, targets):
        p = outputs[:, 0]
        mse = F.mse_loss(p, targets)
        pc, tc = p - p.mean(), targets - targets.mean()
        den = torch.sqrt((pc * pc).sum() * (tc * tc).sum()) + 1e-8
        return mse + self.weight * (1.0 - (pc * tc).sum() / den)

    def predict(self, outputs):
        return outputs[:, 0]


class BinnedLoss:
    """Cross-entropy against Gaussian-smoothed targets over ``n_bins`` bins spanning the z-scored label range; the prediction is the expected bin value."""

    def __init__(
        self, n_bins: int = 32, z_range: tuple[float, float] = (-4.0, 6.0), sigma_bins: float = 1.0
    ):
        self.out_dim = n_bins
        lo, hi = z_range
        self.centers = torch.linspace(lo, hi, n_bins)
        self.sigma = sigma_bins * (hi - lo) / max(n_bins - 1, 1)

    def _c(self, ref):
        return self.centers.to(ref.device, ref.dtype)

    def __call__(self, outputs, targets):
        c = self._c(outputs)
        t = torch.softmax(-((targets[:, None] - c[None, :]) ** 2) / (2 * self.sigma**2), dim=1)
        return -(t * F.log_softmax(outputs.float(), dim=1)).sum(1).mean()

    def predict(self, outputs):
        c = self._c(outputs)
        return (torch.softmax(outputs.float(), dim=1) * c[None, :]).sum(1)


class MultiTask:
    """Apply a per-task loss to a (B, T * out_dim) output; ``predict`` returns (B,) for one task and (B, T) otherwise."""

    def __init__(self, base: Loss, n_tasks: int):
        self.base, self.T = base, n_tasks
        self.out_dim = base.out_dim * n_tasks

    def _split(self, outputs):
        return outputs.reshape(outputs.shape[0], self.T, self.base.out_dim)

    def __call__(self, outputs, targets):
        if self.T == 1:
            return self.base(outputs, targets.reshape(-1))
        o = self._split(outputs)
        tt = targets.reshape(targets.shape[0], self.T)
        return sum(self.base(o[:, t], tt[:, t]) for t in range(self.T)) / self.T

    def predict(self, outputs):
        if self.T == 1:
            return self.base.predict(outputs)
        o = self._split(outputs)
        return torch.stack([self.base.predict(o[:, t]) for t in range(self.T)], 1)


_FACTORIES = {
    "mse": lambda cfg, zr: MSELoss(),
    "huber": lambda cfg, zr: HuberLoss(float(cfg.get("delta", 1.0))),
    "hetero": lambda cfg, zr: HeteroLoss(),
    "corr": lambda cfg, zr: CorrLoss(float(cfg.get("weight", 0.5))),
    "binned": lambda cfg, zr: BinnedLoss(
        int(cfg.get("n_bins", 32)), zr, float(cfg.get("sigma_bins", 1.0))
    ),
}


def register_loss(name: str, factory) -> None:
    """Add a loss (used by PROCEDURE plug-ins). ``factory(cfg: dict, z_range) -> Loss``."""
    _FACTORIES[name] = factory


def build_loss(
    loss_cfg: dict | str | None, n_tasks: int = 1, z_range: tuple[float, float] = (-4.0, 6.0)
) -> MultiTask:
    """Build the (multi-task wrapped) loss for a recipe's ``loss`` field, e.g. ``{"name": "mse"}``."""
    cfg = {"name": loss_cfg} if isinstance(loss_cfg, str) else dict(loss_cfg or {"name": "mse"})
    name = cfg.get("name", "mse")
    if name not in _FACTORIES:
        raise KeyError(f"unknown loss {name!r}; known: {sorted(_FACTORIES)}")
    return MultiTask(_FACTORIES[name](cfg, z_range), n_tasks)


def loss_names() -> list[str]:
    return sorted(_FACTORIES)
