"""How a cell identity enters an architecture, and the controls that say whether it did.

Every scheme here turns a cell type into a vector ``z`` and then uses ``z`` to modulate a
backbone that is otherwise the study's ordinary student. They differ only in where ``z``
comes from, which is exactly the axis we want to measure:

  ``agnostic``    ``z`` is nothing; the model predicts the cross-cell mean. The control that
                  matters most, because the cell lines correlate at r about 0.7-0.8 and a
                  mean-predictor is therefore a strong baseline on absolute activity. A
                  conditioning scheme has to beat it on the cell-type DIFFERENCE.
  ``onehot``      ``z`` is a one-hot. Capacity of a per-cell head with a shared trunk.
  ``embed``       ``z`` is a free learned embedding. This is the CEILING, not a competitor:
                  it can memorise anything cell-specific, so the gap to ``text`` is what the
                  description loses as a parameterisation.
  ``text``        ``z`` is a frozen sentence embedding of the cell's description. The only
                  scheme besides ``structured`` that can reach a cell type with no labels.
  ``text_shuffled`` ``text`` with the descriptions permuted across cells. If held-out
                  performance survives this, the model is not reading the text and the
                  result is a mean-profile prediction wearing a description. Mandatory
                  alongside ``text`` for the same reason ``badge_kmeanspp_only`` is
                  mandatory alongside BADGE.
  ``structured``  ``z`` is a handful of interpretable categorical facts. Says how much of
                  the text route is carried by lineage and tissue alone.
  ``hybrid``      ``text`` plus a free per-cell residual, penalised toward zero. The residual
                  norm after training measures what the description failed to say.

Modulation is FiLM, applied either at the readout only (``depth="readout"``: ``z`` rescales
and shifts the penultimate feature vector, so the trunk stays cell-agnostic and only the
feature weighting is cell-specific) or throughout (``depth="deep"``: every normalisation
layer in the trunk gets its own per-channel gain and bias from ``z``). Both are attached
with hooks and neither needs the backbone to know it is being conditioned, so the menu
architectures, the LegNet reference and an oracle trunk all take the same treatment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def can_describe_novel(mode: str) -> bool:
    """Can this mode produce a conditioning vector for a cell it never trained on?

    This is not a nicety, it decides whether a held-out-cell score means anything. `embed`
    gives every cell a free row initialised at std=0.02, and `onehot` gives it a column of
    a trained projection; a held-out cell's row or column receives no gradient, so at
    evaluation it is still an arbitrary small vector. Scoring through it does not measure
    zero-shot transfer -- it measures whatever that arbitrary direction happens to do -- and
    the number comes out looking like a real result. `agnostic` reads no cell at all, so its
    held-out score is the cross-cell mean: a legitimate floor, not a transfer claim.

    Only the modes that build their vector from a DESCRIPTION can be asked about a cell they
    have never seen.
    """
    return mode in ("text", "text_shuffled", "structured", "hybrid")


CONDITIONING_MODES = (
    "agnostic",
    "onehot",
    "embed",
    "text",
    "text_shuffled",
    "structured",
    "hybrid",
)
DEPTHS = ("readout", "deep")


class _CellEncoder(nn.Module):
    """Cell index -> conditioning vector. Frozen sources are buffers, free ones parameters."""

    def __init__(
        self,
        mode: str,
        cells: list[str],
        d_cond: int = 32,
        *,
        seed: int = 0,
        source: np.ndarray | None = None,
    ):
        super().__init__()
        if mode not in CONDITIONING_MODES:
            raise ValueError(f"mode must be one of {CONDITIONING_MODES}, got {mode!r}")
        self.mode = mode
        self.cells = list(cells)
        self.d_cond = int(d_cond)
        n = len(cells)

        if mode == "agnostic":
            self.register_buffer("table", torch.zeros(n, 1))
            self.proj: nn.Module = nn.Identity()
            self.d_out = 0
            return

        if mode == "onehot":
            table = np.eye(n, dtype=np.float32)
        elif mode in ("text", "text_shuffled", "structured"):
            if source is None:
                raise ValueError(f"mode {mode!r} needs its `source` matrix passed in")
            table = np.asarray(source, dtype=np.float32)
            if mode == "text_shuffled":
                # Permute WHICH description goes to which cell. A derangement where one
                # exists, so no cell keeps its own text and the control cannot pass by
                # accident on a cell that happened to be a fixed point.
                rng = np.random.default_rng(seed)
                perm = np.arange(n)
                for _ in range(64):
                    perm = rng.permutation(n)
                    if n < 2 or not (perm == np.arange(n)).any():
                        break
                table = table[perm]
                self.register_buffer("perm", torch.as_tensor(perm))
        elif mode in ("embed", "hybrid"):
            table = None
        else:  # pragma: no cover - CONDITIONING_MODES is exhaustive
            raise AssertionError(mode)

        if mode == "embed":
            self.free = nn.Embedding(n, self.d_cond)
            nn.init.normal_(self.free.weight, std=0.02)
            self.proj = nn.Identity()
        elif mode == "hybrid":
            if source is None:
                raise ValueError("hybrid needs the text matrix as `source`")
            self.register_buffer("table", torch.as_tensor(np.asarray(source, np.float32)))
            self.proj = nn.Linear(self.table.shape[1], self.d_cond)
            self.free = nn.Embedding(n, self.d_cond)
            nn.init.zeros_(self.free.weight)
        else:
            self.register_buffer("table", torch.as_tensor(table))
            self.proj = nn.Linear(self.table.shape[1], self.d_cond)
        self.d_out = self.d_cond

    def residual_penalty(self) -> torch.Tensor:
        """L2 of the free residual; zero for every mode that has none."""
        if self.mode == "hybrid":
            return self.free.weight.pow(2).mean()
        return torch.zeros((), device=next(self.parameters()).device)

    def vectors(self) -> torch.Tensor:
        """The conditioning vector for every known cell, (n_cells, d_out)."""
        if self.mode == "agnostic":
            return self.table[:, :0]
        if self.mode == "embed":
            return self.free.weight
        z = self.proj(self.table)
        if self.mode == "hybrid":
            z = z + self.free.weight
        return z

    def forward(self, cell_idx: torch.Tensor) -> torch.Tensor:
        return self.vectors()[cell_idx]

    def encode_novel(self, source_row: np.ndarray) -> torch.Tensor:
        """Conditioning vector for a cell never trained on, from its description vector.

        Only the schemes with a frozen source can do this; ``embed`` and ``onehot`` raise,
        which is the point of running them -- they mark where generalisation is impossible
        rather than merely poor. ``hybrid`` contributes no residual for an unseen cell,
        which is the honest zero-shot behaviour.
        """
        if self.mode in ("agnostic", "onehot", "embed"):
            raise NotImplementedError(
                f"mode {self.mode!r} cannot reach a cell type it was not trained on"
            )
        w = next(self.proj.parameters())
        x = torch.as_tensor(np.asarray(source_row, np.float32), device=w.device).reshape(1, -1)
        return self.proj(x)


class _FiLM(nn.Module):
    """Per-channel gain and bias produced from the conditioning vector."""

    def __init__(self, d_cond: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(d_cond, 2 * channels)
        # Start at the identity so an untrained conditioner cannot damage a good backbone,
        # and so `agnostic` and an all-zero z give exactly the unconditioned model.
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
        self.channels = channels

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gb = self.to_gb(z)
        g, b = gb[:, : self.channels], gb[:, self.channels :]
        if h.dim() == 3:  # (B, C, L)
            return h * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)
        return h * (1.0 + g) + b


class ConditionedModel(nn.Module):
    """A backbone plus a conditioning scheme, with FiLM attached by hooks.

    The backbone is used unmodified and still returns a scalar per sequence; conditioning is
    injected by hooks so that a menu architecture, the LegNet reference and an oracle trunk
    are all wrapped the same way and nothing in the architecture search has to know.
    """

    def __init__(
        self,
        net: nn.Module,
        cells: list[str],
        mode: str = "text",
        *,
        depth: str = "deep",
        d_cond: int = 32,
        source: np.ndarray | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}, got {depth!r}")
        self.net = net
        self.cells = list(cells)
        self.mode = mode
        self.depth = depth
        self.encoder = _CellEncoder(mode, cells, d_cond, seed=seed, source=source)
        self._z: torch.Tensor | None = None
        self.films = nn.ModuleList()

        if mode == "agnostic":
            return  # nothing to attach; this is the unconditioned control

        readouts = [m for m in net.modules() if isinstance(m, nn.Linear) and m.out_features == 1]
        if not readouts:
            raise ValueError("backbone has no final Linear(*, 1) readout to condition")
        last = readouts[-1]

        film_r = _FiLM(self.encoder.d_out, last.in_features)
        self.films.append(film_r)

        def _pre(_m: nn.Module, inp: tuple[Any, ...]):
            h = inp[0]
            if self._z is None or h.dim() != 2:
                return None
            return (film_r(h, self._z),) + tuple(inp[1:])

        last.register_forward_pre_hook(_pre)

        if depth == "deep":
            norms = [
                m
                for m in net.modules()
                if isinstance(m, nn.BatchNorm1d | nn.GroupNorm | nn.InstanceNorm1d)
            ]
            for m in norms:
                ch = getattr(m, "num_features", None) or getattr(m, "num_channels", None)
                if not ch:
                    continue
                film = _FiLM(self.encoder.d_out, int(ch))
                self.films.append(film)

                def _post(_mod, _inp, out, _film=film):
                    if self._z is None or out.dim() != 3:
                        return out
                    return _film(out, self._z)

                m.register_forward_hook(_post)

    def forward(
        self,
        x: torch.Tensor,
        cell_idx: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``z`` overrides ``cell_idx`` so a held-out cell can be scored from its description
        alone, with no index for it anywhere in the model."""
        if self.mode == "agnostic":
            self._z = None
        elif z is not None:
            self._z = z if z.shape[0] == x.shape[0] else z.expand(x.shape[0], -1)
        elif cell_idx is not None:
            self._z = self.encoder(cell_idx)
        else:
            raise ValueError("pass cell_idx or z")
        try:
            return self.net(x).reshape(-1)
        finally:
            self._z = None


def wrap(
    net: nn.Module,
    cells: list[str],
    mode: str = "text",
    *,
    depth: str = "deep",
    d_cond: int = 32,
    seed: int = 0,
) -> ConditionedModel:
    """Build the conditioning source each mode needs, then wrap."""
    from albench.celltype import descriptions as D

    source = None
    if mode in ("text", "text_shuffled", "hybrid"):
        source = D.text_embeddings(cells)
    elif mode == "structured":
        source = D.structured_matrix(cells)
    return ConditionedModel(net, cells, mode, depth=depth, d_cond=d_cond, source=source, seed=seed)
