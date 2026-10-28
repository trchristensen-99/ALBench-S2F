"""Training-procedure plug-ins (spec 13.5): loss plug-ins that follow the Loss protocol, and training callbacks (``on_step_end`` / ``on_epoch_end``).

A plug-in is a Python file defining either ``make_loss(cfg, z_range) -> Loss`` or a ``Callbacks`` class. It must pass :func:`gate_plugin` (finite loss, non-zero
gradients, right predict shape) before any recipe may use it. Plug-ins are loaded and run only inside evaluator or gate subprocesses.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .losses import register_loss


@dataclass
class PluginGate:
    ok: bool
    detail: str = ""


def load_plugin(path: str | Path):
    """Import a plug-in file as a module (never call this in the orchestrator process)."""
    spec = importlib.util.spec_from_file_location(f"ar2_plugin_{Path(path).stem}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def gate_plugin(path: str | Path, name: str, z_range=(-4.0, 6.0)) -> PluginGate:
    """Finite loss, non-zero gradients and the right ``predict`` shape on random outputs."""
    import torch

    try:
        mod = load_plugin(path)
        if not hasattr(mod, "make_loss"):
            return PluginGate(False, "plug-in defines no make_loss(cfg, z_range)")
        loss = mod.make_loss({"name": name}, z_range)
        out = torch.randn(32, loss.out_dim, requires_grad=True)
        tgt = torch.randn(32)
        val = loss(out, tgt)
        if val.ndim != 0 or not torch.isfinite(val):
            return PluginGate(False, f"loss is not a finite scalar: {val}")
        val.backward()
        if (
            out.grad is None
            or not torch.isfinite(out.grad).all()
            or float(out.grad.abs().sum()) == 0.0
        ):
            return PluginGate(False, "gradient is zero or non-finite")
        p = loss.predict(out.detach())
        if tuple(p.shape) != (32,):
            return PluginGate(False, f"predict shape {tuple(p.shape)}, expected (32,)")
        return PluginGate(True)
    except Exception as e:  # noqa: BLE001
        return PluginGate(False, f"{type(e).__name__}: {e}")


def install_plugin(path: str | Path, name: str) -> None:
    """Register a gated loss plug-in under ``name`` so recipes can say ``loss: {name: <name>}``."""
    mod = load_plugin(path)
    register_loss(name, mod.make_loss)
