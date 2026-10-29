"""Student-harness options for weight averaging and the shift range (all off by default, so every existing cell behaves exactly as before).

    AL_EMA=0.9995        keep an exponential moving average of the weights and BN running statistics (decay warmed up as min(decay, (1+n)/(10+n))); validation, early stopping and the saved model
                         all use the averaged weights
    AL_BN_RECALC=1       after training, reset the BatchNorm running statistics of the chosen weights and recompute them on up to 16,384 training sequences (forward and reverse complement, no shift)
    AL_SHIFT_MAX=10      the +-shift of the training augmentation (default 10, the value that was hard-coded); 0 disables it (used when flank_context.WideCrop shifts over real bases)

Why: a collaborator's search reached a much better single model with LegNet + weight averaging + BN recalculation; this makes that an ablation we can run on our own menu, one switch at a time.

Patches albench/arch_search/arch_autoresearch.py and/or scratchpad/arch_autoresearch.py in place (backup .bak_ema next to each). Idempotent.

    python scripts/program_oct22/patch_harness_ema.py [path ...]
"""

import ast
import shutil
import sys

EMA_CLASS = '''

class _EMA:
    """Exponential moving average of a model's weights and BN statistics; ``.model`` is the shadow copy that validation and the final model are taken from."""

    def __init__(self, model, decay):
        import copy

        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        ms, es = model.state_dict(), self.model.state_dict()
        self._fl = [(ms[k], es[k]) for k in ms if ms[k].dtype.is_floating_point]
        self._int = [(ms[k], es[k]) for k in ms if not ms[k].dtype.is_floating_point]

    def update(self, n):
        import torch

        d = min(self.decay, (1.0 + n) / (10.0 + n))
        with torch.no_grad():
            torch._foreach_lerp_([e for _, e in self._fl], [m.detach() for m, _ in self._fl], 1.0 - d)
            for m, e in self._int:
                e.copy_(m)


def _bn_recalc(model, X, dev, amp_on, amp_dt, n=16384, bs=256):
    """Recompute BatchNorm running statistics (cumulative average) of ``model`` on up to n training sequences, both strands, no shift augmentation."""
    import torch

    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    idx = np.random.default_rng(0).permutation(len(X))[: min(n, len(X))]
    model.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
        for i in range(0, len(idx), bs):
            xb = torch.from_numpy(X[np.sort(idx[i : i + bs])]).to(dev)
            model(xb)
            model(rc(xb))
    for m, mo in zip(bns, moms):
        m.momentum = mo
    model.eval()
    return len(bns)
'''

EDITS = [
    # (old, new, label)
    (
        "def build_opt(model, name, lr, wd):",
        EMA_CLASS.lstrip("\n") + "\n\ndef build_opt(model, name, lr, wd):",
        "classes",
    ),
    (
        "    best, best_state, bad = np.inf, None, 0\n    t0 = time.time()\n",
        "    best, best_state, bad = np.inf, None, 0\n"
        '    ema_decay = float(os.environ.get("AL_EMA", "0") or 0)\n'
        "    ema = _EMA(model, ema_decay) if ema_decay > 0 else None\n"
        "    ema_n = 0\n"
        "    t0 = time.time()\n",
        "ema init",
    ),
    (
        "xb = shift(xb.to(dev), 10)",
        'xb = shift(xb.to(dev), int(os.environ.get("AL_SHIFT_MAX", "10")))',
        "shift max",
    ),
    (
        "            if per_step:\n                sched.step()\n        model.eval()\n",
        "            if ema is not None:\n                ema_n += 1\n                ema.update(ema_n)\n"
        "            if per_step:\n                sched.step()\n        model.eval()\n        evm = ema.model if ema is not None else model\n",
        "ema update",
    ),
    (
        "(model(Xva_t[i:i+512]).squeeze() + model(rc(Xva_t[i:i+512])).squeeze())",
        "(evm(Xva_t[i:i+512]).squeeze() + evm(rc(Xva_t[i:i+512])).squeeze())",
        "val eval",
    ),
    (
        "best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}",
        "best_state = {k: v.detach().clone() for k, v in evm.state_dict().items()}",
        "best state",
    ),
    (
        '    if best_state is not None:\n        model.load_state_dict(best_state)\n    return model, {"best_val_mse": best, "epochs_trained": ep + 1, "optimizer_used": used}, used\n',
        "    if best_state is not None:\n        model.load_state_dict(best_state)\n"
        "    extra = {}\n"
        '    if best_state is not None and os.environ.get("AL_BN_RECALC") == "1":\n'
        '        extra["val_mse_before_bn_recalc"] = best\n'
        '        extra["bn_layers_recalculated"] = _bn_recalc(model, Xtr, dev, amp_on, amp_dt)\n'
        '        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):\n'
        "            vp = torch.cat([(model(Xva_t[i : i + 512]).squeeze() + model(rc(Xva_t[i : i + 512])).squeeze()) / 2 for i in range(0, len(Xva_t), 512)]).float()\n"
        "        best = float(((vp - yva_t) ** 2).mean())\n"
        "    if ema is not None:\n"
        '        extra["ema_decay"] = ema_decay\n'
        '    return model, {"best_val_mse": best, "epochs_trained": ep + 1, "optimizer_used": used, **extra}, used\n',
        "finish",
    ),
]


def patch(p):
    s = open(p).read()
    if "_EMA" in s:
        print("already patched", p)
        return
    for old, new, label in EDITS:
        if old not in s:
            raise SystemExit(f"{p}: anchor not found: {label}")
        s = s.replace(old, new, 1)
    ast.parse(s)
    shutil.copy(p, p + ".bak_ema")
    open(p, "w").write(s)
    print("patched", p)


if __name__ == "__main__":
    paths = sys.argv[1:] or ["albench/arch_search/arch_autoresearch.py"]
    for p in paths:
        patch(p)
