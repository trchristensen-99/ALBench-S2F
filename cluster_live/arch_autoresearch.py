"""Iterative free-architecture AutoResearch, now with the SAME training levers the HP-menu path
has. Each round the model WRITES a Net(nn.Module) AND may specify its training recipe (optimizer
incl. muon, lr, weight_decay, batch_size, epochs, lr_schedule). Reverse-complement + random-shift
augmentation are ALWAYS applied (the two settled defaults); anything it omits is randomly sampled,
so defaults vary rather than being fixed. Results feed back so the next round improves.

COST CONTROL: per-model training wall-clock cap (--model_max_min) + a per-cell total training
budget (--budget_hours) that stops starting new rounds; plus the proposal param/latency caps.
Writes the same val_pred/test_pred_* npz + meta -> comparable to the HP-menu grid via enet_all_pos.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from albench.arch_search.propose import build_arch_prompt, call_model, parse_proposals
from albench.arch_search.train_generated import one_hot, predict
from albench.arch_search.validate import build_and_check, static_check

TRAIN_NOTE = (
    "SETUP: you write ONLY the architecture (the Net class, always required and complete). A fixed "
    "harness trains it: reverse-complement + random-shift augmentation are ALWAYS applied, optimizers "
    "available are muon (orthogonalized momentum, often best on conv stacks) / adamw / adam, with "
    "onecycle|plateau|cosine schedules. BUDGET: each model must train within ~{max_min:.0f} min of GPU "
    "wall-clock (slow models are stopped early and keep their best epoch) and <= {max_params:,} params; "
    "design within that. Each tried architecture below reports its validation r AND its train seconds, so "
    "prefer ideas that are both strong and efficient.\n\n"
    "You may ALSO choose the TRAINING recipe for each architecture. After its code block, add a line:\n"
    '  ### TRAIN: {{"optimizer":"muon|adamw|adam","lr":2e-3,"weight_decay":1e-5,'
    '"batch_size":128,"epochs":80,"lr_schedule":"onecycle|plateau|cosine"}}\n'
    "muon (orthogonalized momentum) often beats AdamW on conv stacks. Reverse-complement and "
    "random-shift augmentation are ALWAYS applied for you, so do not implement them. If you omit "
    "### TRAIN, reasonable settings are sampled for you — so feel free to specify only what you care about."
)
_TRAIN = re.compile(r"###\s*TRAIN:\s*(\{.*?\})", re.S)
REFERENCE_SRC = """\
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        h = max(c // r, 4)
        self.fc1 = nn.Linear(c, h)
        self.fc2 = nn.Linear(h, c)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(x.mean(-1)))))
        return x * s.unsqueeze(-1)


class EffBlock(nn.Module):
    # inverted residual + squeeze-excitation (LegNet-style)
    def __init__(self, cin, cout, ks=7, expand=2):
        super().__init__()
        mid = cin * expand
        self.body = nn.Sequential(
            nn.Conv1d(cin, mid, 1, bias=False), nn.BatchNorm1d(mid), nn.SiLU(),
            nn.Conv1d(mid, mid, ks, padding=ks // 2, groups=mid, bias=False), nn.BatchNorm1d(mid), nn.SiLU(),
        )
        self.se = SE(mid)
        self.proj = nn.Sequential(nn.Conv1d(mid, cout, 1, bias=False), nn.BatchNorm1d(cout))
        self.skip = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.proj(self.se(self.body(x)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [64, 96, 128, 128, 128, 96, 64]
        self.stem = nn.Sequential(nn.Conv1d(4, widths[0], 7, padding=3, bias=False), nn.BatchNorm1d(widths[0]), nn.SiLU())
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(EffBlock(c, w, ks=7 if i % 2 == 0 else 9))
            c = w
            if i in (1, 3, 5):
                blocks.append(nn.MaxPool1d(2))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c * 2, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h).squeeze(-1)
"""

KNOWN_GOOD = (
    "WHAT WE ALREADY KNOW WORKS ON THIS TASK (from thousands of earlier training runs on this and related datasets; "
    "a starting point, not a constraint):\n"
    "- Reverse-complement and shift augmentation are applied for you; do not re-implement them.\n"
    "- Optimizer: muon (orthogonalized momentum) had the highest mean validation Pearson of the three optimizers we tried "
    "(muon 0.76 vs adamw and adam lower). Typical muon settings: lr 1e-3 to 5e-3, weight decay 1e-5 to 3e-4, "
    "onecycle or cosine schedule, batch size 256 to 512.\n"
    "- Backbone: LegNet-style inverted-residual blocks with squeeze-excitation had the highest mean validation Pearson of "
    "the conv block types we tried (0.77). Good settings: 8 to 12 blocks, base width 64 to 128 with a wider middle, "
    "kernel size 7 to 9, low conv dropout (about 0.05 to 0.1) and higher dense dropout (about 0.3 to 0.4).\n"
    "- Free-form architectures reached validation Pearson 0.80 to 0.82 when they paired a convolutional motif stem with "
    "residual/squeeze-excite towers, a U-Net style encoder-decoder, or one small attention layer; purely hand-built "
    "motif-grammar, frequency-domain or wavelet models scored clearly lower (0.55 to 0.74).\n"
    "- A good single model reaches validation Pearson of about 0.78 to 0.82 with 30,000 training sequences."
)
PHASE_EXPLOIT = (
    "PHASE 1 - BUILD ON WHAT WORKS: for these rounds, propose variations on the reference network below and on the best "
    "architectures already tried (change depth, width, kernel size or block design; add squeeze-excitation, gating or one "
    "small attention layer; change pooling or residual wiring), keeping what makes them strong. Do not propose unrelated "
    "designs yet."
)
PHASE_EXPLORE = (
    "PHASE 2 - EXPLORE WIDER: the well-known designs are now covered. Propose structurally different architectures that "
    "could complement the best ones (different inductive biases; different ways of modelling motif spacing, arrangement "
    "and composition), while still using ingredients known to work (muon, residual structure, normalisation). Avoid "
    "near-duplicates of anything in the tried list."
)
ENSEMBLE_NOTE = (
    "Your models are combined by an ElasticNet ensemble fit on validation predictions, so ensemble value matters: a "
    "decorrelated architecture that is slightly weaker alone can be worth more than another near-duplicate of the best. "
    "Propose architectures that differ from each other as well as from those already tried."
)
FULL_NOTE = (
    "FULL CONTROL: besides the Net class, you may ALSO write a function "
    "`def train(model, Xtr, ytr, Xva, yva, device, time_budget_s):` that performs the ENTIRE "
    "training yourself (optimizer, schedule, augmentation — anything in torch/numpy) and returns the "
    "trained model. Xtr/Xva are float32 arrays (N,4,200) one-hot ACGT; ytr/yva are float32 (N,). You "
    "MUST return within time_budget_s (~{max_min:.0f} min of GPU wall-clock here) — checkpoint your "
    "best and stop in time. Reverse-complement + random-shift augmentation are good defaults but now "
    "YOURS to include or not. If you omit train(), a default harness (muon/adamw, RC+shift) is used. "
    "Model params <= {max_params:,}. Each tried architecture below reports validation r AND train seconds; "
    "prefer ideas that are strong and efficient.\n\n"
    "You may add ### TRAIN: {{...}} for the default harness instead, if you did not write train()."
)


def pear(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    d = (a * a).sum() ** 0.5 * (b * b).sum() ** 0.5
    return float((a * b).sum() / d) if d > 0 else 0.0


def rc(x):  # (B,4,L) ACGT one-hot -> reverse complement
    return x.flip(-1)[:, [3, 2, 1, 0], :]


def shift(x, m):  # padded random shift along length
    if m <= 0:
        return x
    import torch
    s = random.randint(-m, m)
    if s == 0:
        return x
    y = torch.zeros_like(x)
    flank = os.environ.get("AL_FLANK_SHIFT") == "1"  # real plasmid bases instead of zeros where the window moves past the insert
    if s > 0:
        y[:, :, s:] = x[:, :, :-s]
        if flank:
            y[:, :, :s] = _flank(x.device, x.dtype)[0][:, -s:]
    else:
        y[:, :, :s] = x[:, :, -s:]
        if flank:
            y[:, :, s:] = _flank(x.device, x.dtype)[1][:, : -s]
    return y


_FLANK_CACHE: dict = {}


def _flank(device, dtype):
    """(upstream tail, downstream head) one-hot (4, N) tensors in ACGT order."""
    key = (str(device), dtype)
    if key not in _FLANK_CACHE:
        import torch

        from albench.arch_search.plasmid_flank import DOWNSTREAM_HEAD, UPSTREAM_TAIL

        def oh(seq):
            t = torch.zeros(4, len(seq), device=device, dtype=dtype)
            for i, c in enumerate(seq):
                t["ACGT".index(c), i] = 1.0
            return t

        _FLANK_CACHE[key] = (oh(UPSTREAM_TAIL), oh(DOWNSTREAM_HEAD))
    return _FLANK_CACHE[key]


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


def build_opt(model, name, lr, wd):
    import torch
    name = (name or "adamw").lower()
    if name == "muon":
        try:
            from muon import SingleDeviceMuonWithAuxAdam
            p2 = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
            p1 = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
            groups = [dict(params=p2, use_muon=True, lr=lr, weight_decay=wd, momentum=0.95)]
            if p1:
                groups.append(dict(params=p1, use_muon=False, lr=lr, weight_decay=wd, betas=(0.9, 0.95)))
            return SingleDeviceMuonWithAuxAdam(groups), "muon"
        except Exception as e:
            print(f"    [muon unavailable: {e}; falling back to adamw]", flush=True)
            name = "adamw"
    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
    return cls(model.parameters(), lr=lr, weight_decay=wd), name


def sample_train(spec):
    """Fill missing training HPs with randomly-varied defaults."""
    if os.environ.get("AL_RECIPE_MODE") == "harness":  # the harness assigns the recipe: Muon with a small spread around the settings that every recipe study found equivalent
        r = random.Random()
        return {
            "optimizer": "muon",
            "lr": r.choice([1e-3, 2e-3, 2e-3, 4e-3]),
            "weight_decay": r.choice([1e-4, 1e-4, 3e-4, 1e-3]),
            "batch_size": r.choice([128, 256, 256, 512]),
            "epochs": 80,
            "lr_schedule": r.choice(["onecycle", "onecycle", "cosine"]),
        }
    r = random.Random()
    out = {
        "optimizer": spec.get("optimizer", r.choice(["muon", "muon", "adamw"])),
        "lr": float(spec.get("lr", 10 ** r.uniform(-3.3, -2.3))),
        "weight_decay": float(spec.get("weight_decay", 10 ** r.uniform(-6, -3))),
        "batch_size": int(spec.get("batch_size", r.choice([64, 128, 256]))),
        "epochs": int(spec.get("epochs", 80)),
        "lr_schedule": spec.get("lr_schedule", r.choice(["onecycle", "plateau", "cosine"])),
    }
    return out


def train_one(model, Xtr, ytr, Xva, yva, tc, device="cuda", max_sec=900, log=print):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt, used = build_opt(model, tc["optimizer"], tc["lr"], tc["weight_decay"])
    bs, epochs = tc["batch_size"], tc["epochs"]
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=bs, shuffle=True, drop_last=True)
    if tc["lr_schedule"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], epochs=epochs,
                                                    steps_per_epoch=max(1, len(dl)), cycle_momentum=False)
        per_step = True
    elif tc["lr_schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs); per_step = False
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3); per_step = False
    lossf = torch.nn.MSELoss()
    # Precision is set by the harness for every model alike (env AL_PRECISION); a proposal cannot opt in or out, so no model gets a private speed-up.
    prec = os.environ.get("AL_PRECISION", "fp32").lower()
    amp_on = prec in ("fp16", "bf16") and dev.type == "cuda"
    amp_dt = torch.float16 if prec == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (prec == "fp16" and dev.type == "cuda") else None
    if prec == "tf32" and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    Xva_t = torch.from_numpy(Xva).to(dev); yva_t = torch.from_numpy(yva).to(dev)
    best, best_state, bad = np.inf, None, 0
    _nf_run = _nf_total = 0
    ema_decay = float(os.environ.get("AL_EMA", "0") or 0)
    ema = _EMA(model, ema_decay) if ema_decay > 0 else None
    ema_n = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = shift(xb.to(dev), int(os.environ.get("AL_SHIFT_MAX", "10"))); yb = yb.to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
                if os.environ.get("AL_RC_MODE", "consistency") == "aug":
                    # oracle-style augmentation: flip half of the samples, one forward pass
                    _flip = torch.rand(xb.shape[0], device=xb.device) < 0.5
                    out = model(torch.where(_flip[:, None, None], rc(xb), xb)).squeeze()
                    out_rc = out
                else:
                    out = model(xb).squeeze()
                    # RC consistency: average fwd + reverse-complement prediction in the loss
                    out_rc = model(rc(xb)).squeeze()
            loss = 0.5 * (lossf(out.float(), yb) + lossf(out_rc.float(), yb))
            if not torch.isfinite(loss):
                if scaler is not None and os.environ.get("AL_NONFINITE_SKIP") == "1":  # fp16: skip the batch, give up after 20 in a row
                    _nf_run += 1
                    _nf_total += 1
                    if _nf_run > 20:
                        return None, {"error": f"non-finite loss ep{ep} (20 batches in a row)"}, used
                    continue
                return None, {"error": f"non-finite loss ep{ep}"}, used
            _nf_run = 0
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if ema is not None:
                ema_n += 1
                ema.update(ema_n)
            if per_step:
                sched.step()
        model.eval()
        evm = ema.model if ema is not None else model
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
            vp = torch.cat([(evm(Xva_t[i:i+512]).squeeze() + evm(rc(Xva_t[i:i+512])).squeeze()) / 2
                            for i in range(0, len(Xva_t), 512)]).float()
            vmse = float(((vp - yva_t) ** 2).mean())
        if not per_step:
            sched.step(vmse) if tc["lr_schedule"] == "plateau" else sched.step()
        if vmse < best - 1e-4:
            best, bad = vmse, 0
            best_state = {k: v.detach().clone() for k, v in evm.state_dict().items()}
        else:
            bad += 1
            # AL_PATIENCE (default 8 as before); AL_ONECYCLE_FULL=1: a one-cycle schedule only anneals at its end, so patience never stops it (the time cap below still does)
            if bad >= int(os.environ.get("AL_PATIENCE", "8")) and not (
                os.environ.get("AL_ONECYCLE_FULL") == "1" and tc["lr_schedule"] == "onecycle"
            ):
                break
        if time.time() - t0 > max_sec:
            log(f"    [time cap {max_sec}s hit at ep{ep}]"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    extra = {}
    if best_state is not None and os.environ.get("AL_BN_RECALC") == "1":
        extra["val_mse_before_bn_recalc"] = best
        extra["bn_layers_recalculated"] = _bn_recalc(model, Xtr, dev, amp_on, amp_dt)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
            vp = torch.cat([(model(Xva_t[i : i + 512]).squeeze() + model(rc(Xva_t[i : i + 512])).squeeze()) / 2 for i in range(0, len(Xva_t), 512)]).float()
        best = float(((vp - yva_t) ** 2).mean())
    if ema is not None:
        extra["ema_decay"] = ema_decay
    if _nf_total:
        extra["nonfinite_skipped"] = _nf_total
    return model, {"best_val_mse": best, "epochs_trained": ep + 1, "optimizer_used": used, **extra}, used


def predict_rc(model, X):  # RC-averaged prediction
    import torch
    dev = next(model.parameters()).device
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i+512]).to(dev)
            p = (model(xb).squeeze() + model(rc(xb)).squeeze()) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out)


import builtins as _bi
_SAFE_BUILTINS = {k: getattr(_bi, k) for k in (
    "range len enumerate zip min max sum abs int float bool str bytes list dict tuple set "
    "frozenset sorted reversed map filter any all round pow print isinstance issubclass super "
    "getattr setattr hasattr delattr type object Exception ValueError RuntimeError TypeError "
    "KeyError IndexError StopIteration slice iter next repr format divmod complex callable id "
    "True False None NotImplemented"
).split() if hasattr(_bi, k)}
from albench.arch_search.validate import ALLOWED_IMPORTS as _ALLOWED


def _gated_import(name, *a, **k):
    import importlib
    if name.split(".")[0] not in {i.split(".")[0] for i in _ALLOWED}:
        raise ImportError(f"import of {name!r} not permitted")
    return importlib.__import__(name, *a, **k)


def extract_train(src):
    """Return the model-written train() if the source defines one and is AST-safe, else None."""
    if static_check(src):   # non-None = unsafe
        return None
    import torch, numpy, math, random
    import torch.nn as nn, torch.nn.functional as F
    sb = dict(_SAFE_BUILTINS); sb["__import__"] = _gated_import
    ns = {"__builtins__": sb, "torch": torch, "nn": nn, "F": F,
          "np": numpy, "numpy": numpy, "math": math, "random": random}
    try:
        exec(compile(src, "<gen-train>", "exec"), ns)   # noqa: S102 (AST-checked + sandboxed builtins)
    except Exception:
        return None
    t = ns.get("train")
    return t if callable(t) else None


def run_full_control(tfn, model, Xtr, ytr, Xva, yva, cap_s):
    """Run a model-written train() under a hard wall-clock cap (SIGALRM)."""
    import signal

    class _TO(Exception):
        pass

    def _h(sig, frm):
        raise _TO()
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(cap_s * 1.3))
    try:
        m = tfn(model, Xtr, ytr, Xva, yva, "cuda", cap_s)
        return (m if m is not None else model), {"best_val_mse": None, "epochs_trained": None}, "model_train"
    except _TO:
        return None, {"error": "full-control train() exceeded hard cap"}, "model_train"
    except Exception as e:
        return None, {"error": f"model train() failed: {type(e).__name__}: {str(e)[:80]}"}, "model_train"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


STYLE_EXPLORE = ("EXPLORATION STYLE: be bold. Spend most of your proposals on structurally different ideas that the history does not yet contain "
                 "(different block types, receptive-field layouts, pooling, attention placement, heads, widths and depths), not small variations of the current best. "
                 "Each proposal should differ from the others and from earlier ones in more than one design choice, while still being a model you expect to train well.")
STYLE_REFLECT = ("REFLECTION STYLE: before the proposals, write one short paragraph that starts with the word REFLECTION: and states what the results so far show about which design "
                 "choices raise or lower validation Pearson r (cite specific earlier ideas and their scores) and which hypotheses are still untested. Then give the proposals, "
                 "chosen to test those hypotheses.")


def _style_parts(args, lessons):
    st = getattr(args, "style", "none")
    parts = []
    if "explore" in st:
        parts.append(STYLE_EXPLORE)
    if "reflect" in st:
        parts.append(STYLE_REFLECT)
        if lessons:
            parts.append("YOUR EARLIER REFLECTIONS (oldest first):\n" + "\n".join(f"- {x}" for x in lessons[-5:]))
    return parts


FLANK_NOTE = (
    "INPUT CONTEXT: x has length L = {L}: the 200-bp variable insert in the centre plus {F} real bases of the constant plasmid sequence on EACH side. "
    "The assay and the oracle that labels the sequences see the insert inside this plasmid, so a motif that spans the insert boundary is part of the label; the flank bases are identical for every "
    "sequence. During training the window is shifted by up to +-{S} bases over real sequence; evaluation uses the centred window. Write the architecture for any L divisible by 8: do not hard-code 200 "
    "or {L} (use global or adaptive pooling and operations that do not depend on the length). A model that cannot run at L = {L} is discarded."
)


FLANK_NOTE_FIXED = (
    "INPUT CONTEXT: x has length L = {L} for every model in this study, fixed by the harness and never changed: the 200-bp variable insert in the centre plus {F} real bases of the constant plasmid sequence on EACH side. "
    "The assay and the oracle that labels the sequences see the insert inside this plasmid, so a motif that spans the insert boundary is part of the label; the flank bases are identical for every sequence. "
    "During training the window is shifted by up to +-{S} bases over real sequence; evaluation uses the centred window. Because L is fixed you may use position-specific designs sized for L "
    "(position-resolved readouts, fixed windows, learned positional parameters); you are free to explore any architecture."
)

REFERENCE_LEGNET_NOTE = (
    "REFERENCE: THE PUBLISHED MPRA-LegNet (the standard baseline for this task), written to this contract. A model of this kind trained by this harness on this training set reaches the validation r given with it. "
    "It has about 1.3 million parameters; whether a larger or a smaller model is better can depend on the training-set size (in a first test, doubling the channel widths of our earlier configurations at 100k sequences changed the genomic-test r by -0.002 to +0.004, within noise), so capacity is worth testing, not assuming. It is a starting point and a bar, not a template: the goal is a set of strong architectures that differ from each other and from it, so propose different designs as well as improvements of it.\n```python\n"
)



MINIMAL_GUIDANCE = (
    "WHAT EARLIER SEARCHES ON THIS TASK SHOWED (facts, not instructions):\n"
    "- The ensemble of your models is what is scored, so a model whose predictions are decorrelated from the best ones is worth more than a slightly better copy of a design that is already trained "
    "(a 0.002 gain on a near-duplicate adds nothing to the ensemble).\n"
    "- Convolutional models, most often with squeeze-excitation blocks, have been the strongest family: validation Pearson 0.77 to 0.82 at this size. Pure transformer, MLP-mixer, spectral or wavelet "
    "and hand-built motif-grammar models have scored clearly lower (0.43 to 0.74), as have models that discard position entirely.\n"
    "- Reverse-complement and shift augmentation are applied for you and the input includes real plasmid context (see below); do not re-implement them.\n"
    "- Retraining the same design with another seed moves its validation Pearson by about 0.005 to 0.01: do not read a smaller gap between two designs as real."
)


OPEN_GUIDANCE = (
    "WHAT EARLIER SEARCHES ON THIS TASK SHOWED (facts, not instructions):\n"
    "- The ensemble of your models is what is scored, so a model whose predictions are decorrelated from the best ones is worth more than a slightly better copy of a design that is already trained "
    "(a 0.002 gain on a near-duplicate adds nothing to the ensemble).\n"
    "- No architecture family is known to be best on this task; unusual mechanisms have not been ruled out, so look widely before refining.\n"
    "- Reverse-complement and shift augmentation are applied for you and the input includes real plasmid context (see below); do not re-implement them.\n"
    "- Retraining the same design with another seed moves its validation Pearson by about 0.005 to 0.01: do not read a smaller gap between two designs as real."
)


def _round_note(args, rd, cap_min, lessons=()):
    """Per-round prompt note. Default (no --guidance) is byte-identical to the old behaviour."""
    base = (FULL_NOTE if args.full_control else TRAIN_NOTE).format(max_min=cap_min, max_params=args.max_params)
    if getattr(args, "wide", False):
        base += "\n\n" + (FLANK_NOTE_FIXED if getattr(args, "fixed_length", False) else FLANK_NOTE).format(L=200 + 2 * args.ctx_F, F=args.ctx_F, S=args.ctx_S)
    ctx = ""
    if getattr(args, "context_file", ""):
        ctx = open(args.context_file).read().strip()  # pre-built cross-D hand-off (see scripts/analysis/make_ladder_context.py)
    sp = _style_parts(args, lessons)
    if args.guidance not in ("known_good", "minimal", "open"):
        return "\n\n".join(([ctx] if ctx else []) + sp + [base])
    parts = [{"minimal": MINIMAL_GUIDANCE, "open": OPEN_GUIDANCE}.get(args.guidance, KNOWN_GOOD)] + ([ctx] if ctx else [])
    if getattr(args, "target_val", 0):
        parts.append(f"TARGET: models with validation Pearson r of about {args.target_val:.3f} have been trained on this task at this training-set size.")
    elif getattr(args, "target_delta", 0) and getattr(args, "baseline_file", ""):
        import re as _re
        try:
            _b = float(_re.search(r"validation Pearson r = ([0-9]+[.][0-9]+)", open(args.baseline_file).read()).group(1))
            parts.append(f"TARGET: models with validation Pearson r of about {_b + args.target_delta:.3f} (a little above a well-tuned convolutional baseline) have been trained on this task at this training-set size.")
        except Exception:
            pass
    if getattr(args, "reference_legnet", False) and rd >= getattr(args, "reference_from_round", 0):
        from albench.arch_search.mpra_legnet_ref import SOURCE as LEGNET_SRC

        bl = open(args.baseline_file).read().strip() + "\n\n" if getattr(args, "baseline_file", "") else ""
        parts.append(bl + REFERENCE_LEGNET_NOTE + LEGNET_SRC + "```")
    elif getattr(args, "baseline_file", "") and not getattr(args, "reference_legnet", False):
        parts.append(open(args.baseline_file).read().strip())
    if args.phase_split and rd < args.phase_split:
        parts += [PHASE_EXPLOIT, "REFERENCE IMPLEMENTATION (a LegNet-style network that trains in this harness; copy or "
                  "modify it):\n```python\n" + REFERENCE_SRC + "```"]
    elif args.phase_split:
        parts.append(PHASE_EXPLORE)
    if args.ensemble_aware:
        parts.append(ENSEMBLE_NOTE)
    parts += sp
    parts.append(base)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoir_cache", required=True)
    ap.add_argument("--D", type=int, default=30000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--per_round", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--max_params", type=int, default=40_000_000)
    ap.add_argument("--base_cap_min", type=float, default=None)   # per-model cap, H100-equiv min (x2.5 on V100)
    ap.add_argument("--model_max_min", type=float, default=None)  # back-compat: flat per-model cap (no GPU scale)
    ap.add_argument("--full_control", action="store_true")        # model writes the whole training loop
    ap.add_argument("--budget_hours", type=float, default=40.0)    # per-cell total train budget
    ap.add_argument("--guidance", default="none", choices=["none", "known_good", "minimal", "open"])  # prior knowledge block in the prompt
    ap.add_argument("--phase_split", type=int, default=0)  # rounds < split: build on known-good; after: explore wider (0 = off)
    ap.add_argument("--ensemble_aware", action="store_true")  # tell the proposer the deploy metric is an ensemble
    ap.add_argument("--context_file", default="")  # text block with results from a smaller dataset size (cross-D context)
    ap.add_argument("--style", default="none", choices=["none", "explore", "reflect", "explore_reflect"])  # extra prompt persona; none = unchanged prompt
    ap.add_argument("--resume", action="store_true")  # rebuild history from out_dir and continue (safe under requeue)
    ap.add_argument("--loop", default="single", choices=["single", "agent"])  # agent = multi-agent proposer/coder/reflector loop (albench/arch_search/agent_loop.py)
    ap.add_argument("--agent_cfg", default="")  # agent loop only: JSON object or path (stages, repairs, ...)
    ap.add_argument("--val_cache", default="", help="npz (sequences, oracle_labels, oracle_id) replacing the chr19/21/X validation set for early stopping, val_pred, val_labels and proposer feedback; default empty = unchanged behaviour")
    ap.add_argument("--flank_ctx", type=int, default=-1, help="F real plasmid bases of input context on each side of the 200-bp insert (shift over real bases); -1 = off")
    ap.add_argument("--reference_from_round", type=int, default=0, help="show the reference LegNet code and baseline only from this round on (0 = every round)")
    ap.add_argument("--target_val", type=float, default=0.0, help="optional known-achievable validation r stated in the prompt (0 = off)")
    ap.add_argument("--target_delta", type=float, default=0.0, help="state a target = the in-run baseline validation r + this delta (needs the baseline, i.e. --reference_legnet; 0 = off)")
    ap.add_argument("--reference_legnet", action="store_true", help="show the proposer the published MPRA-LegNet (as a bar and a starting point)")
    ap.add_argument("--baseline_file", default="", help="text file appended to the guidance block (e.g. the baseline's validation r on this training set)")
    ap.add_argument("--fixed_length", action="store_true", help="with --flank_ctx: the input length is fixed by the harness, position-specific designs are allowed")
    ap.add_argument("--no_context", action="store_true", help="train on the bare 200-bp insert (an experiment that deliberately differs from the standard: 20 real flank bases each side, shift +-10)")
    ap.add_argument("--length_agnostic", action="store_true", help="with context: ask for length-agnostic architectures instead of telling the proposer the length is fixed")
    ap.add_argument("--shift_max", type=int, default=-1, help="+-shift of the training window with --flank_ctx (default 10); -1 = off")
    args = ap.parse_args()

    from scaling_hp_search import (
        _atomic_savez, _atomic_write_text,
        load_all_test_sets, load_chr_test_genomic, load_chr_train_pool,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    import torch as _t
    gpu = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu"
    gpu_factor = 5.0 if "V100" in gpu else 1.0  # V100 ~8x slower on compute (measured); 5x end-to-end
    if args.base_cap_min is not None:
        cap_min = args.base_cap_min * gpu_factor
    elif args.model_max_min is not None:
        cap_min = args.model_max_min
    else:
        cap_min = 20.0 * gpu_factor
    print(f"  GPU={gpu} factor={gpu_factor} -> per-model cap={cap_min:.0f}min; full_control={args.full_control}", flush=True)
    tr_s, tr_y, va_s, va_y = load_chr_train_pool(
        args.D, ref_only=True, val_frac=0.1, seed=0, reservoir_cache=args.reservoir_cache, chr_val=True)
    if args.val_cache:  # alternative validation set (albench/arch_search/val_cache.py); the default chr19/21/X path above is untouched
        from albench.arch_search.val_cache import guard_out_dir, load_val_cache
        va_s, va_y, _vc = load_val_cache(args.val_cache, train_seqs=tr_s)
        guard_out_dir(args.out_dir, _vc)
        print(f"  val_cache={_vc['path']} variant={_vc['variant']} n={_vc['n']:,} sha={_vc['sha1_16']} oracle={_vc['oracle_id']} "
              f"test_overlap={_vc['n_test_overlap']} train_overlap={_vc['n_train_overlap']}", flush=True)
    elif os.path.exists(os.path.join(args.out_dir, "val_cache.json")):
        raise SystemExit(f"{args.out_dir} was started with --val_cache; pass the same --val_cache or use a fresh --out_dir")
    te_s, te_oracle, te_true = load_chr_test_genomic()
    tests = load_all_test_sets()
    from albench.arch_search.harness_standard import FLANK_CTX as STD_F, SHIFT_MAX as STD_S

    if args.no_context:
        args.flank_ctx = args.shift_max = -1
    elif args.flank_ctx < 0 and args.shift_max < 0:  # the standard, unless this directory already holds models trained without context
        _legacy = bool(list(Path(args.out_dir).glob("r*_arch_*_meta.json"))) and not (Path(args.out_dir) / "flank_ctx.json").exists()
        if not _legacy:
            args.flank_ctx, args.shift_max = STD_F, STD_S
    args.wide = args.flank_ctx >= 0 or args.shift_max >= 0
    if args.wide and not args.length_agnostic:
        args.fixed_length = True
    args.ctx_F, args.ctx_S = max(args.flank_ctx, 0), (args.shift_max if args.shift_max >= 0 else 10)
    _cp = Path(args.out_dir) / "flank_ctx.json"
    if args.wide:
        from albench.arch_search.flank_context import wide_onehot, wrap as wrap_ctx

        if os.environ.get("AL_FLANK_SHIFT") == "1":
            raise SystemExit("--flank_ctx/--shift_max replace AL_FLANK_SHIFT; unset it")
        if not _cp.exists() and list(Path(args.out_dir).glob("r*_arch_*_meta.json")):
            raise SystemExit(f"{args.out_dir} already holds models trained without flank context; use a fresh --out_dir")
        if _cp.exists() and json.load(open(_cp)) != {"F": args.ctx_F, "S": args.ctx_S}:
            raise SystemExit(f"{args.out_dir} was started with {open(_cp).read()}; pass the same flags")
        _cp.write_text(json.dumps({"F": args.ctx_F, "S": args.ctx_S}))
        os.environ["AL_SHIFT_MAX"] = "0"  # WideCrop shifts over real bases
        args.seq_len = 200 + 2 * args.ctx_F  # the contract the proposer and the coder see
    elif _cp.exists():
        raise SystemExit(f"{args.out_dir} was started with flank context; pass the same flags or use a fresh --out_dir")

    def _enc(seqs):
        return wide_onehot(seqs, args.ctx_F + args.ctx_S) if args.wide else one_hot(seqs, args.seq_len)

    def _wrap(m):
        return wrap_ctx(m, args.ctx_F, args.ctx_S) if args.wide else m

    args.harness = {k: os.environ[k] for k in ("AL_EMA", "AL_BN_RECALC", "AL_FLANK_SHIFT", "AL_SHIFT_MAX", "AL_PATIENCE", "AL_PRECISION", "AL_ONECYCLE_FULL", "AL_NONFINITE_SKIP") if os.environ.get(k)}
    if args.wide:
        args.harness.update({"flank_ctx": args.ctx_F, "shift_max": args.ctx_S})

    Xtr, ytr = _enc(tr_s), np.asarray(tr_y, np.float32)
    Xva, yva = _enc(va_s), np.asarray(va_y, np.float32)
    Xte = _enc(te_s)
    Xsets = {k: _enc(v[0]) for k, v in tests.items()}
    labels = {"val_labels": yva, "test_oracle": te_oracle, "test_true": te_true}
    for k, v in tests.items():
        labels[f"oracle_{k}"] = v[1]
    _atomic_savez(Path(args.out_dir) / "labels.npz", **labels)
    print(f"  train={len(tr_s):,} val={len(va_s):,} test={len(te_s):,} evals={len(tests)}", flush=True)

    if args.reference_legnet and not args.baseline_file:  # bank-derived reference bar (docs/TUNED_LEGNET.md)
        _bl = Path(args.out_dir) / "baseline_legnet.txt"
        _bank = {  # D: (tuned recipe description, tuned bank mean val r, out-of-the-box bank mean val r); 20 seeds each, V0c
            10000: ("weight decay 0.4, 800 epochs", 0.809, 0.756),
            30000: ("weight decay 0.2, 500 epochs", 0.905, 0.882),
            100000: ("weight decay 0.1, 500 epochs", 0.945, 0.940),
        }.get(int(args.D))
        if not _bl.exists() and _bank is not None:
            _bl.write_text(
                f"BASELINE: the MPRA-LegNet (shown below), trained 20 times with its reference script on {len(tr_s):,} sequences of this kind, reaches validation Pearson r = {_bank[1]:.3f} (mean over seeds; tuned recipe: "
                f"AdamW, learning rate 0.01, {_bank[0]}, batch 1024, one-cycle). Its out-of-the-box recipe (weight decay 0.1, 100 epochs) reaches {_bank[2]:.3f}. Networks trained by this harness can score lower than the reference "
                f"script does for the same architecture, so treat the first value as the level a single model should approach; what is scored is the ensemble of everything you train."
            )
        if _bl.exists():
            args.baseline_file = str(_bl)

    if args.loop == "agent":  # opt-in; everything below this branch (the default loop) is untouched
        from types import SimpleNamespace

        from albench.arch_search.agent_loop import run_agent_loop

        return run_agent_loop(args, SimpleNamespace(
            Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, Xsets=Xsets, te_oracle=te_oracle, gpu=gpu,
            cap_min=cap_min, train_one=train_one, sample_train=sample_train, predict_rc=predict_rc,
            wrap_model=_wrap, harness=args.harness,
            pear=pear, atomic_savez=_atomic_savez, atomic_write_text=_atomic_write_text,
            guidance_note=lambda rd: _round_note(args, rd, cap_min),
            train_note=(FULL_NOTE if args.full_control else TRAIN_NOTE).format(
                max_min=cap_min, max_params=args.max_params)))
    history, midx, t_start = [], 0, time.time()
    lessons = []
    if args.resume and (Path(args.out_dir) / "reflections.jsonl").exists():
        lessons = [json.loads(l)["text"] for l in open(Path(args.out_dir) / "reflections.jsonl") if l.strip()]
    last_err, streak, start_rd, budget_hit = None, 0, 0, False
    out_p = Path(args.out_dir)
    (out_p / "prompts").mkdir(exist_ok=True)
    if args.resume:
        rows = []
        for f in sorted(out_p.glob("r*_arch_*_meta.json")):
            try:
                m = json.loads(f.read_text())
            except Exception:
                continue
            if m.get("strategy") == "arch_search" and (out_p / (m["model_id"] + ".npz")).exists():
                rows.append(m)
        rows.sort(key=lambda m: (int(m["round"]), m["model_id"]))
        for m in rows:
            history.append({"idea": m["idea"], "ok": True, "score": m.get("val_pearson"), "n_params": m.get("n_params"),
                            "val": m.get("val_pearson"), "train_s": m.get("train_time_sec")})
        if rows:
            midx = 1 + max(int(m["model_id"].split("_")[-1]) for m in rows)
            start_rd = len({int(m["round"]) for m in rows})  # productive rounds; failed/empty rounds are redone
            print(f"  [resume] {len(rows)} models on disk; continuing at round {start_rd}, model index {midx}", flush=True)
    _atomic_write_text(out_p / "config.json", json.dumps(
        {"args": vars(args), "LLM_MODEL": os.environ.get("LLM_MODEL"), "argv": sys.argv, "resumed_from_round": start_rd},
        indent=2, default=str))
    target = args.rounds * args.per_round  # every cell ends with the same number of TRAINED models, not rounds
    max_rounds = args.rounds + max(6, args.rounds // 4)
    rd = start_rd - 1
    while midx < target and rd + 1 < max_rounds:
        rd += 1
        if time.time() - t_start > args.budget_hours * 3600:
            print(f"  [budget {args.budget_hours}h reached at round {rd}] stopping", flush=True); budget_hit = True; break
        try:
            sys_p, user = build_arch_prompt(args.per_round, history, seq_len=args.seq_len,
                                            D=args.D, max_params=args.max_params, note=_round_note(args, rd, cap_min, lessons))
            if rd in (start_rd, args.phase_split):
                _atomic_write_text(out_p / "prompts" / f"prompt_r{rd:02d}.txt", sys_p + "\n\n=====\n\n" + user)
            text = call_model(sys_p, user)
            if "reflect" in args.style:
                import re as _re
                _m = _re.search(r"REFLECTION:\s*(.+?)(?:\n\s*\n|\Z)", text, _re.S)
                if _m:
                    lessons.append(_m.group(1).strip()[:900])
                    with open(Path(args.out_dir) / "reflections.jsonl", "a") as _fh:
                        _fh.write(json.dumps({"round": rd, "text": lessons[-1]}) + "\n")
            props = parse_proposals(text)
            trains = _TRAIN.findall(text)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  [round {rd}] propose failed: {msg}", flush=True)
            if type(e).__name__ == "RateLimitExceeded":
                print(f"RATE_LIMITED at round {rd} ({midx} models so far): exiting 75 so the dispatcher backs off; no rounds burned", flush=True)
                raise SystemExit(75)
            streak = streak + 1 if msg == last_err else 1
            last_err = msg
            if midx == 0 and streak >= 4:
                print(f"FAILED_ARCH: identical propose failure x{streak} with nothing trained; aborting", flush=True)
                raise SystemExit(3)
            continue
        print(f"  [round {rd}] {len(props)} proposals", flush=True)
        for pi, p in enumerate(props):
            model, verdict = build_and_check(p["source"], seq_len=args.seq_len, max_params=args.max_params)
            if model is None:
                history.append({"idea": p["idea"], "ok": False, "reason": verdict.reason})
                print(f"    discard: {verdict.reason[:75]}", flush=True); continue
            spec = {}
            try:
                if pi < len(trains):
                    spec = json.loads(trains[pi])
            except Exception:
                spec = {}
            tc = sample_train(spec)
            import torch
            torch.manual_seed(1000 + midx)
            t0 = time.time()
            try:
                tfn = extract_train(p["source"]) if args.full_control else None
                if tfn is not None:
                    model, info, used = run_full_control(tfn, model, Xtr, ytr, Xva, yva, cap_min * 60)
                else:
                    model, info, used = train_one(_wrap(model), Xtr, ytr, Xva, yva, tc, max_sec=cap_min * 60)
            except Exception as e:
                history.append({"idea": p["idea"], "ok": False, "reason": f"train err {type(e).__name__}"})
                print(f"    train-err: {type(e).__name__}: {str(e)[:60]}", flush=True); continue
            if model is None:
                history.append({"idea": p["idea"], "ok": False, "reason": info.get("error", "fail")})
                print(f"    train-fail: {info.get('error')}", flush=True); continue
            res = {"val_pred": predict_rc(model, Xva), "test_pred": predict_rc(model, Xte)}
            _vp = res["val_pred"]
            if (not np.isfinite(_vp).all()) or float(np.abs(_vp).max()) > 1e3:  # finite-but-absurd outputs (seen: ~1e20) wreck any ensemble
                history.append({"idea": p["idea"], "ok": False, "reason": "exploded predictions (|pred|>1e3)"})
                print("    train-fail: exploded predictions (|pred|>1e3)", flush=True); continue
            for k, Xk in Xsets.items():
                res[f"test_pred_{k}"] = predict_rc(model, Xk)
            _bad = [k for k, v in res.items() if (not np.isfinite(v).all()) or float(np.abs(v).max()) > 1e3]
            if _bad:  # ~0.8% of models validate normally but explode on a shifted eval set; such a model poisons any ensemble that includes it
                history.append({"idea": p["idea"], "ok": False, "reason": f"exploded predictions on {_bad[0]} (|pred|>1e3)"})
                print(f"    train-fail: exploded predictions on {_bad[0]} (|pred|>1e3)", flush=True); continue
            r = pear(res["test_pred"], te_oracle); vr = pear(res["val_pred"], yva)
            mid = f"r{rd:02d}_arch_{midx:02d}"
            meta = {"model_id": mid, "strategy": "arch_search", "round": rd, "idea": p["idea"],
                    "why": p.get("why", ""), "n_params": verdict.n_params, "train_config": tc,
                    "optimizer_used": used, "val_pearson": vr, "best_val_pearson": vr,
                    "test_oracle_pearson": r, "val_mse": info.get("best_val_mse"),
                    "epochs_trained": info.get("epochs_trained"),
                    "train_time_sec": time.time() - t0, "gpu": gpu, "source": p["source"], "harness": args.harness}
            if args.val_cache:
                meta["val_cache"] = {"path": _vc["path"], "sha1_16": _vc["sha1_16"], "variant": _vc["variant"], "n": _vc["n"]}
            _atomic_savez(Path(args.out_dir) / f"{mid}.npz",
                          **{k: np.asarray(x, np.float32) for k, x in res.items()})
            _atomic_write_text(Path(args.out_dir) / f"{mid}_meta.json", json.dumps(meta, indent=2, default=str))
            history.append({"idea": p["idea"], "ok": True, "score": vr, "n_params": verdict.n_params,
                            "val": vr, "train_s": time.time() - t0})
            midx += 1
            print(f"    {mid} r={r:.4f} val={vr:.4f} {verdict.n_params:,}p {used} "
                  f"{info.get('epochs_trained')}ep {time.time()-t0:.0f}s  {p['idea'][:46]}", flush=True)
    print(f"DONE_ARCH trained={midx} elapsed={(time.time()-t_start)/3600:.1f}h", flush=True)
    if not budget_hit and midx > 0:
        (out_p / "COMPLETE").write_text(f"rounds_run={rd + 1} models={midx} target={target}\n")
    if midx == 0:
        print("FAILED_ARCH: 0 models trained", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
