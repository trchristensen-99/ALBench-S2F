"""The single training and validation loop (spec 9.1). One entry point per run:

    python -m ar2.evaluator.train --spec ws/runs/<run_id>/spec.json

Scores differ only because of the candidate, the recipe and the seed: one loop, one loader, one metric function for every run. The process exits with code 0
even for diverged / oom / exploded runs, because those are results; metrics.json is written last (tmp file + rename) and its existence is what marks a run done.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ar2.data.encode import encode, gc_fraction, revcomp_onehot
from ar2.data.loader import LabelScaler, add_label_noise, load_all
from ar2.evaluator import metrics as M
from ar2.evaluator.losses import build_loss
from ar2.evaluator.stress import load_stress_set, make_stress_set, save_stress_set, stress_check

DEFAULT_RECIPE = {
    "optimizer": "adamw",
    "lr": 0.002,
    "weight_decay": 0.01,
    "batch_size": 512,
    "schedule": "onecycle",
    "warmup_frac": 0.05,
    "max_epochs": 40,
    "patience": 8,
    "grad_clip": 1.0,
    "loss": {"name": "mse"},
    "rc_augment": True,
    "ema": None,
    "amp": True,
    "tasks": ["K562"],
    "hparams": {},
}
NO_EARLY_TERM_PURPOSES = {"first_look", "confirm", "final", "screen", "procedure"}


class _Log:
    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        self.fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def close(self) -> None:
        self.fh.close()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_model_module(code_dir: str | Path):
    """Import ``code_dir/model.py`` (evaluator subprocesses are one of the two places candidate code may run)."""
    path = Path(code_dir) / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"ar2_candidate_{abs(hash(str(path)))}", str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def build_model(code_dir, out_dim: int, hparams: dict | None):
    mod = load_model_module(code_dir)
    hp = dict(getattr(mod, "HPARAMS", {}))
    hp.update(hparams or {})
    return mod.build_model(out_dim, **hp)


def build_optimizer(model, recipe: dict):
    import torch

    name = str(recipe.get("optimizer", "adamw")).lower()
    lr, wd = float(recipe["lr"]), float(recipe["weight_decay"])
    if name == "muon":
        try:
            from muon import SingleDeviceMuonWithAuxAdam
        except ImportError as e:  # no silent fallback to AdamW: it would train a different model than the recipe says
            raise RuntimeError(
                "recipe asks for the Muon optimizer but the `muon` package is not importable"
            ) from e
        p2 = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
        p1 = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
        groups = [dict(params=p2, use_muon=True, lr=lr, weight_decay=wd, momentum=0.95)]
        if p1:
            groups.append(
                dict(params=p1, use_muon=False, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
            )
        return SingleDeviceMuonWithAuxAdam(groups)
    cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}.get(name)
    if cls is None:
        raise ValueError(f"unknown optimizer {name!r}")
    return cls(model.parameters(), lr=lr, weight_decay=wd)


def lr_factor(schedule: str, step: int, total: int, warmup_frac: float) -> float:
    """Multiplier on the base learning rate at ``step`` of ``total`` (per-step schedules; length = max_epochs * steps per epoch)."""
    total = max(total, 1)
    t = min(step, total)
    if schedule == "onecycle":
        pct = max(float(warmup_frac), 0.02)
        initial, final = 1 / 25.0, 1 / (25.0 * 1e3)
        if t < pct * total:
            frac = t / (pct * total)
            return initial + (1 - initial) * 0.5 * (1 - math.cos(math.pi * frac))
        frac = (t - pct * total) / max((1 - pct) * total, 1)
        return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * frac))
    if schedule == "cosine_warmup":
        w = float(warmup_frac) * total
        if w > 0 and t < w:
            return max(t / w, 1e-3)
        frac = (t - w) / max(total - w, 1)
        return max(0.5 * (1 + math.cos(math.pi * frac)), 1e-3)
    raise ValueError(f"unknown schedule {schedule!r}")


class _Averager:
    """EMA (decay) or equal-weight average (decay None) of weights, evaluated as a copy of the model with the live model's buffers."""

    def __init__(self, model, decay: float | None):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    def update(self, model) -> None:
        import torch

        self.n += 1
        d = self.decay if self.decay is not None else 1.0 - 1.0 / self.n
        if self.decay is not None:
            d = min(self.decay, (1 + self.n) / (10 + self.n))
        with torch.no_grad():
            for pe, pm in zip(self.module.parameters(), model.parameters(), strict=True):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(self.module.buffers(), model.buffers(), strict=True):
                be.copy_(bm)


def _amp(device: str, enabled: bool):
    """(autocast context factory, GradScaler or None). bf16 where supported (H100), fp16 + scaler otherwise (V100)."""
    import torch

    if not enabled or not device.startswith("cuda"):
        return (lambda: nullcontext()), None
    if torch.cuda.is_bf16_supported():
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None
    return (lambda: torch.autocast("cuda", dtype=torch.float16)), torch.amp.GradScaler("cuda")


def predict_z(model, X, loss, rc_average: bool = True, bs: int = 1024, amp_ctx=None) -> np.ndarray:
    """z-scored point predictions of the primary task, averaged over the forward and reverse-complement strands (spec 2.1)."""
    import torch

    amp_ctx = amp_ctx or (lambda: nullcontext())
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i : i + bs]
            with amp_ctx():
                p = loss.predict(model(xb).float())
                if rc_average:
                    p = (p + loss.predict(model(revcomp_onehot(xb)).float())) / 2
            outs.append((p[:, 0] if p.ndim == 2 else p).float().cpu())
    return torch.cat(outs).numpy() if outs else np.zeros(0, np.float32)


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: None if o != o else float(o)))
    os.replace(tmp, path)


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def run(spec: dict) -> dict:
    """Execute one run from a spec dict and write its outputs; returns the metrics dict."""
    import torch

    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(out_dir / "log.txt")
    t_wall = time.perf_counter()
    recipe = {**DEFAULT_RECIPE, **spec["recipe"]}
    seed = int(spec.get("seed", 0))
    device = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    cap_s = (
        (float(spec["time_cap_s"]) * float(spec.get("speed_factor", 1.0)))
        if spec.get("time_cap_s")
        else float("inf")
    )
    metrics: dict = {
        "run_id": spec["run_id"],
        "status": "error",
        "stop_reason": None,
        "val_r": None,
        "purpose": spec.get("purpose"),
        "seed": seed,
        "cap_s_used": None if cap_s == float("inf") else cap_s,
        "censored": False,
        "device": device,
    }
    try:
        log(
            f"run {spec['run_id']} seed {seed} device {device} torch {torch.__version__} python {platform.python_version()} numpy {np.__version__}"
        )
        if device.startswith("cuda"):
            log(
                f"gpu {torch.cuda.get_device_name(0)} cudnn {torch.backends.cudnn.version()} cudnn.benchmark={torch.backends.cudnn.benchmark}"
            )
            torch.cuda.reset_peak_memory_stats()
        seed_everything(seed)
        tr, vl, vs = load_all(
            spec["train_manifest"], spec["val_loop_manifest"], spec.get("val_select_manifest")
        )
        tasks = list(recipe.get("tasks") or [tr.tasks[0]])
        cols = [tr.tasks.index(t) for t in tasks]
        y_tr_raw = add_label_noise(
            tr.labels[:, cols], float(spec.get("label_noise_sd") or 0.0), seed
        )
        scaler = LabelScaler.fit(tr.labels[:, cols])  # training statistics only
        Ytr = torch.from_numpy(scaler.transform(y_tr_raw).reshape(len(tr), -1)).to(device)
        y_val = vl.labels[:, vl.tasks.index(tasks[0])]
        Xtr = torch.from_numpy(encode(tr.seqs)).to(device)
        Xva = torch.from_numpy(encode(vl.seqs)).to(device)
        z0 = Ytr[:, 0].cpu().numpy()
        loss = build_loss(
            recipe.get("loss"), len(tasks), (float(z0.min()) - 0.5, float(z0.max()) + 0.5)
        )
        model = build_model(spec["code_dir"], loss.out_dim, recipe.get("hparams")).to(device)
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        amp_ctx, scaler_amp = _amp(device, bool(recipe.get("amp", True)))
        opt = build_optimizer(model, recipe)
        N, bs, E = len(tr), int(recipe["batch_size"]), int(recipe["max_epochs"])
        spe = math.ceil(N / bs)
        total_steps = E * spe
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: lr_factor(recipe["schedule"], s, total_steps, recipe.get("warmup_frac", 0.0)),
        )
        ema = _Averager(model, float(recipe["ema"])) if recipe.get("ema") else None
        swa = _Averager(model, None) if recipe.get("swa") else None
        gen = torch.Generator().manual_seed(seed)
        et = spec.get("early_term") or {}
        et_ref = (et.get("reference") or {}).get(recipe["schedule"], {})
        et_on = bool(et_ref) and spec.get("purpose") not in NO_EARLY_TERM_PURPOSES
        et_cp = {int(round(float(f) * E)): float(f) for f in et.get("checkpoints", [0.5, 0.75])}
        et_ref = {float(k): float(v) for k, v in et_ref.items()}

        best_r, best_state, best_ev, best_epoch, bad, stop_reason, status = (
            -float("inf"),
            None,
            None,
            0,
            0,
            "max_epochs",
            "ok",
        )
        val_curve: list[float] = []
        train_seconds, t_train0 = 0.0, time.perf_counter()
        rows = []
        step = 0
        for epoch in range(1, E + 1):
            model.train()
            t_ep = time.perf_counter()
            perm = torch.randperm(N, generator=gen)
            flip = torch.rand(N, generator=gen) < 0.5 if recipe.get("rc_augment") else None
            run_loss, n_b, diverged, capped = 0.0, 0, False, False
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                if len(idx) < 2:
                    continue
                xb, yb = Xtr[idx.to(device)], Ytr[idx.to(device)]
                if flip is not None:
                    fm = flip[idx].to(device)[:, None, None]
                    xb = torch.where(fm, revcomp_onehot(xb), xb)
                with amp_ctx():
                    out = model(xb)
                lv = loss(out.float(), yb)
                if not torch.isfinite(lv):
                    diverged = True
                    break
                opt.zero_grad(set_to_none=True)
                if scaler_amp is not None:
                    scaler_amp.scale(lv).backward()
                    scaler_amp.unscale_(opt)
                else:
                    lv.backward()
                if recipe.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
                if scaler_amp is not None:
                    scaler_amp.step(opt)
                    scaler_amp.update()
                else:
                    opt.step()
                sched.step()
                step += 1
                if ema is not None:
                    ema.update(model)
                run_loss += float(lv.detach())
                n_b += 1
                if time.perf_counter() - t_train0 > cap_s:
                    capped = True
                    break
            train_seconds += time.perf_counter() - t_ep
            if diverged:
                status, stop_reason = "diverged", "diverged"
                log(f"epoch {epoch}: non-finite loss, stopping")
                break
            if swa is not None and epoch > int(0.75 * E):
                swa.update(model)
            ev = (
                ema.module
                if ema is not None
                else (swa.module if swa is not None and swa.n > 0 else model)
            )
            zv = predict_z(ev, Xva, loss, bool(spec.get("rc_average", True)), amp_ctx=amp_ctx)
            r = (
                M.pearson(zv, (y_val - scaler.mu[0]) / scaler.sd[0])
                if np.isfinite(zv).all()
                else float("nan")
            )
            val_curve.append(r)
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": run_loss / max(n_b, 1),
                    "val_r": r,
                    "lr": opt.param_groups[0]["lr"],
                    "elapsed_s": time.perf_counter() - t_train0,
                }
            )
            log(
                f"epoch {epoch} loss {rows[-1]['train_loss']:.4f} val_r {r:.4f} lr {rows[-1]['lr']:.2e}"
            )
            if not math.isfinite(r):
                status, stop_reason = "diverged", "diverged"
                break
            if r > best_r:
                best_r, best_epoch, bad = r, epoch, 0
                best_state, best_ev = (
                    {k: v.detach().clone() for k, v in ev.state_dict().items()},
                    ev,
                )
            else:
                bad += 1
            if (
                et_on
                and epoch in et_cp
                and et_cp[epoch] in et_ref
                and best_r < et_ref[et_cp[epoch]] - float(et.get("margin", 0.03))
            ):
                status, stop_reason = "early_stopped_by_tuner", "tuner_early_term"
                log(
                    f"epoch {epoch}: best {best_r:.4f} is more than the margin below the reference {et_ref[et_cp[epoch]]:.4f}; stopped by the tuner"
                )
                break
            if capped or (time.perf_counter() - t_train0 > cap_s):
                status, stop_reason = "time_cap", "time_cap"
                metrics["censored"] = True
                log(f"epoch {epoch}: time cap {cap_s:.0f}s reached")
                break
            if bad >= int(recipe["patience"]):
                stop_reason = "early_stop"
                break
        epochs_run = len(rows)
        metrics.update(
            n_params=n_params,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            cost_equiv=epochs_run / E if E else 1.0,
            stop_reason=stop_reason,
            throughput_seq_per_s=(N * epochs_run / train_seconds) if train_seconds > 0 else None,
            peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9)
            if device.startswith("cuda")
            else 0.0,
        )
        if best_state is None:
            metrics.update(
                status=status if status != "ok" else "diverged",
                error="no epoch produced a finite validation score",
            )
        else:
            state, ev = best_state, best_ev
            ev.load_state_dict(state)
            ev.eval()
            zv = predict_z(ev, Xva, loss, bool(spec.get("rc_average", True)), amp_ctx=amp_ctx)
            pv = scaler.inverse(zv)
            sub = np.random.default_rng(0).choice(N, min(N, 10_000), replace=False)
            ztr = predict_z(
                ev,
                Xtr[torch.from_numpy(sub).to(device)],
                loss,
                bool(spec.get("rc_average", True)),
                amp_ctx=amp_ctx,
            )
            if spec.get("stress_path"):
                sp = Path(spec["stress_path"])
                if not sp.exists():
                    save_stress_set(sp, make_stress_set(vl.seqs))
                Xs = torch.from_numpy(encode(load_stress_set(sp))).to(device)
                stress = stress_check(
                    predict_z(ev, Xs, loss, bool(spec.get("rc_average", True)), amp_ctx=amp_ctx),
                    float(spec.get("stress_max_abs_z", 10.0)),
                )
            else:
                stress = {"finite": True, "max_abs_z": 0.0, "exploded": False}
            if stress["exploded"]:
                status = "exploded"
            y_tr_task0 = tr.labels[sub, cols[0]]
            metrics.update(
                status=status,
                val_r=_nan_none(M.pearson(pv, y_val)),
                val_mse=_nan_none(M.mse(pv, y_val)),
                val_spearman=_nan_none(M.spearman(pv, y_val)),
                train_r=_nan_none(M.pearson(scaler.inverse(ztr), y_tr_task0)),
                still_improving=M.still_improving(val_curve),
                pred_sd_ratio=_nan_none(M.pred_sd_ratio(pv, y_val)),
                mse_by_activity_quintile=M.mse_by_activity_quintile(pv, y_val),
                mse_by_gc_tercile=M.mse_by_gc_tercile(pv, y_val, gc_fraction(vl.seqs)),
                stress=stress,
            )
            if vs is not None:  # Phase D only: ordinary specs carry no val_select path
                Xs_sel = torch.from_numpy(encode(vs.seqs)).to(device)
                ps = scaler.inverse(
                    predict_z(ev, Xs_sel, loss, bool(spec.get("rc_average", True)), amp_ctx=amp_ctx)
                )
                metrics["val_select_r"] = _nan_none(
                    M.pearson(ps, vs.labels[:, vs.tasks.index(tasks[0])])
                )
            if spec.get("save_preds", True):
                np.save(out_dir / "preds_val_loop.npy", pv.astype(np.float32))
            if spec.get("save_ckpt"):
                torch.save(
                    {"state_dict": state, "recipe": recipe, "mu": scaler.mu, "sd": scaler.sd},
                    out_dir / "ckpt.pt",
                )
        with open(out_dir / "curves.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_r", "lr", "elapsed_s"])
            w.writeheader()
            w.writerows(rows)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        metrics.update(
            status="oom" if oom else "error", error=msg, traceback=traceback.format_exc()[-4000:]
        )
        if metrics.get("stop_reason") is None:
            metrics["stop_reason"] = "oom" if oom else "error"
        log(f"FAILED {msg}")
    metrics["wall_time_s"] = time.perf_counter() - t_wall
    log(
        f"done status={metrics['status']} val_r={metrics.get('val_r')} wall={metrics['wall_time_s']:.0f}s"
    )
    log_close = getattr(log, "close", None)
    _atomic_json(
        out_dir / "metrics.json",
        {k: _nan_none(v) if not isinstance(v, (dict, list)) else v for k, v in metrics.items()},
    )  # metrics.json written last
    if log_close:
        log_close()
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one ar2 training/evaluation job")
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    run(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
