"""In-process N-model trainer.

Holds N LegNet/DREAM-ATTN models in ONE Python process and trains them in
lockstep on the same shared batch. Eliminates subprocess overhead
(Python+Torch startup, repeated load_data, per-trial CUDA context init),
and shares a single dataset tensor on GPU.

vs parallel_gpu_runner.py:
- 1× Python startup     (vs N×)
- 1× torch import       (vs N×)
- 1× load_data + cache  (vs N× — though shared cache helps subprocess too)
- 1× host-→GPU transfer per batch (vs N×)
- Single CUDA context   (vs N independent contexts contending for the SM)

Constraints:
- All configs must share (d_train, seed, label_source, in_channels,
  aug, payload_len). Heterogeneous data ⇒ falls back to subprocess.
- Shared batch_size for the DataLoader (auto = max of configs).
- Each model has its own optimizer, scheduler, LR, weight_decay,
  block_class, dropout, etc.

Output (per config):
  output_dir/best.pt           — best-val state_dict
  output_dir/history.json      — per-epoch train/val/test loss
  output_dir/result.json       — summary matching run_single's schema
  output_dir/config.json       — resolved HPs

Usage:
    uv run --no-sync python scripts/preflight/inprocess_runner.py \\
        configs.json [--shared_batch_size 1024]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

# Import shared helpers from run_single without invoking its main()
from scripts.preflight.run_single import (  # noqa: E402
    ARCH_PRIORS,
    _eval_loss,
    _make_optimizer,
    _rc_flip,
    _shift_window_crop,
    build_model,
    load_data,
    parse_overrides,
    set_seed,
)


def _resolve_hp(arch: str, overrides: list[str]) -> dict[str, Any]:
    hp = dict(ARCH_PRIORS[arch])
    hp.update(parse_overrides(overrides))
    return hp


def _check_homogeneous(configs: list[dict]) -> tuple[int, int, str, int, str]:
    """All configs must share d_train, seed, label_source, in_channels, aug.
    Returns the shared (d_train, seed, label_source, in_channels, aug)."""
    keys = set()
    for cfg in configs:
        arch = cfg.get("arch", "legnet")
        ic = ARCH_PRIORS[arch].get("in_channels", 4)
        hp_dict = _resolve_hp(arch, cfg.get("hp_overrides", []))
        # Honor in_channels override if present
        ic = hp_dict.get("in_channels", ic)
        keys.add(
            (
                cfg.get("d_train"),
                cfg.get("seed"),
                cfg.get("label_source", "ag_oracle"),
                ic,
                cfg.get("aug", "rev_complement"),
            )
        )
    if len(keys) != 1:
        raise SystemExit(
            f"inprocess_runner requires homogeneous data; got {len(keys)} "
            f"distinct (d_train, seed, label_source, in_channels, aug) tuples. "
            f"Use parallel_gpu_runner.py for heterogeneous cells."
        )
    return next(iter(keys))


def _summary_for(
    cfg: dict,
    hp: dict,
    history: dict,
    best_val: float,
    best_epoch: int,
    best_test: float,
    n_params: int,
    gpu_hrs: float,
) -> dict:
    """Mirror the summary schema written by run_single.train()."""
    epochs = cfg.get("epochs", 60)
    final_window_start = int(epochs * 0.9)  # 10% final-window flag
    return {
        "arch": cfg["arch"],
        "d_train": cfg["d_train"],
        "seed": cfg["seed"],
        "epochs": epochs,
        "n_params": n_params,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "test_mse_at_best_val": best_test,
        "min_val_in_final_pct_window": best_epoch >= final_window_start,
        "report_min_val_in_final_pct": 0.1,
        "gpu_hrs": gpu_hrs,
        "augmentations": cfg.get("aug", "rev_complement"),
        "hp": hp,
        "wandb_run_id": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs_json")
    ap.add_argument(
        "--shared_batch_size",
        type=int,
        default=0,
        help="Override DataLoader batch_size (0 = max of configs)",
    )
    ap.add_argument(
        "--bf16", action="store_true", default=None, help="Use bf16 autocast (auto-on for Ampere+)"
    )
    ap.add_argument(
        "--val_subsample", type=int, default=0, help="Eval on first N val/test rows (0 = all)"
    )
    ap.add_argument("--cache_dir", type=str, default=str(REPO / "outputs" / "tensor_cache"))
    args = ap.parse_args()

    configs = json.loads(Path(args.configs_json).read_text())
    if not configs:
        print("No configs to run.")
        return

    # Filter out already-completed configs
    pending = []
    for cfg in configs:
        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = REPO / out_dir
        if (out_dir / "result.json").exists():
            print(f"  [skip] {cfg.get('label', '?')} (already done)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg["_out_dir"] = out_dir
        pending.append(cfg)
    if not pending:
        print("All configs already completed.")
        return

    d_train, seed, label_source, in_channels, aug = _check_homogeneous(pending)
    print(
        f"Inprocess runner: {len(pending)} models, d_train={d_train}, seed={seed}, "
        f"label_source={label_source}, in_channels={in_channels}, aug={aug}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Decide bf16 default: auto-on for Ampere+ (compute capability ≥ 8.0)
    if args.bf16 is None:
        args.bf16 = (
            device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
        )
    print(f"Device: {device}  bf16={args.bf16}")

    # cudnn benchmark + TF32 (always wins for tiny models)
    set_seed(seed, cudnn_benchmark=True)

    augment_rc = aug in ("rev_complement", "rc_shift", "rc_shift_evoaug")
    use_shift = aug in ("rc_shift", "rc_shift_evoaug")
    max_shift = 0
    payload_len = 200
    # Adapter-padded inputs needed when use_shift=True
    pad_with_adapters = use_shift
    if use_shift:
        # First config's hp.max_shift (all should agree; if they don't, take min)
        max_shift = min(
            _resolve_hp(cfg["arch"], cfg.get("hp_overrides", [])).get("max_shift", 25)
            for cfg in pending
        )

    # Load data once
    t0 = time.time()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_data(
        d_train,
        seed,
        in_channels=in_channels,
        seq_len=payload_len + (2 * max_shift if use_shift else 0),
        label_source=label_source,
        pad_with_adapters=pad_with_adapters,
        cache_dir=args.cache_dir,
    )
    if args.val_subsample > 0:
        Xva = Xva[: args.val_subsample]
        yva = yva[: args.val_subsample]
        Xte = Xte[: args.val_subsample]
        yte = yte[: args.val_subsample]
    print(
        f"  data loaded in {time.time() - t0:.1f}s  "
        f"train={len(Xtr):,}  val={len(Xva):,}  test={len(Xte):,}"
    )

    # Move all tensors to GPU once. LegNet at D=100k = 100k×4×200×4B = 320 MB — fits.
    Xtr_g = torch.as_tensor(Xtr).to(device, non_blocking=True)
    ytr_g = torch.as_tensor(ytr).to(device, non_blocking=True)
    Xva_g = torch.as_tensor(Xva).to(device, non_blocking=True)
    yva_g = torch.as_tensor(yva).to(device, non_blocking=True)
    Xte_g = torch.as_tensor(Xte).to(device, non_blocking=True)
    yte_g = torch.as_tensor(yte).to(device, non_blocking=True)

    # Shared DataLoader batch_size — max of per-config batch_sizes (within reason).
    cfg_bss = [
        _resolve_hp(cfg["arch"], cfg.get("hp_overrides", [])).get("batch_size", 1024)
        for cfg in pending
    ]
    shared_bs = args.shared_batch_size or max(cfg_bss)
    if shared_bs != min(cfg_bss):
        print(
            f"  NOTE: configs have batch_size in {sorted(set(cfg_bss))}; "
            f"using shared bs={shared_bs} for the DataLoader."
        )

    # Build models + optimizers + schedulers
    n_train = len(Xtr_g)
    epochs_per_cfg = [cfg.get("epochs", 60) for cfg in pending]
    max_epochs = max(epochs_per_cfg)
    n_batches = max(1, math.ceil(n_train / shared_bs))
    print(f"  max_epochs={max_epochs}  n_batches/ep={n_batches}  shared_bs={shared_bs}")

    models = []
    for cfg in pending:
        arch = cfg["arch"]
        hp = _resolve_hp(arch, cfg.get("hp_overrides", []))
        # Use different model seed per slot so weights differ; data order is shared.
        slot = len(models)
        torch.manual_seed(seed + slot)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + slot)
        m = build_model(arch, hp, device)
        n_params = sum(p.numel() for p in m.parameters())
        opt = _make_optimizer(m, hp.get("optimizer", "adamw"), hp["lr"], hp["weight_decay"])
        # OneCycleLR over THIS model's epochs (independent schedule even though
        # we share the data iter; LR ends at its own total_steps).
        total_steps = max(1, epochs_per_cfg[slot] * n_batches)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=hp["lr"],
            total_steps=total_steps,
            pct_start=hp.get("warmup_pct", 0.1),
            anneal_strategy="cos",
        )
        models.append(
            {
                "cfg": cfg,
                "hp": hp,
                "model": m,
                "opt": opt,
                "sched": sched,
                "n_params": n_params,
                "history": {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []},
                "best_val": math.inf,
                "best_epoch": -1,
                "best_test": math.nan,
                "stopped": False,
                "stop_reason": None,
                "t_start": time.time(),
                "epochs": epochs_per_cfg[slot],
                "patience": cfg.get("patience", 10),
            }
        )
        print(
            f"  [{slot}] {cfg.get('label', '?')}  arch={arch}  "
            f"params={n_params:,}  lr={hp['lr']:.4g}  wd={hp['weight_decay']:.4g}  "
            f"block_class={hp.get('block_class', '-')}  opt={hp.get('optimizer', 'adamw')}"
        )

    criterion = torch.nn.MSELoss()
    # AMP only used when bf16 is supported (no GradScaler in this runner).
    # On pre-Ampere GPUs (no bf16), fp32 path is more stable than fp16
    # without scaling (which overflows for some block classes).
    use_amp = device.type == "cuda" and args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Pre-shuffled indices (one shuffle stream per epoch — shared across models)
    rng = torch.Generator(device=device if device.type == "cuda" else "cpu")
    rng.manual_seed(seed)

    # Build small val/test loaders over GPU tensors using simple index batching.
    def _eval_on_gpu(model_m, X_g, y_g, eval_bs=2048):
        model_m.eval()
        preds = []
        targets = []
        with torch.inference_mode():
            for i in range(0, len(X_g), eval_bs):
                xb = X_g[i : i + eval_bs]
                yb = y_g[i : i + eval_bs]
                if use_shift:
                    xb = _shift_window_crop(xb, payload_len, max_shift, training=False)
                yhat = model_m(xb).reshape(-1).clone()
                if augment_rc:
                    yhat_rc = model_m(_rc_flip(xb)).reshape(-1).clone()
                    yhat = 0.5 * (yhat + yhat_rc)
                preds.append(yhat.detach())
                targets.append(yb.detach())
        p = torch.cat(preds).float().cpu().numpy()
        t = torch.cat(targets).float().cpu().numpy()
        return float(np.mean((p - t) ** 2))

    t_global = time.time()
    for epoch in range(max_epochs):
        # Skip epoch if every active model has already stopped
        active = [s for s in models if not s["stopped"] and epoch < s["epochs"]]
        if not active:
            break

        # Single shared shuffle per epoch
        perm = torch.randperm(n_train, generator=rng, device=Xtr_g.device)

        epoch_losses = {id(s): 0.0 for s in active}
        epoch_n = {id(s): 0 for s in active}
        for b in range(n_batches):
            idx = perm[b * shared_bs : (b + 1) * shared_bs]
            xb_base = Xtr_g[idx]
            yb = ytr_g[idx]
            # Apply per-batch augmentations ONCE (shared across all models).
            if use_shift:
                xb_base = _shift_window_crop(xb_base, payload_len, max_shift, training=True)
            if augment_rc and torch.rand(1, generator=rng, device=Xtr_g.device).item() < 0.5:
                xb_base = _rc_flip(xb_base)

            for s in active:
                m = s["model"]
                opt = s["opt"]
                sched = s["sched"]
                m.train()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    yhat = m(xb_base).reshape(-1)
                    loss = criterion(yhat, yb)
                loss.backward()
                opt.step()
                sched.step()
                epoch_losses[id(s)] += loss.item()
                epoch_n[id(s)] += 1

        # End-of-epoch eval + best-tracking + early-stop check, per model
        for s in active:
            train_loss = epoch_losses[id(s)] / max(1, epoch_n[id(s)])
            val_loss = _eval_on_gpu(s["model"], Xva_g, yva_g)
            is_new_best = val_loss < s["best_val"]
            test_loss = (
                _eval_on_gpu(s["model"], Xte_g, yte_g)
                if is_new_best
                else (s["history"]["test_loss"][-1] if s["history"]["test_loss"] else float("nan"))
            )
            s["history"]["train_loss"].append(train_loss)
            s["history"]["val_loss"].append(val_loss)
            s["history"]["test_loss"].append(test_loss)
            s["history"]["lr"].append(s["opt"].param_groups[0]["lr"])
            if is_new_best:
                s["best_val"] = val_loss
                s["best_epoch"] = epoch
                s["best_test"] = test_loss
                out_dir = s["cfg"]["_out_dir"]
                torch.save(
                    {"epoch": epoch, "state_dict": s["model"].state_dict(), "hp": s["hp"]},
                    out_dir / "best.pt",
                )
            # Early stop
            since_best = epoch - s["best_epoch"]
            if s["patience"] > 0 and since_best >= s["patience"]:
                s["stopped"] = True
                s["stop_reason"] = f"early-stop (no improvement in {s['patience']} ep)"
            if epoch + 1 >= s["epochs"]:
                s["stopped"] = True
                s["stop_reason"] = s["stop_reason"] or "max_epochs reached"

        # Progress line
        elapsed = time.time() - t_global
        line = f"  ep {epoch + 1}  ({elapsed:.0f}s elapsed)  "
        line += "  ".join(
            f"[{i}]v={s['history']['val_loss'][-1]:.4f}"
            + ("*" if s["best_epoch"] == epoch else "")
            + ("!" if s["stopped"] else "")
            for i, s in enumerate(models)
            if s["history"]["val_loss"]
        )
        print(line)

    # Write per-config result.json + history.json
    for s in models:
        out_dir = s["cfg"]["_out_dir"]
        gpu_hrs = (time.time() - s["t_start"]) / 3600.0
        summary = _summary_for(
            s["cfg"],
            s["hp"],
            s["history"],
            s["best_val"],
            s["best_epoch"],
            s["best_test"],
            s["n_params"],
            gpu_hrs,
        )
        (out_dir / "history.json").write_text(json.dumps(s["history"], indent=2))
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "arch": s["cfg"]["arch"],
                    "d_train": s["cfg"]["d_train"],
                    "seed": s["cfg"]["seed"],
                    "epochs": s["epochs"],
                    "patience": s["patience"],
                    "aug": s["cfg"].get("aug"),
                    "hp": s["hp"],
                    "stop_reason": s["stop_reason"],
                },
                indent=2,
            )
        )

    total = time.time() - t_global
    print(f"\nTotal wall: {total:.0f}s ({total / 60:.1f} min)  N_models={len(models)}")
    for i, s in enumerate(models):
        print(
            f"  [{i}] {s['cfg'].get('label', '?'):<40}  "
            f"best_val={s['best_val']:.4f}  test={s['best_test']:.4f}  "
            f"@ ep {s['best_epoch'] + 1}  ({s['stop_reason']})"
        )


if __name__ == "__main__":
    main()
