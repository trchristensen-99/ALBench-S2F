"""Ray Tune Trainable: subprocess wrapper around run_single.py with epoch heartbeat.

Why subprocess: in-process training requires Ray to ship the whole codebase
to workers via runtime_env packaging. The repo contains GB-scale data and
weights, so packaging hangs. Running run_single.py as a subprocess inside
the trainable function keeps Ray's package small (just this file).

The trainable:
  1. Builds the run_single.py command from the Ray config
  2. Spawns subprocess
  3. Polls history.json every 5s
  4. Reports each new epoch via tune.report (ASHA can early-stop)
  5. On ASHA stop, kills the subprocess
  6. Reads result.json on completion and reports final metric
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def _to_overrides(arch: str, config: dict[str, Any]) -> list[str]:
    """Translate abstract HP keys → run_single.py --hp k=v overrides per arch."""
    from scripts.preflight.hpsearch.hp_space import expand_block_sizes

    lr = config["lr"]
    bs = config["batch_size"]
    wd = config["weight_decay"]
    dr = config["dropout"]
    w = int(config["width"])
    d = int(config["depth"])
    out = [f"lr={lr}", f"batch_size={bs}", f"weight_decay={wd}"]
    if "optimizer" in config:
        out.append(f"optimizer={config['optimizer']}")
    if arch == "legnet":
        shape = config.get("shape", "flat")
        block_sizes = expand_block_sizes(w, d, shape)
        out.append(f"block_sizes={block_sizes}")
        # Persist shape so result.json's hp dict can be audited post-hoc.
        out.append(f"shape={shape}")
        # Prefer explicit conv_dropout (new) over legacy `dropout` when present.
        if "conv_dropout" in config:
            out.append(f"conv_dropout={config['conv_dropout']}")
        else:
            out.append(f"dropout={dr}")
        if "dense_dropout" in config:
            out.append(f"dense_dropout={config['dense_dropout']}")
        if "dense_dims" in config:
            dd = config["dense_dims"]
            dd_str = (
                "[" + ",".join(str(int(x)) for x in dd) + "]"
                if isinstance(dd, (list, tuple))
                else str(dd)
            )
            out.append(f"dense_dims={dd_str}")
        if "block_class" in config:
            out.append(f"block_class={config['block_class']}")
    elif arch == "dream_rnn":
        out.append(f"hidden_dim={w}")
        out.append(f"cnn_filters={min(w, 320)}")
        out.append(f"num_lstm_layers={d}")
        out.append(f"dropout_cnn={dr}")
        out.append(f"dropout_lstm={dr}")
    elif arch == "dream_attn":
        out.append(f"embedding_dim={w}")
        out.append(f"num_blocks={d}")
        out.append(f"first_block_dropout={dr}")
        out.append(f"core_dropout={dr}")
        out.append(f"head_dropout={dr}")
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return out


def trainable(config: dict[str, Any]):
    """Ray Tune function trainable. Spawns run_single.py subprocess."""
    from ray import tune

    # Ray Tune writes its own result.json to trial_dir. We need to keep
    # run_single.py's output (history.json, result.json, best.pt) in a
    # separate subdirectory so it doesn't clobber Ray's result.json.
    trial_dir = Path(tune.get_context().get_trial_dir())
    trial_dir.mkdir(parents=True, exist_ok=True)
    run_dir = trial_dir / "run"
    run_dir.mkdir(exist_ok=True)

    arch = config["arch"]
    # If `aug` is "rc_shift_evoaug" and intensity is 0, treat as plain rc_shift.
    # If `aug` is "rc_shift" with max_shift=0, treat as plain rev_complement.
    aug = config.get("aug", "rev_complement")
    max_shift = int(config.get("max_shift", 25 if aug.startswith("rc_shift") else 0))
    evoaug_intensity = int(config.get("evoaug_intensity", 0))
    if aug == "rc_shift_evoaug" and evoaug_intensity == 0:
        aug = "rc_shift"
    if aug == "rc_shift" and max_shift == 0:
        aug = "rev_complement"
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(REPO / "scripts/preflight/run_single.py"),
        "--arch",
        arch,
        "--d_train",
        str(config["d_train"]),
        "--seed",
        str(config["seed"]),
        "--epochs",
        str(int(config.get("epochs", 60))),
        "--early_stop_patience",
        str(int(config.get("patience", 15))),
        "--augmentations",
        aug,
        "--label_source",
        config.get("label_source", "ag_oracle"),
        "--output_dir",
        str(run_dir),
        "--sweep_name",
        f"hpsearch_{config.get('strategy', '?')}",
    ]
    if max_shift > 0:
        cmd.extend(["--max_shift", str(max_shift)])
    if evoaug_intensity > 0 and aug == "rc_shift_evoaug":
        # run_single expects {light, medium, heavy}; map our integer scale 1/2/4 → labels.
        intensity_map = {1: "light", 2: "medium", 4: "heavy"}
        cmd.extend(["--evoaug_intensity", intensity_map.get(evoaug_intensity, "light")])
        cmd.extend(["--evoaug_prob", str(config.get("evoaug_prob", 0.5))])
    # Speedup flags — set via SLURM env vars so we can toggle for a full job
    # without modifying every trial config. Round 2 turns these on by default.
    # HP_FAST=1 turns on all speedups (most jobs should use this)
    if os.environ.get("HP_FAST", "1") == "1":
        cmd.append("--fast")
    else:
        if os.environ.get("USE_COMPILE", "0") == "1":
            cmd.append("--use_compile")
        if os.environ.get("CUDNN_BENCHMARK", "0") == "1":
            cmd.append("--cudnn_benchmark")
        if os.environ.get("EVAL_ON_GPU", "0") == "1":
            cmd.append("--eval_on_gpu")
        if os.environ.get("TRAIN_ON_GPU", "0") == "1":
            cmd.append("--train_on_gpu")
        if os.environ.get("SKIP_LAST_CKPT", "0") == "1":
            cmd.append("--skip_last_ckpt")
        if os.environ.get("EVAL_TEST_EVERY"):
            cmd.extend(["--eval_test_every", os.environ["EVAL_TEST_EVERY"]])
        if os.environ.get("EVAL_BATCH_MULT"):
            cmd.extend(["--eval_batch_mult", os.environ["EVAL_BATCH_MULT"]])
    # Shared one-hot tensor cache (populated by parallel_gpu_runner pre-build
    # or by the very first trial; subsequent trials hit cache)
    cache_dir = os.environ.get("HP_CACHE_DIR", str(REPO / "outputs/tensor_cache"))
    cmd.extend(["--cache_dir", cache_dir])
    cmd.extend(["--hp", *_to_overrides(arch, config)])

    env = os.environ.copy()
    # Make sure W&B identifies trials uniquely
    env["WANDB_RUN_GROUP"] = f"{config.get('strategy', '?')}_{arch}_d{config['d_train']}"
    env["WANDB_TAGS"] = (
        f"phase=hpsearch,arch={arch},d={config['d_train']},strategy={config.get('strategy', '?')}"
    )
    env["WANDB_PROJECT"] = env.get("WANDB_PROJECT", "albench-s2f-hpsearch")

    log_path = run_dir / "subprocess.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(REPO),
        env=env,
        preexec_fn=os.setsid,  # so we can kill the process group
    )

    last_epoch = 0
    history_path = run_dir / "history.json"
    result_path = run_dir / "result.json"

    try:
        while True:
            if history_path.exists():
                try:
                    h = json.loads(history_path.read_text())
                    n = len(h.get("val_loss", []))
                    if n > last_epoch:
                        for i in range(last_epoch, n):
                            tune.report(
                                {
                                    "epoch": i + 1,
                                    "val_loss": h["val_loss"][i],
                                    "train_loss": h["train_loss"][i],
                                    "test_loss": h["test_loss"][i],
                                    "best_val": min(h["val_loss"][: i + 1]),
                                }
                            )
                        last_epoch = n
                except Exception:
                    pass
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(5)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(2)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log_f.close()

    # Read final result
    if result_path.exists():
        summary = json.loads(result_path.read_text())
        tune.report(
            {
                "epoch": summary["best_epoch"] + 1,
                "val_loss": summary["best_val_mse"],
                "test_loss": summary["test_mse_at_best_val"],
                "best_val": summary["best_val_mse"],
                "n_params": summary["n_params"],
                "gpu_hrs": summary["gpu_hrs"],
                "done": True,
            }
        )
    elif proc.returncode != 0:
        # Failed; report something so ASHA can move on
        tune.report({"val_loss": float("inf"), "done": True, "failed": True})
