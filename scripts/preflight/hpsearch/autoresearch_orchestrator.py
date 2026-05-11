"""AutoResearch orchestrator.

Bootstraps round 0 with diverse exploration configs (since no prior trials
exist). For rounds 1+, subagents propose based on current_state.json which
the main Claude process digests from completed trials.

This script handles the deterministic plumbing:
  - generate_round_0_proposals(cell_dir, arch, d_train, role)
       writes proposed_configs.json with 5 role-appropriate seed configs
  - convert_to_runner(cell_dir, arch, d_train, role, round_idx)
       reads proposed_configs.json → writes runner_configs.json for parallel_gpu_runner
  - submit_round(cell_dir, round_idx)
       submits SLURM jobs for all 3 roles, returns job IDs

Subagent invocation (Agent() tool calls) is done by the main Claude
process, which reads the briefing from autoresearch.py:make_briefing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results/preflight/hpsearch/autoresearch"


# Seed configs per role for round 0 (no prior trials available)
# Role A (explore): corners + edges
# Role B (exploit but no prior): well-known sensible defaults
# Role C (ablate): center + one-axis perturbations
def _round_0_seeds(arch: str, role: str, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    from scripts.preflight.hpsearch.hp_space import sample_random

    if role == "A":
        # Explore: 5 diverse configs spanning corners
        configs = []
        for _ in range(5):
            cfg = sample_random(arch, rng)
            configs.append(cfg)
        return configs
    if role == "B":
        # Exploit defaults: known sensible per-arch
        if arch == "legnet":
            base = {
                "lr": 5e-3,
                "batch_size": 1024,
                "weight_decay": 0.1,
                "dropout": 0.1,
                "width": 512,
                "depth": 4,
            }
            perturbations = [
                {"lr": 3e-3, "width": 1024, "depth": 3},
                {"lr": 1e-3, "batch_size": 512, "depth": 5},
                {"weight_decay": 0.01, "dropout": 0.2, "width": 256},
                {"lr": 7e-3, "batch_size": 2048 if 2048 <= 1024 else 1024, "depth": 3},
            ]
        elif arch == "dream_rnn":
            base = {
                "lr": 1e-3,
                "batch_size": 1024,
                "weight_decay": 0.01,
                "dropout": 0.2,
                "width": 320,
                "depth": 1,
            }
            base["width"] = 256  # nearest valid in search space
            perturbations = [
                {"lr": 5e-4, "width": 128, "depth": 2},
                {"lr": 2e-3, "width": 512, "depth": 1},
                {"weight_decay": 0.001, "dropout": 0.1, "width": 256},
                {"batch_size": 512, "depth": 3, "dropout": 0.3},
            ]
        else:  # dream_attn
            base = {
                "lr": 3e-4,
                "batch_size": 512,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "width": 256,
                "depth": 4,
            }
            perturbations = [
                {"lr": 1e-4, "width": 512, "depth": 4},
                {"lr": 5e-4, "batch_size": 256, "depth": 6},
                {"weight_decay": 0.001, "dropout": 0.2, "width": 128},
                {"batch_size": 1024, "depth": 2, "dropout": 0.0},
            ]
        configs = [dict(base)]
        for p in perturbations:
            cfg = dict(base)
            cfg.update(p)
            configs.append(cfg)
        return configs[:5]
    if role == "C":
        # Ablate: center config + 4 single-axis perturbations
        if arch == "legnet":
            center = {
                "lr": 1e-3,
                "batch_size": 256,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "width": 512,
                "depth": 4,
            }
        elif arch == "dream_rnn":
            center = {
                "lr": 1e-3,
                "batch_size": 256,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "width": 256,
                "depth": 2,
            }
        else:
            center = {
                "lr": 3e-4,
                "batch_size": 256,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "width": 256,
                "depth": 4,
            }
        configs = [dict(center)]
        # vary one HP at a time
        axes = [
            ("lr", 1e-2),
            ("batch_size", 1024),
            ("weight_decay", 1e-4),
            ("dropout", 0.3),
        ]
        for axis, val in axes:
            cfg = dict(center)
            cfg[axis] = val
            configs.append(cfg)
        return configs
    raise ValueError(f"Unknown role: {role}")


def round_0_setup(arch: str, d_train: int, seed: int = 42) -> Path:
    """Generate round 0 proposals for all 3 roles. Returns cell_dir."""
    cell_dir = ROOT / f"{arch}_d{d_train}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    for role in ["A", "B", "C"]:
        role_dir = cell_dir / "round_0" / f"agent_{role}"
        role_dir.mkdir(parents=True, exist_ok=True)
        configs = _round_0_seeds(arch, role, seed)
        (role_dir / "proposed_configs.json").write_text(json.dumps(configs, indent=2))
    return cell_dir


def proposals_to_runner_cfg(
    cell_dir: Path,
    arch: str,
    d_train: int,
    role: str,
    round_idx: int,
    seed: int = 42,
    epochs: int = 60,
    patience: int = 15,
) -> Path:
    """Convert proposed_configs.json → parallel_gpu_runner configs.json."""
    from scripts.preflight.hpsearch.hp_space import to_run_single_overrides

    role_dir = cell_dir / f"round_{round_idx}" / f"agent_{role}"
    proposed = json.loads((role_dir / "proposed_configs.json").read_text())
    cells = []
    for i, hp in enumerate(proposed):
        label = f"ar_{role}_r{round_idx}_{arch}_d{d_train}_t{i}"
        out_dir = role_dir / label
        overrides = to_run_single_overrides(arch, hp)
        cells.append(
            {
                "label": label,
                "arch": arch,
                "d_train": d_train,
                "seed": seed,
                "epochs": epochs,
                "patience": patience,
                "aug": "rev_complement",
                "output_dir": str(out_dir.relative_to(REPO)),
                "hp_overrides": overrides,
            }
        )
    cfg_path = role_dir / "runner_configs.json"
    cfg_path.write_text(json.dumps(cells, indent=2))
    return cfg_path


def submit_round(
    cell_dir: Path,
    arch: str,
    d_train: int,
    round_idx: int,
    qos: str = "default",
) -> dict[str, str]:
    """Submit one SLURM job per role (3 jobs). Returns {role: job_id}."""
    jids = {}
    for role in ["A", "B", "C"]:
        cfg_path = proposals_to_runner_cfg(cell_dir, arch, d_train, role, round_idx)
        # Time limit: D=5k cell ~ 1.5h, D=100k cell ~ 6h
        if d_train >= 50000:
            timelimit = "12:00:00"
        else:
            timelimit = "06:00:00"

        jobname = f"ar_{role}_{arch:.3}_d{d_train}_r{round_idx}"
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={REPO}/logs/%x-%j.out
#SBATCH --error={REPO}/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time={timelimit}
#SBATCH --mem=80G

export CFG_PATH={cfg_path}
bash {REPO}/scripts/slurm/autoresearch_cell.sh
"""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        tmp.write(sbatch_script)
        tmp.close()
        result = subprocess.run(
            ["/cm/shared/apps/slurm/current/bin/sbatch", "--parsable", tmp.name],
            capture_output=True,
            text=True,
        )
        Path(tmp.name).unlink()
        if result.returncode != 0:
            print(f"  sbatch failed for {role}: {result.stderr}")
            continue
        jid = result.stdout.strip()
        jids[role] = jid
        print(f"  {jobname}: {jid}")
    return jids


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup-round-0")
    p_setup.add_argument("--arch", required=True)
    p_setup.add_argument("--d_train", type=int, required=True)
    p_setup.add_argument("--seed", type=int, default=42)

    p_submit = sub.add_parser("submit-round")
    p_submit.add_argument("--arch", required=True)
    p_submit.add_argument("--d_train", type=int, required=True)
    p_submit.add_argument("--round_idx", type=int, required=True)
    p_submit.add_argument("--qos", default="default")

    p_submit_all = sub.add_parser("submit-all-cells")
    p_submit_all.add_argument("--round_idx", type=int, default=0)
    p_submit_all.add_argument("--qos", default="default")

    args = ap.parse_args()

    if args.cmd == "setup-round-0":
        cell_dir = round_0_setup(args.arch, args.d_train, args.seed)
        print(f"Setup round 0 at {cell_dir}")
    elif args.cmd == "submit-round":
        cell_dir = ROOT / f"{args.arch}_d{args.d_train}"
        jids = submit_round(cell_dir, args.arch, args.d_train, args.round_idx, args.qos)
        print(json.dumps(jids))
    elif args.cmd == "submit-all-cells":
        cells = [
            ("legnet", 5000),
            ("legnet", 100000),
            ("dream_rnn", 5000),
            ("dream_rnn", 100000),
            ("dream_attn", 5000),
            ("dream_attn", 100000),
        ]
        all_jids = {}
        for arch, d in cells:
            print(f"\n=== {arch} D={d} round {args.round_idx} ===")
            if args.round_idx == 0:
                round_0_setup(arch, d)
            cell_dir = ROOT / f"{arch}_d{d}"
            jids = submit_round(cell_dir, arch, d, args.round_idx, args.qos)
            all_jids[f"{arch}_d{d}"] = jids
        print("\nAll submitted:")
        print(json.dumps(all_jids, indent=2))


if __name__ == "__main__":
    main()
