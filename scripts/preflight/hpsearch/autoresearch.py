"""AutoResearch: LLM-driven HP proposer using 3 parallel subagents.

This is the SCAFFOLD that the main Claude process uses to delegate HP
proposals to subagents. Each subagent runs autonomously inside a single
Agent() tool call, proposing 5 configs per round across 4 rounds.

Agents (parallel, per (arch, D) cell):
    A: explorative — try unusual / diverse combos
    B: exploitative — refine around top-K observed configs
    C: ablation — vary one HP at a time around current best

The driver here only writes the *plan files*. The actual subagent
invocations are done by the main Claude process via Agent() tool calls.

What lives here:
    - The shared HP space spec used to brief subagents (deterministic JSON)
    - A run_proposed_configs() function that takes a JSON of configs and
      runs them via parallel_gpu_runner on the GPU
    - A digest_results() function that reads result.json files and writes a
      "current state" summary the next round's subagent will read

Each subagent reads the current_state.json (HPs tried so far + scores), and
writes proposed_configs.json. The driver then runs these and updates
current_state.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def make_briefing(arch: str, d_train: int, agent_role: str) -> str:
    """Return the prompt that the calling Agent() will send to the subagent.

    The briefing is fully self-contained (subagent has no prior context).
    """
    from scripts.preflight.hpsearch.hp_space import get_full_space

    space = get_full_space(arch)
    space_str = json.dumps(
        {k: list(v) if isinstance(v[1], list) else v for k, v in space.items()},
        indent=2,
        default=str,
    )

    role_text = {
        "A": (
            "Role: EXPLORATIVE. Your goal is to propose DIVERSE configurations that "
            "explore corners of the HP space the prior trials haven't touched. "
            "Favor configurations whose (width, depth, lr) triplets are far from "
            "all previously-tried configurations. Aim for chemical diversity, not "
            "performance — leave exploitation to Agent B."
        ),
        "B": (
            "Role: EXPLOITATIVE. Refine around the TOP-3 prior trials (lowest val_loss). "
            "Propose 5 perturbations: small lr/bs/wd/dropout tweaks around each top config. "
            "Stay in the same (width, depth) regime as the top configs unless one "
            "is clearly underperforming."
        ),
        "C": (
            "Role: ABLATION. Take the SINGLE BEST prior trial as anchor. Propose 5 "
            "configs that each vary EXACTLY ONE HP from the anchor (cycle through "
            "lr, batch_size, weight_decay, dropout, width). This isolates which "
            "axis the search is most sensitive to."
        ),
    }[agent_role]

    return f"""You are an HP-optimization subagent for ALBench-S2F.

{role_text}

# Task
Propose 5 HP configurations for arch={arch}, d_train={d_train}. Write them
to `proposed_configs.json` at the path you'll be given. Each config must be
a dict with these keys (no extras):
  - lr (float, log-scale)
  - batch_size (int)
  - weight_decay (float, log-scale)
  - dropout (float)
  - width (int)
  - depth (int)

# Search space (constraints — values outside are rejected)
{space_str}

# Read the current state
Read `current_state.json` at the working directory. It contains:
  - prior_trials: list of {{config, val_loss, test_loss}}, sorted by val_loss ascending
  - round_idx: 0-indexed round number (0..3)

# Output
Write `proposed_configs.json` as a JSON list of exactly 5 dicts. Each dict has
the 6 HP keys above. Do not write anything else.

You have ONE goal: write that one file. Then stop. Do not run any training.
"""


def digest_results(
    cell_dir: Path,
    round_idx: int,
) -> dict[str, Any]:
    """Aggregate all completed trials in cell_dir into a current_state snapshot.

    Reads cell_dir/round_*/configs.json and each trial's output_dir/result.json.
    Sorts by val_loss ascending. Writes cell_dir/current_state.json.
    """
    prior_trials: list[dict[str, Any]] = []
    for round_dir in sorted(cell_dir.glob("round_*")):
        cfg_file = round_dir / "configs.json"
        if not cfg_file.exists():
            continue
        configs = json.loads(cfg_file.read_text())
        for cfg in configs:
            out = cell_dir / cfg["output_dir"]
            if not out.is_absolute():
                out = REPO / cfg["output_dir"]
            result_file = out / "result.json"
            if result_file.exists():
                summary = json.loads(result_file.read_text())
                # Extract abstract HPs back from the run config
                hp = summary.get("hp", {})
                trial_record = {
                    "config": {
                        "lr": hp.get("lr"),
                        "batch_size": hp.get("batch_size"),
                        "weight_decay": hp.get("weight_decay"),
                        "dropout": (
                            hp.get("dropout")
                            or hp.get("dropout_cnn")
                            or hp.get("first_block_dropout")
                            or 0.0
                        ),
                        # width/depth need to be reconstructed from arch HPs
                        "width": (
                            (hp.get("block_sizes") or [None])[-1]
                            or hp.get("hidden_dim")
                            or hp.get("embedding_dim")
                        ),
                        "depth": (
                            len(hp.get("block_sizes") or [])
                            or hp.get("num_lstm_layers")
                            or hp.get("num_blocks")
                        ),
                    },
                    "val_loss": summary.get("best_val_mse"),
                    "test_loss": summary.get("test_mse_at_best_val"),
                }
                prior_trials.append(trial_record)
    prior_trials.sort(key=lambda r: r["val_loss"] if r["val_loss"] is not None else float("inf"))

    state = {
        "round_idx": round_idx,
        "n_completed": len(prior_trials),
        "prior_trials": prior_trials,
    }
    (cell_dir / "current_state.json").write_text(json.dumps(state, indent=2))
    return state


def configs_for_parallel_runner(
    arch: str,
    d_train: int,
    seed: int,
    proposed: list[dict[str, Any]],
    round_idx: int,
    agent_role: str,
    cell_dir: Path,
) -> list[dict[str, Any]]:
    """Convert a list of abstract HP dicts into parallel_gpu_runner cell configs."""
    from scripts.preflight.hpsearch.hp_space import to_run_single_overrides

    cells = []
    for i, hp in enumerate(proposed):
        label = f"ar_{agent_role}_r{round_idx}_t{i}"
        out_dir = cell_dir / f"round_{round_idx}_agent_{agent_role}" / label
        overrides = to_run_single_overrides(arch, hp)
        cells.append(
            {
                "label": label,
                "arch": arch,
                "d_train": d_train,
                "seed": seed,
                "epochs": 60,
                "patience": 15,
                "aug": "rev_complement",
                "output_dir": str(out_dir.relative_to(REPO)),
                "hp_overrides": overrides,
            }
        )
    return cells


def main():
    """CLI utility for use INSIDE a subagent or by the orchestrator.

    Subcommands:
      brief        — print the briefing prompt for a (arch, d, role) (for inspection)
      digest       — aggregate completed trials → current_state.json
      to-runner    — convert proposed_configs.json + cell metadata → parallel_gpu_runner configs.json
    """
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_brief = sub.add_parser("brief")
    p_brief.add_argument("--arch", required=True)
    p_brief.add_argument("--d_train", type=int, required=True)
    p_brief.add_argument("--role", choices=["A", "B", "C"], required=True)

    p_digest = sub.add_parser("digest")
    p_digest.add_argument("--cell_dir", required=True)
    p_digest.add_argument("--round_idx", type=int, required=True)

    p_to_run = sub.add_parser("to-runner")
    p_to_run.add_argument("--cell_dir", required=True)
    p_to_run.add_argument("--proposed", required=True)
    p_to_run.add_argument("--arch", required=True)
    p_to_run.add_argument("--d_train", type=int, required=True)
    p_to_run.add_argument("--seed", type=int, default=42)
    p_to_run.add_argument("--round_idx", type=int, required=True)
    p_to_run.add_argument("--role", choices=["A", "B", "C"], required=True)
    p_to_run.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "brief":
        print(make_briefing(args.arch, args.d_train, args.role))
    elif args.cmd == "digest":
        state = digest_results(Path(args.cell_dir), args.round_idx)
        print(json.dumps(state, indent=2))
    elif args.cmd == "to-runner":
        proposed = json.loads(Path(args.proposed).read_text())
        cell_dir = Path(args.cell_dir)
        cells = configs_for_parallel_runner(
            args.arch,
            args.d_train,
            args.seed,
            proposed,
            args.round_idx,
            args.role,
            cell_dir,
        )
        Path(args.out).write_text(json.dumps(cells, indent=2))
        print(f"wrote {len(cells)} cells to {args.out}")


if __name__ == "__main__":
    main()
