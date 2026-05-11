"""Consolidate a cell's AutoResearch round-0 configs into a single configs.json.

Round 0 has 3 roles × 5 configs each = 15 trials per cell. The old workflow
launched 3 SLURM jobs per cell (one per role). The new workflow launches ONE
job per cell that runs all 15 trials via parallel_gpu_runner — much less queue
pressure, matches the shootout pattern.

Usage:
    python -m scripts.preflight.hpsearch.autoresearch_consolidate \\
        --arch legnet --d_train 5000
    # writes results/preflight/hpsearch/autoresearch/{arch}_d{D}/round_0/all_configs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results/preflight/hpsearch/autoresearch"


def consolidate(arch: str, d_train: int, round_idx: int = 0) -> Path:
    cell_dir = ROOT / f"{arch}_d{d_train}"
    out_path = cell_dir / f"round_{round_idx}" / "all_configs.json"
    all_cells: list[dict] = []
    for role in ["A", "B", "C"]:
        runner = cell_dir / f"round_{round_idx}" / f"agent_{role}" / "runner_configs.json"
        if not runner.exists():
            continue
        all_cells.extend(json.loads(runner.read_text()))
    out_path.write_text(json.dumps(all_cells, indent=2))
    print(
        f"  {arch} D={d_train} round={round_idx}: consolidated "
        f"{len(all_cells)} configs → {out_path.relative_to(REPO)}"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--d_train", type=int, required=True)
    ap.add_argument("--round_idx", type=int, default=0)
    args = ap.parse_args()
    consolidate(args.arch, args.d_train, args.round_idx)


if __name__ == "__main__":
    main()
