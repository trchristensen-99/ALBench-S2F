"""Convert subagent-proposed HP configs (agent_proposals_*.json) into a
parallel_gpu_runner cell configs.json that can be sbatch'd.

Inputs:
    --proposals path/to/agent_proposals_legnet_d20k.json
    --arch legnet
    --d_train 20000
    --output_dir results/preflight/hpsearch/agent_legnet_d20k_r1
    --epochs 60
    --patience 10

Output:
    {output_dir}/configs.json  ← for parallel_gpu_runner.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def to_hp_overrides(proposal: dict) -> list[str]:
    """Translate a subagent proposal dict to run_single.py --hp overrides."""
    out = []
    if "lr" in proposal:
        out.append(f"lr={proposal['lr']}")
    if "batch_size" in proposal:
        out.append(f"batch_size={int(proposal['batch_size'])}")
    if "weight_decay" in proposal:
        out.append(f"weight_decay={proposal['weight_decay']}")
    if "dropout" in proposal:
        out.append(f"dropout={proposal['dropout']}")
    if "block_sizes" in proposal:
        # list literal — parse_overrides understands [a,b,c]
        bs = ",".join(str(int(x)) for x in proposal["block_sizes"])
        out.append(f"block_sizes=[{bs}]")
    if "ks" in proposal:
        out.append(f"ks={int(proposal['ks'])}")
    if "optimizer" in proposal:
        out.append(f"optimizer={proposal['optimizer']}")
    if "block_class" in proposal:
        out.append(f"block_class={proposal['block_class']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--d_train", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    proposals = json.loads(Path(args.proposals).read_text())
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfgs = []
    for prop in proposals:
        label = prop.get("label", f"prop_{len(cfgs)}")
        trial_dir = out_root / label
        cfgs.append(
            {
                "label": label,
                "arch": args.arch,
                "d_train": args.d_train,
                "seed": args.seed,
                "epochs": args.epochs,
                "patience": args.patience,
                "aug": prop.get("aug", "rev_complement"),
                "output_dir": str(trial_dir.relative_to(REPO))
                if trial_dir.is_absolute()
                else str(trial_dir),
                "hp_overrides": to_hp_overrides(prop),
            }
        )
    out_path = out_root / "configs.json"
    out_path.write_text(json.dumps(cfgs, indent=2))
    print(f"wrote {len(cfgs)} cells → {out_path}")


if __name__ == "__main__":
    main()
