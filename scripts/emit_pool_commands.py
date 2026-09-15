"""Emit the pool build commands for the screen."""

import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albench.run import ScreenConfig, expand_screen
from albench.pools import pool_plan, describe_saving

cfg = ScreenConfig.from_dict(yaml.safe_load(open("configs/screen/human_30k_300k.yaml")))
cells = expand_screen(cfg, "screen")
pools = pool_plan(cells, pool_size=300_000, n_generation_seeds=1)
print(describe_saving(cells, pools, seq_per_s=300.0))
print()
lines = []
for p in pools:
    sets = " ".join(f"--set {k}={v}" for k, v in sorted(p.params.items()))
    lines.append(
        f"python scripts/build_pool.py --strategy {p.strategy} --size {p.size} "
        f"--generation-seed {p.generation_seed} {sets} --out-dir outputs/pools".replace("  ", " ")
    )
Path("outputs/pools/build_commands.sh").parent.mkdir(parents=True, exist_ok=True)
Path("outputs/pools/build_commands.sh").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} pool build commands -> outputs/pools/build_commands.sh")
for ln in lines[:3]:
    print("  " + ln)
