"""Verify early stopping in completed LegNet D=2000 sweep cells."""

import json
import sys
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
results = list((REPO / "results/preflight/legnet_d2k_highdim").rglob("result.json"))
results += list((REPO / "results/preflight/legnet_d2k_v23").rglob("result.json"))

print(f"Found {len(results)} completed cells. Checking early-stopping behavior:")
print(f"{'cell':<45} {'best_ep':>8} {'final_ep':>8} {'early_stopped':>14}")
print("-" * 80)
for f in results:
    try:
        d = json.loads(f.read_text())
        be = d.get("best_epoch")
        ep = d.get("epochs")
        if be is None or ep is None:
            continue
        es = be < ep - 5  # significantly before final epoch
        label = f.parent.name
        print(f"{label[:43]:<45} {be:>8} {ep:>8} {str(es):>14}")
    except Exception:
        pass
