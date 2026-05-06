"""Dump top retry cells for task3 retry sweeps."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def dump(label: str, base: Path):
    print(f"\n--- {label} ---")
    rows = []
    for rj in base.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
            rows.append(
                {
                    "lr": d["hp"]["lr"],
                    "bs": d["hp"]["batch_size"],
                    "test_mse": d["test_mse_at_best_val"],
                    "best_val": d["best_val_mse"],
                    "best_epoch": d.get("best_epoch", -1),
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"  skip {rj}: {e}")
    rows.sort(key=lambda r: r["test_mse"])
    print(f"  n_results: {len(rows)}")
    print(f"  {'lr':>10}  {'bs':>5}  {'test_mse':>10}  {'best_val':>10}  {'best_ep':>8}")
    for r in rows[:8]:
        print(
            f"  {r['lr']:>10.0e}  {int(r['bs']):>5}  {r['test_mse']:>10.4f}  "
            f"{r['best_val']:>10.4f}  {int(r['best_epoch']):>8}"
        )


def main():
    dump("LegNet retry (aug=none)", REPO / "results/preflight/task3_retry_legnet_noaug")
    dump(
        "dream_attn retry (aug=rc_shift)", REPO / "results/preflight/task3_retry_dream_attn_rcshift"
    )


if __name__ == "__main__":
    main()
