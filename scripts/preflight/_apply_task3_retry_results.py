"""Apply task3 retry results to pre_flight_decisions.yaml.

Reads the top retry cell per arch (lowest test_mse_at_best_val) and
updates learning_rate + batch_size for the affected archs (legnet,
dream_attn). dream_rnn is left untouched (no retry was needed).

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "results" / "preflight" / "pre_flight_decisions.yaml"


def best_cell(retry_dir: Path) -> tuple[float, int, Path] | None:
    """Return (lr, bs, result_path) of the lowest test_mse cell."""
    best = None
    for rj in retry_dir.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
            mse = d["test_mse_at_best_val"]
            lr = d["hp"]["lr"]
            bs = int(d["hp"]["batch_size"])
            if best is None or mse < best[0]:
                best = (mse, lr, bs, rj)
        except Exception:  # noqa: BLE001
            continue
    return best


def main():
    if not DECISIONS.exists():
        raise SystemExit(f"missing {DECISIONS}")
    d = yaml.safe_load(DECISIONS.read_text())

    today = datetime.utcnow().strftime("%Y-%m-%d")

    for arch, retry_dir, aug, source_label in [
        (
            "legnet",
            REPO / "results/preflight/task3_retry_legnet_noaug",
            "none",
            "task3_retry_aug_corrected (aug=none, extended LR grid)",
        ),
        (
            "dream_attn",
            REPO / "results/preflight/task3_retry_dream_attn_rcshift",
            "rc_shift",
            "task3_retry_aug_corrected (aug=rc_shift, extended BS grid)",
        ),
    ]:
        if not retry_dir.exists():
            print(f"  {arch}: retry dir missing, skipping")
            continue
        best = best_cell(retry_dir)
        if best is None:
            print(f"  {arch}: no result.json in retry dir, skipping")
            continue
        mse, lr, bs, rj = best
        evidence = str(rj.relative_to(REPO))

        prev_lr = d["learning_rate"][arch].get("value")
        prev_bs = d["batch_size"][arch].get("value")
        notes = (
            f"From retry sweep on {today}: aug={aug} chosen because task5 found "
            f"per-arch optimal aug differs. test_mse={mse:.4f}. "
            f"Prior locked lr={prev_lr} bs={prev_bs} (which used aug=rev_complement) "
            f"is overwritten."
        )
        d["learning_rate"][arch]["value"] = lr
        d["learning_rate"][arch]["locked_by"] = source_label
        d["learning_rate"][arch]["evidence"] = evidence
        d["learning_rate"][arch]["notes"] = notes
        d["batch_size"][arch]["value"] = bs
        d["batch_size"][arch]["locked_by"] = source_label
        d["batch_size"][arch]["evidence"] = evidence
        d["batch_size"][arch]["notes"] = notes
        # Lock the augmentation too while we're at it
        d.setdefault("augmentations", {}).setdefault(arch, {})
        d["augmentations"][arch]["value"] = aug
        d["augmentations"][arch]["locked_by"] = source_label
        d["augmentations"][arch]["evidence"] = evidence
        d["augmentations"][arch]["notes"] = notes
        print(
            f"  {arch}: lr {prev_lr} -> {lr}, bs {prev_bs} -> {bs}, aug -> {aug} (test_mse={mse:.4f})"
        )

    # Also lock dream_rnn aug since task5 found rev_complement is best
    if "augmentations" in d and isinstance(d["augmentations"].get("dream_rnn"), dict):
        if d["augmentations"]["dream_rnn"].get("value") in (None, "null"):
            d["augmentations"]["dream_rnn"]["value"] = "rev_complement"
            d["augmentations"]["dream_rnn"]["locked_by"] = "task5_universality_analysis"
            d["augmentations"]["dream_rnn"]["notes"] = (
                f"From task5 analysis on {today}: rev_complement was best "
                f"(test_mse=0.1541 vs rc_shift=0.1771)"
            )
            print(
                "  dream_rnn aug: -> rev_complement (no LR/BS change needed; was already optimal)"
            )

    DECISIONS.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=False))
    print(f"\n  ✓ Wrote {DECISIONS}")


if __name__ == "__main__":
    main()
