"""Score already-trained cells on eval sets that did not exist when they were trained.

The eval battery is baked into each member's npz at training time, so a new eval set would
normally mean retraining everything. The checkpoints are on disk, so instead this reloads
them and runs the new sets through -- the same RC-averaged, flank-wrapped input the trainer
used, because `LegNetBank` imports the trainer's own helpers rather than reimplementing them.

Predictions are stored PER MEMBER and keyed by member id, not averaged: the figures stack
members with the validation-fitted ElasticNet, and an average here would quietly replace
that stack with a uniform one on these endpoints only.

    python scratchpad/nov25/rescore_extra_sets.py --sets motif_grammar motif_planted
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad/nov25"))

# Several directories are searched for an eval set, and the sidecar may be written somewhere
# other than the cell. That is not generality for its own sake: the filesystem holding the
# trained cells ran out of INODES on 2026-11-25 (52,428,800 of 52,428,800 used), so a script
# that could only write next to the cell it scores could not run at all.
BATTERIES = [
    Path(p)
    for p in (
        os.environ.get("ALBENCH_EXTRA_BATTERY", ""),
        str(REPO / "data/k562/test_sets_ag_s2_chrsplit"),
    )
    if p
]
SIDECAR = "extra_evalsets_v1.npz"


def load_set(name: str) -> tuple[list[str], np.ndarray]:
    for b in BATTERIES:
        p = b / f"{name}_oracle.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            key = "oracle_mean" if "oracle_mean" in z.files else "oracle_labels"
            return [str(x) for x in z["sequences"]], np.asarray(z[key], np.float32).ravel()
    raise FileNotFoundError(f"{name} in none of {[str(b) for b in BATTERIES]}")


def cells(max_members: int) -> list[Path]:
    """Every tuned per-size cell the scaling figures read, plus the acquisition branches."""
    from scaling_mse import RESERVOIRS, TUNED, cell_dir

    out = []
    for _res, suffix in RESERVOIRS:
        for D in sorted(TUNED):
            d = cell_dir(suffix, D)
            if d is not None and len(glob.glob(str(d / "*.pt"))) >= 1:
                out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["motif_grammar", "motif_planted"])
    ap.add_argument("--max-members", type=int, default=3)
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--sidecar-root",
        default=os.environ.get("ALBENCH_SIDECAR_ROOT", ""),
        help="write <root>/<cell name>.npz instead of a file inside the cell directory",
    )
    a = ap.parse_args()
    root = Path(a.sidecar_root) if a.sidecar_root else None
    if root:
        root.mkdir(parents=True, exist_ok=True)

    data = {}
    for s in a.sets:
        try:
            seqs, y = load_set(s)
            data[s] = (seqs, y)
            print(f"{s}: {len(seqs):,} sequences, oracle mean {y.mean():.3f}", flush=True)
        except FileNotFoundError:
            print(f"{s}: not built yet, skipping", flush=True)
    if not data:
        return 1

    targets = [Path(d) for d in a.dirs] if a.dirs else cells(a.max_members)
    print(f"{len(targets)} cells to score\n", flush=True)

    from albench.acquisition.legnet_bank import LegNetBank

    for d in targets:
        pts = sorted(glob.glob(str(d / "*.pt")))[: a.max_members]
        if not pts:
            print(f"SKIP {d.name}: no checkpoints")
            continue
        side = (root / f"{d.name}.npz") if root else (d / SIDECAR)
        have = {}
        if side.exists() and not a.force:
            z = np.load(side, allow_pickle=True)
            have = {k: z[k] for k in z.files}
            mids = [str(x) for x in have.get("members", [])]
            if mids == [Path(p).stem for p in pts] and all(f"pred_{s}" in have for s in data):
                print(f"skip {d.name}: already scored")
                continue

        # A bank restricted to exactly the members the figures stack, in the same order the
        # `*_meta.json` glob produces, so column j here is member j there.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for p in pts:
                os.symlink(p, os.path.join(tmp, os.path.basename(p)))
            bank = LegNetBank(tmp)
            out = {"members": np.asarray([Path(p).stem for p in pts], dtype=object)}
            for s, (seqs, y) in data.items():
                out[f"pred_{s}"] = bank._member_predictions(seqs).astype(np.float32)
                out[f"oracle_{s}"] = y
        tmpf = str(side) + f".tmp{os.getpid()}.npz"
        np.savez(tmpf, **out)
        os.replace(tmpf, side)
        r = [
            f"{s} r={np.corrcoef(out[f'pred_{s}'].mean(0), out[f'oracle_{s}'])[0, 1]:.3f}"
            for s in data
        ]
        print(f"{d.name:<52} {len(pts)} members  " + "  ".join(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
