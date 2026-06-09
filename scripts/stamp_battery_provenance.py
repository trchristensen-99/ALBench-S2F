"""Stamp the chr-split K562 test battery with canonical provenance.

Run this AFTER re-scoring the battery (genomic / snv / ood) with the canonical
AG_S2 oracle. It validates the battery is complete + that the SNV file is the
monoallelic v1 build, then writes <battery_dir>/PROVENANCE.json with the
oracle_id + test_set_version that experiments/test_set_guards.assert_battery_provenance
gates on. Without this stamp the training/eval consumers hard-fail by design.

Usage (on HPC, where the battery lives):
    uv run --no-sync python scripts/stamp_battery_provenance.py
    # or override: --battery-dir ... --oracle-id ... --version ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.test_set_guards import (  # noqa: E402
    BATTERY_VERSION,
    CANONICAL_ORACLE_ID,
    assert_mono_snv,
)

DEFAULT_DIR = REPO / "data" / "k562" / "test_sets_ag_s2_chrsplit"
REQUIRED_FILES = ["genomic_oracle.npz", "snv_oracle.npz", "ood_oracle.npz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--oracle-id", default=CANONICAL_ORACLE_ID)
    ap.add_argument("--version", default=BATTERY_VERSION)
    args = ap.parse_args()

    d = args.battery_dir
    if not d.exists():
        raise SystemExit(f"battery dir not found: {d}")

    missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
    if missing:
        raise SystemExit(f"battery incomplete — missing {missing} in {d}")

    # The SNV file must be the monoallelic v1 build (same guard the consumers use).
    snv_path = d / "snv_oracle.npz"
    assert_mono_snv(np.load(snv_path, allow_pickle=True), snv_path)

    files_meta = {}
    for f in REQUIRED_FILES:
        z = np.load(d / f, allow_pickle=True)
        key = "sequences" if "sequences" in z.files else "ref_sequences"
        files_meta[f] = {"n": int(len(z[key])), "keys": list(z.files)}

    prov = {
        "oracle_id": args.oracle_id,
        "test_set_version": args.version,
        "stamped_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "files": files_meta,
    }
    (d / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {d / 'PROVENANCE.json'}")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
