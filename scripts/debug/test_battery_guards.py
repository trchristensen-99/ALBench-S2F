"""Verify the whole-battery + label-cache provenance guards accept canonical stamps
and reject missing / stale ones. Local-only, no GPU or HPC needed."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.test_set_guards import (
    BATTERY_VERSION,
    CANONICAL_ORACLE_ID,
    assert_battery_provenance,
    assert_label_cache_oracle,
)


def _expect_raise(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        print(f"  OK (rejected): {label}")
        return
    raise AssertionError(f"expected RuntimeError but none raised: {label}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        # Battery provenance ----------------------------------------------------
        _expect_raise(lambda: assert_battery_provenance(d), "battery: no PROVENANCE.json")

        (d / "PROVENANCE.json").write_text(
            json.dumps({"oracle_id": "legacy_stage2", "test_set_version": BATTERY_VERSION})
        )
        _expect_raise(lambda: assert_battery_provenance(d), "battery: wrong oracle_id")

        (d / "PROVENANCE.json").write_text(
            json.dumps({"oracle_id": CANONICAL_ORACLE_ID, "test_set_version": BATTERY_VERSION})
        )
        prov = assert_battery_provenance(d)
        assert prov["oracle_id"] == CANONICAL_ORACLE_ID
        print("  OK (accepted): battery canonical stamp")

        # Label-cache oracle ----------------------------------------------------
        unstamped = d / "unstamped.npz"
        np.savez_compressed(unstamped, sequences=np.array(["A"], dtype=object), labels=np.zeros(1))
        _expect_raise(
            lambda: assert_label_cache_oracle(
                np.load(unstamped, allow_pickle=True), unstamped, "ag"
            ),
            "label cache: unstamped",
        )

        stamped = d / "stamped.npz"
        np.savez_compressed(
            stamped,
            sequences=np.array(["A"], dtype=object),
            labels=np.zeros(1),
            oracle_id=np.array("ag"),
        )
        _expect_raise(
            lambda: assert_label_cache_oracle(
                np.load(stamped, allow_pickle=True), stamped, "dream_rnn"
            ),
            "label cache: oracle mismatch",
        )
        assert_label_cache_oracle(np.load(stamped, allow_pickle=True), stamped, "ag")
        print("  OK (accepted): label cache matching oracle_id")

    print("PASS: battery + label-cache provenance guards")


if __name__ == "__main__":
    main()
