"""Provenance guards for the chr-split K562 test sets.

Single source of truth so every consumer refuses silently-wrong test sets. The proper
SNV test set is the strict-monoallelic chr 7+13 build (~29.4k pairs) scored by the
canonical AG_S2 oracle, stamped test_set_version='snv_mono_chrsplit_v1'. Historically
two wrong SNV files existed: the legacy hashfrag set (n=2962) and an over-sized
unfiltered set (n=45,543). Both lack the version stamp, so assert_mono_snv rejects them.

The whole battery (genomic / snv / ood) is gated by a directory-level PROVENANCE.json
written by the re-scoring pipeline (scripts/reeval_chrsplit_ag_s2.py + stamping). It
records which oracle produced the labels (oracle_id) and which battery build it is
(test_set_version). assert_battery_provenance hard-fails unless those match the
canonical values below, so a stale or differently-scored battery can never be trained
or evaluated against by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

SNV_MONO_VERSION = "snv_mono_chrsplit_v1"

# Canonical battery provenance. CANONICAL_ORACLE_ID is the 10-fold CV AG_S2 ensemble
# trained on a random 90/10 split of the full 856,290-row pool (outputs/oracle_full856k_clean/s2).
# BATTERY_VERSION is bumped whenever the battery sequences/labels are rebuilt.
CANONICAL_ORACLE_ID = "full856k_clean"
BATTERY_VERSION = "ag_s2_chrsplit_v1"

_UNSTAMPED = {"oracle_id": "unstamped", "test_set_version": "unstamped"}


def assert_mono_snv(z, path) -> None:
    """Raise unless ``z`` (an np.load of snv_oracle.npz) is the monoallelic v1 set."""
    files = getattr(z, "files", None)
    ver = str(z["test_set_version"]) if (files and "test_set_version" in files) else None
    if ver != SNV_MONO_VERSION:
        n = len(z["ref_sequences"]) if (files and "ref_sequences" in files) else "?"
        raise RuntimeError(
            f"{path} is not the monoallelic chr-split SNV set "
            f"(version={ver!r}, n={n}). Rebuild via scripts/build_chrsplit_snv_mono.py + "
            f"scripts/score_chrsplit_snv_mono_ag_s2.py."
        )


def read_battery_provenance(battery_dir) -> dict:
    """Tolerant read of ``<battery_dir>/PROVENANCE.json``.

    Returns {'oracle_id', 'test_set_version'}, defaulting both to 'unstamped' when the
    file is absent or unreadable. Used for *recording* provenance (e.g. into an HP-run
    regime stamp); use assert_battery_provenance when you need to *gate* on it.
    """
    prov_path = Path(battery_dir) / "PROVENANCE.json"
    if prov_path.exists():
        try:
            p = json.loads(prov_path.read_text())
            return {
                "oracle_id": str(p.get("oracle_id", "unstamped")),
                "test_set_version": str(p.get("test_set_version", "unstamped")),
            }
        except Exception:
            pass
    return dict(_UNSTAMPED)


def assert_battery_provenance(
    battery_dir,
    *,
    require_oracle: str = CANONICAL_ORACLE_ID,
    require_version: str = BATTERY_VERSION,
) -> dict:
    """Hard-fail unless the battery at ``battery_dir`` carries the canonical stamps.

    Mirrors assert_mono_snv's contract for the whole battery: a missing, unstamped, or
    mismatched PROVENANCE.json halts the run rather than letting a stale / differently
    scored test set through. Returns the validated provenance dict on success.
    """
    prov = read_battery_provenance(battery_dir)
    ok = prov["oracle_id"] == require_oracle and prov["test_set_version"] == require_version
    if not ok:
        raise RuntimeError(
            f"Battery at {battery_dir} has provenance {prov} but the canonical battery "
            f"requires oracle_id={require_oracle!r}, test_set_version={require_version!r}. "
            "Re-score + stamp it via scripts/reeval_chrsplit_ag_s2.py against the canonical "
            "AG_S2 oracle (outputs/oracle_full856k_clean/s2), which writes PROVENANCE.json."
        )
    return prov


def assert_label_cache_oracle(z, path, expected_oracle_id: str) -> None:
    """Raise unless a cached ``oracle_labels.npz`` was produced by ``expected_oracle_id``.

    Prevents silently reusing a label cache that a *different* oracle wrote (e.g. after
    switching --oracle or pointing at a new oracle ensemble). An unstamped cache is
    treated as wrong so legacy caches are regenerated rather than trusted.
    """
    files = getattr(z, "files", None)
    got = str(z["oracle_id"]) if (files and "oracle_id" in files) else None
    if got != expected_oracle_id:
        raise RuntimeError(
            f"{path} was produced by oracle_id={got!r} but this run uses "
            f"oracle_id={expected_oracle_id!r}. Delete the stale cache so it is "
            "regenerated with the current oracle."
        )
