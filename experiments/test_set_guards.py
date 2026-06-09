"""Provenance guards for the chr-split K562 test sets.

Single source of truth so every consumer refuses silently-wrong test sets. The proper
SNV test set is the strict-monoallelic chr 7+13 build (~29.4k pairs) scored by the
canonical AG_S2 oracle, stamped test_set_version='snv_mono_chrsplit_v1'. Historically
two wrong SNV files existed: the legacy hashfrag set (n=2962) and an over-sized
unfiltered set (n=45,543). Both lack the version stamp, so assert_mono_snv rejects them.
"""

from __future__ import annotations

SNV_MONO_VERSION = "snv_mono_chrsplit_v1"


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
