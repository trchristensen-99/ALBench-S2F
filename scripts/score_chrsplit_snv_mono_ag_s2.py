"""Score the chr 7+13 strict-mono SNV pairs with the canonical AG_S2 oracle ensemble.

Produces a drop-in replacement for the legacy/over-sized snv_oracle.npz, stamped with
provenance metadata so downstream loaders can assert they are using the proper
monoallelic (~29.4k) set scored by the canonical oracle.

Oracle selection: the chr-split natural ensemble (outputs/oracle_chrsplit_natural/s2)
does not currently exist, and the legacy stage2_k562_oracle fallback is also gone, so
the default _load_oracle path would crash. We therefore pin the canonical oracle
explicitly via AG_S2_ORACLE_DIR (default: outputs/oracle_full856k_clean/s2 — the same
"designed-included, canonical" oracle used for the rest of the chr-split battery).

Before writing, we re-score a sample of the existing genomic_oracle.npz with the same
oracle and assert it matches, guaranteeing the new SNV labels are consistent with the
already-materialized genomic/ood test labels.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TEST_DIR = REPO / "data/k562/test_sets_ag_s2_chrsplit"
INTERMEDIATE = TEST_DIR / "snv_mono_chr7_13_intermediate.npz"
OUT_NEW = TEST_DIR / "snv_oracle.npz"
BACKUP_OVERSIZED = TEST_DIR / "snv_oracle_legacy_oversized_45543.npz"
GENOMIC = TEST_DIR / "genomic_oracle.npz"

CANONICAL_ORACLE = REPO / "outputs/oracle_full856k_clean/s2"
VERIFY_SAMPLE = 256
VERIFY_MIN_CORR = 0.999

# Expected monoallelic count from build_chrsplit_snv_mono.py (chr 7+13 strict-mono).
EXPECTED_N_MIN = 25_000
EXPECTED_N_MAX = 35_000


def _verify_canonical(oracle) -> None:
    """Assert the loaded oracle reproduces the existing genomic battery labels."""
    if not GENOMIC.exists():
        raise FileNotFoundError(
            f"missing {GENOMIC}; cannot verify SNV oracle consistency with the battery"
        )
    z = np.load(GENOMIC, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"][:VERIFY_SAMPLE]]
    stored = z["oracle_mean"][:VERIFY_SAMPLE].astype(np.float64)
    pred = np.asarray(oracle.predict(seqs), dtype=np.float64)
    corr = float(np.corrcoef(stored, pred)[0, 1])
    mae = float(np.mean(np.abs(stored - pred)))
    logger.info(f"consistency check vs genomic battery: corr={corr:.6f}  MAE={mae:.4f}")
    if corr < VERIFY_MIN_CORR:
        raise RuntimeError(
            f"oracle MISMATCH: re-scoring genomic_oracle.npz with {os.environ.get('AG_S2_ORACLE_DIR')} "
            f"gives corr={corr:.5f} (< {VERIFY_MIN_CORR}). The existing battery was scored by a "
            f"DIFFERENT oracle — scoring SNV with this one would make the SNV-Δ panel inconsistent. "
            f"Re-score the whole battery with one oracle, or point AG_S2_ORACLE_DIR at the right one."
        )


def main():
    os.environ.setdefault("AG_S2_ORACLE_DIR", str(CANONICAL_ORACLE))
    logger.info("AG_S2_ORACLE_DIR = %s", os.environ["AG_S2_ORACLE_DIR"])

    from experiments.exp1_1_scaling import _load_oracle

    if not INTERMEDIATE.exists():
        raise FileNotFoundError(
            f"missing {INTERMEDIATE}; run scripts/build_chrsplit_snv_mono.py first"
        )
    z = np.load(INTERMEDIATE, allow_pickle=True)
    ref_seqs = [str(s) for s in z["ref_sequences"]]
    alt_seqs = [str(s) for s in z["alt_sequences"]]
    pair_keys = z["pair_keys"]
    true_ref_label = z["true_ref_label"].astype(np.float32)
    true_alt_label = z["true_alt_label"].astype(np.float32)
    true_delta = z["true_delta"].astype(np.float32)
    n = len(ref_seqs)
    logger.info(f"loaded {n:,} strict-mono SNV pairs")
    if not (EXPECTED_N_MIN <= n <= EXPECTED_N_MAX):
        raise RuntimeError(
            f"mono SNV count {n:,} outside expected [{EXPECTED_N_MIN:,},{EXPECTED_N_MAX:,}] — "
            f"rebuild via scripts/build_chrsplit_snv_mono.py before scoring"
        )

    logger.info("loading AG_S2 ensemble (canonical)...")
    oracle = _load_oracle("k562", oracle_type="ag_s2")

    logger.info("verifying oracle reproduces the existing genomic battery...")
    _verify_canonical(oracle)

    logger.info("scoring ref sequences...")
    ref_mean = np.asarray(oracle.predict(ref_seqs), dtype=np.float32)
    logger.info("scoring alt sequences...")
    alt_mean = np.asarray(oracle.predict(alt_seqs), dtype=np.float32)
    delta_mean = (alt_mean - ref_mean).astype(np.float32)

    # Back up the over-sized (45,543) snv_oracle.npz once.
    if OUT_NEW.exists() and not BACKUP_OVERSIZED.exists():
        OUT_NEW.rename(BACKUP_OVERSIZED)
        logger.info(f"backed up over-sized snv_oracle.npz → {BACKUP_OVERSIZED.name}")

    np.savez_compressed(
        OUT_NEW,
        pair_keys=pair_keys,
        ref_sequences=np.array(ref_seqs, dtype=object),
        alt_sequences=np.array(alt_seqs, dtype=object),
        ref_mean=ref_mean,
        alt_mean=alt_mean,
        delta_mean=delta_mean,
        true_ref_label=true_ref_label,
        true_alt_label=true_alt_label,
        true_delta=true_delta,
        # provenance — asserted at load time (see scaling_hp_search.load_all_test_sets)
        test_set_version=np.str_("snv_mono_chrsplit_v1"),
        monoallelic=np.bool_(True),
        n_pairs=np.int64(n),
        oracle_dir=np.str_(os.environ["AG_S2_ORACLE_DIR"]),
        source_tsv=np.str_("data/k562/test_sets/test_snv_pairs.tsv"),
        created_utc=np.str_(datetime.now(timezone.utc).isoformat()),
    )
    logger.info(f"saved {OUT_NEW}  (n={n:,}, version=snv_mono_chrsplit_v1)")


if __name__ == "__main__":
    main()
