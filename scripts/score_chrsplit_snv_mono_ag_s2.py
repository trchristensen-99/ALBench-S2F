"""Score the chr 7+13 strict-mono SNV pairs with the AG_S2 10-fold oracle ensemble.

Produces a drop-in replacement for the legacy hashfrag-derived snv_oracle.npz.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INTERMEDIATE = REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_mono_chr7_13_intermediate.npz"
OUT_NEW = REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_oracle.npz"
BACKUP_OLD = REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_oracle_legacy_hashfrag_2962.npz"


def main():
    from experiments.exp1_1_scaling import _load_oracle

    if not INTERMEDIATE.exists():
        raise FileNotFoundError(
            f"missing {INTERMEDIATE}; run scripts/build_chrsplit_snv_mono.py first"
        )
    z = np.load(INTERMEDIATE, allow_pickle=True)
    ref_seqs = list(z["ref_sequences"])
    alt_seqs = list(z["alt_sequences"])
    pair_keys = z["pair_keys"]
    true_ref_label = z["true_ref_label"].astype(np.float32)
    true_alt_label = z["true_alt_label"].astype(np.float32)
    true_delta = z["true_delta"].astype(np.float32)
    logger.info(f"loaded {len(ref_seqs):,} strict-mono SNV pairs")

    logger.info("loading AG_S2 10-fold ensemble...")
    oracle = _load_oracle("k562", oracle_type="ag_s2")

    logger.info("scoring ref sequences...")
    ref_mean = oracle.predict(ref_seqs).astype(np.float32)
    logger.info("scoring alt sequences...")
    alt_mean = oracle.predict(alt_seqs).astype(np.float32)
    delta_mean = (alt_mean - ref_mean).astype(np.float32)

    # Back up legacy hashfrag-derived snv_oracle.npz (n=2962) if not already done
    if OUT_NEW.exists() and not BACKUP_OLD.exists():
        OUT_NEW.rename(BACKUP_OLD)
        logger.info(f"backed up old snv_oracle.npz → {BACKUP_OLD.name}")

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
    )
    logger.info(f"saved {OUT_NEW}  (n={len(ref_seqs):,})")


if __name__ == "__main__":
    main()
