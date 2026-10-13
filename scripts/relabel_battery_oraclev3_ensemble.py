#!/usr/bin/env python
"""Relabel the chr-split eval battery UNIFORMLY with the canonical oracle_v3 ensemble.

The battery was oracle-MIXED: the Jun-9 core sets were stamped ``full856k_clean`` (the
superseded RANDOM 10-fold oracle) while the Sep-30 matched sets were already
``oracle_v3``. Here we re-score EVERY set's oracle labels with the oracle_v3 10-fold
ensemble (RC-averaged) -- the exact loader (`_load_oracle("k562","ag_s2")`) that
labelled the grid_pools training caches and the matched sets -- so the whole battery is
one oracle.

Ensemble-mean is used for every set (genomic, snv, ood included) on purpose: the oracle
is the ground-truth LABEL GENERATOR, not an evaluated model, so test-chromosome
"leakage" into the oracle is irrelevant and we want it maximally accurate.

Sequences and ``true_label`` are left byte-identical; only the oracle label arrays,
``oracle_id`` and the dir-level PROVENANCE change. Each original file is copied to
``<file>.premix`` first (non-destructive, recoverable).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.test_set_guards import CANONICAL_ORACLE_ID  # noqa: E402  -> "oracle_v3"

BATTERY = REPO / "data" / "k562" / "test_sets_ag_s2_chrsplit"
NEW_BATTERY_VERSION = "ag_s2_chrsplit_v2"  # same builds, uniformly oracle_v3-scored

# Standard single-sequence sets: key "sequences" -> relabel "oracle_mean".
STD_SETS = [
    "ctrl_neg", "del_high", "del_low", "del_med", "dinuc_shuffle", "gc_matched",
    "gc_sweep", "genomic", "ins_high", "ins_low", "ins_med", "inversion",
    "motif_planted", "ood", "random_10k", "random_32k", "sub_high", "sub_low",
    "sub_med", "translocation", "zoonomia",
]
# Sets already known to be oracle_v3 (Sep-30 matched) -> expect corr_with_old ~= 1.0.
ALREADY_V3 = {"gc_matched", "motif_planted", "zoonomia"}


def _corr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _backup(f: Path):
    bak = f.with_name(f.name + ".premix")
    if not bak.exists():
        shutil.copy2(f, bak)
        print(f"    backup -> {bak.name}")
    else:
        print(f"    backup exists {bak.name} (kept)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    print(f"[relabel] oracle_v3 ensemble relabel @ {now}")
    print(f"[relabel] battery dir: {BATTERY}")

    oracle = None
    if not args.dry_run:
        from experiments.exp1_1_scaling import _load_oracle
        print("[relabel] loading oracle ensemble via _load_oracle('k562','ag_s2') ...")
        oracle = _load_oracle("k562", oracle_type="ag_s2")
        print("[relabel] oracle loaded.")

    prov_files: dict[str, dict] = {}
    anchors: dict[str, float] = {}
    all_finite = True

    # ---- standard single-sequence sets ----------------------------------------
    for name in STD_SETS:
        f = BATTERY / f"{name}_oracle.npz"
        if not f.exists():
            print(f"  SKIP missing {f.name}")
            continue
        z = np.load(f, allow_pickle=True)
        seqs = [str(s) for s in z["sequences"]]
        old = np.asarray(z["oracle_mean"], np.float64) if "oracle_mean" in z.files else None
        print(f"  {f.name}: n={len(seqs)}")
        if args.dry_run:
            continue
        preds = oracle.predict(seqs).astype(np.float32)
        nfin = int(np.isfinite(preds).sum())
        if nfin != preds.size:
            all_finite = False
        cwo = _corr(preds, old) if old is not None else float("nan")
        if name in ALREADY_V3:
            anchors[name] = cwo
        print(f"    relabeled: finite={nfin}/{preds.size} mean={np.nanmean(preds):.4f} "
              f"std={np.nanstd(preds):.4f} corr_with_old={cwo:.4f}")
        _backup(f)
        out = {k: np.array(z[k]) for k in z.files}
        out["oracle_mean"] = preds
        out["oracle_id"] = np.str_(CANONICAL_ORACLE_ID)
        if "test_set_version" not in z.files:
            out["test_set_version"] = np.str_(NEW_BATTERY_VERSION)
        out["relabeled_utc"] = np.str_(now)
        out["relabel_note"] = np.str_("oracle_v3 10-fold ensemble, RC-avg; builds unchanged")
        np.savez_compressed(f, **out)
        prov_files[f.name] = {
            "n": len(seqs),
            "oracle_id": CANONICAL_ORACLE_ID,
            "test_set_version": str(out["test_set_version"]),
        }

    # ---- SNV (ref/alt/delta) --------------------------------------------------
    f = BATTERY / "snv_oracle.npz"
    if f.exists() and not args.dry_run:
        z = np.load(f, allow_pickle=True)
        ref = [str(s) for s in z["ref_sequences"]]
        alt = [str(s) for s in z["alt_sequences"]]
        print(f"  {f.name}: n_pairs={len(ref)}")
        ref_m = oracle.predict(ref).astype(np.float32)
        alt_m = oracle.predict(alt).astype(np.float32)
        new_delta = (alt_m - ref_m).astype(np.float64)
        # Preserve the existing delta sign convention.
        conv = "alt-ref"
        if "delta_mean" in z.files:
            c = _corr(np.asarray(z["delta_mean"]), new_delta)
            if c < 0:
                new_delta = -new_delta
                conv = "ref-alt"
            print(f"    delta convention={conv} (corr vs old delta={c:.4f})")
        if "true_delta" in z.files:
            print(f"    corr(new_delta, true_delta)={_corr(np.asarray(z['true_delta']), new_delta):.4f}")
        for nm, a in (("ref_mean", ref_m), ("alt_mean", alt_m), ("delta_mean", new_delta)):
            nf = int(np.isfinite(a).sum())
            if nf != a.size:
                all_finite = False
            print(f"    {nm}: finite={nf}/{a.size} mean={np.nanmean(a):.4f}")
        _backup(f)
        out = {k: np.array(z[k]) for k in z.files}
        out["ref_mean"] = ref_m
        out["alt_mean"] = alt_m
        out["delta_mean"] = new_delta.astype(np.float32)
        out["oracle_id"] = np.str_(CANONICAL_ORACLE_ID)  # KEEP test_set_version=snv_mono_chrsplit_v1
        out["relabeled_utc"] = np.str_(now)
        np.savez_compressed(f, **out)
        prov_files[f.name] = {
            "n": len(ref),
            "oracle_id": CANONICAL_ORACLE_ID,
            "test_set_version": str(out["test_set_version"]) if "test_set_version" in out else "-",
        }

    if args.dry_run:
        print("[relabel] dry-run: no files written.")
        return

    # ---- dir-level PROVENANCE -------------------------------------------------
    prov = {
        "oracle_id": CANONICAL_ORACLE_ID,
        "test_set_version": NEW_BATTERY_VERSION,
        "oracle_dir": "outputs/oracle_v3/s2",
        "labeling": "oracle_v3 10-fold ensemble mean, RC-averaged (_load_oracle k562 ag_s2)",
        "stamped_at": now,
        "note": (
            "Uniform single-oracle battery. Supersedes the mixed ag_s2_chrsplit_v1 "
            "(core=full856k_clean + matched=oracle_v3). Sequences/true_label unchanged; "
            "only oracle labels re-scored with oracle_v3 ensemble-mean. Originals kept as *.premix."
        ),
        "files": prov_files,
    }
    pf = BATTERY / "PROVENANCE.json"
    if pf.exists():
        shutil.copy2(pf, pf.with_name("PROVENANCE.json.premix"))
    pf.write_text(json.dumps(prov, indent=2))
    print(f"[relabel] wrote {pf}")

    # ---- VERIFICATION ---------------------------------------------------------
    print("\n[verify] === anchors: matched sets should reproduce prior oracle_v3 labels ===")
    for k, v in anchors.items():
        print(f"[verify]   corr_with_old({k}) = {v:.5f}  (expect ~1.0)")

    print("[verify] === random_32k determinism self-consistency ===")
    z = np.load(BATTERY / "random_32k_oracle.npz", allow_pickle=True)
    seqs = [str(s) for s in z["sequences"][:2000]]
    re_pred = oracle.predict(seqs)
    print(f"[verify]   corr(re-score, stored) random_32k[:2000] = "
          f"{_corr(re_pred, z['oracle_mean'][:2000]):.6f}  (expect ~1.0)")

    print("[verify] === gc_sweep U-shape (mean label per GC bin) ===")
    z = np.load(BATTERY / "gc_sweep_oracle.npz", allow_pickle=True)
    if "gc_bin" in z.files:
        om = np.asarray(z["oracle_mean"], np.float64)
        gb = np.asarray(z["gc_bin"], np.float64)
        for b in sorted(set(gb.tolist())):
            m = gb == b
            print(f"[verify]   gc={b:.2f}  mean_label={np.nanmean(om[m]):+.3f}  n={int(m.sum())}")

    print(f"\n[verify] all_finite_across_relabeled_sets = {all_finite}")
    print("[relabel] DONE.")


if __name__ == "__main__":
    main()
