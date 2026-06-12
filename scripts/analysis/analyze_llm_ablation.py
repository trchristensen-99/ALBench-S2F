"""Phase-0b LLM CONFOUND ablation: read every cell's full-pool ensemble and contrast
each probe against the baseline to separate FAIRNESS (Part A) from MECHANISM (Part B).

Part A — is the LLM win FAIR? Strip the context scaffolding the LLM proposer gets
(ctxfull → ctxnokb → ctxnone) and swap the persona to a blank / deliberately misguided
prompt. If the win survives ctxnone + a neutral persona, it isn't a context handout.

Part B — is the feedback loop REAL? Corrupt or remove the feedback the proposer sees:
  _shuffle   — permute score↔config pairing (same score multiset, wrong attribution)
  _nohist    — hide history entirely (propose from priors only)
  _hist5/_histfull/_chrono/_worst — vary depth & ordering of the feedback
  _n1/_n5    — proposals per call (1 vs 5 vs baseline 2)
  _rep       — identical config, different hp_seed → the NOISE FLOOR for every delta
If _shuffle / _nohist drop to baseline-minus-noise, the model is genuinely USING the
loop; if they don't move beyond _rep noise, the "feedback" is decorative.

Metric = full-pool ElasticNet ensemble oracle-Pearson (same fit_elasticnet_stack the
deploy uses), averaged across seeds, with best-single val/oracle for reference. Read the
_rep row FIRST: every other delta is only meaningful relative to that noise floor.

CPU/BLAS-bound — run via srun, never the login node.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from albench.ensemble import fit_elasticnet_stack
from scripts.analysis.greedy_deploy_select import load_pool_models

# Display order: Part A context/persona cells, then Part B feedback probes. The baseline
# (llm_default_ctxnone — neutral persona, no context handout, real 2-per-round feedback)
# is the reference every delta is measured against.
PART_A = [
    "llm_default_ctxfull",
    "llm_default_ctxnokb",
    "llm_default_ctxnone",
    "llm_exploit_ctxfull",
    "llm_exploit_ctxnokb",
    "llm_exploit_ctxnone",
    "llm_blank_ctxnone",
    "llm_misguided_ctxnone",
]
PART_B = [
    "llm_default_ctxnone_rep",
    "llm_default_ctxnone_shuffle",
    "llm_default_ctxnone_nohist",
    "llm_default_ctxnone_hist5",
    "llm_default_ctxnone_histfull",
    "llm_default_ctxnone_chrono",
    "llm_default_ctxnone_worst",
    "llm_default_ctxnone_n1",
    "llm_default_ctxnone_n5",
]
BASELINE = "llm_default_ctxnone"


def cell_metrics(cell_dir: Path):
    try:
        models, labels = load_pool_models(cell_dir)
    except SystemExit:
        return None
    if len(models) < 1:
        return None
    val_y, oracle = labels["val_labels"], labels["test_oracle"]
    val_X = np.vstack([m["val"] for m in models])
    test_X = np.vstack([m["test"] for m in models])
    _vp, test_pred = fit_elasticnet_stack(val_X, val_y, test_X)
    mt = np.isfinite(test_pred) & np.isfinite(oracle)
    ens_oracle = float(pearsonr(test_pred[mt], oracle[mt])[0]) if mt.sum() > 3 else float("nan")
    solos = []
    for m in models:
        mm = np.isfinite(m["val"]) & np.isfinite(val_y)
        solos.append(pearsonr(m["val"][mm], val_y[mm])[0] if mm.sum() > 3 else np.nan)
    best_val = float(np.nanmax(solos)) if solos else float("nan")
    return ens_oracle, best_val, len(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl_root", default="outputs/hp_llm_ablation_e100")
    args = ap.parse_args()
    root = Path(args.abl_root)

    agg = defaultdict(list)
    for labels in root.glob("**/labels.npz"):
        d = labels.parent
        res = cell_metrics(d)
        if res is not None:
            agg[d.name].append(res)

    def row(label):
        recs = agg.get(label)
        if not recs:
            return None
        return {
            "label": label,
            "n_seeds": len(recs),
            "ens_oracle": float(np.mean([r[0] for r in recs])),
            "ens_oracle_sd": float(np.std([r[0] for r in recs])),
            "best_val": float(np.mean([r[1] for r in recs])),
            "n_mem": int(np.mean([r[2] for r in recs])),
        }

    base = row(BASELINE)
    base_or = base["ens_oracle"] if base else float("nan")
    rep = row("llm_default_ctxnone_rep")
    noise = rep["ens_oracle_sd"] if rep else float("nan")

    def emit(title, labels):
        print(f"\n=== {title} ===")
        print(
            f"{'cell':32s} {'seeds':>5} {'ens_oracle':>10} {'Δvs_base':>9} "
            f"{'±sd':>6} {'best_val':>8} {'mem':>4}"
        )
        for lab in labels:
            r = row(lab)
            if r is None:
                print(f"{lab:32s}    (missing)")
                continue
            delta = r["ens_oracle"] - base_or
            mark = ""
            if lab != BASELINE and np.isfinite(noise) and abs(delta) > 2 * noise:
                mark = "  *"  # exceeds 2x the replicate noise floor
            print(
                f"{r['label']:32s} {r['n_seeds']:>5} {r['ens_oracle']:>10.4f} "
                f"{delta:>+9.4f} {r['ens_oracle_sd']:>6.4f} {r['best_val']:>8.4f} "
                f"{r['n_mem']:>4}{mark}"
            )

    print(f"baseline {BASELINE} ens_oracle = {base_or:.4f}")
    print(f"replicate (_rep) noise floor sd = {noise:.4f}  (2σ = {2 * noise:.4f}); '*' = |Δ|>2σ")
    emit("PART A — fairness (context scaffolding + persona)", PART_A)
    emit("PART B — mechanism (feedback corruption / depth / count)", PART_B)


if __name__ == "__main__":
    main()
