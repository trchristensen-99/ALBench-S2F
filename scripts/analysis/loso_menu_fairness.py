"""LOSO-CV "menu fairness" analysis for the reservoir-agnostic deploy MENU.

Question: does a reservoir-AGNOSTIC menu (a small fixed set of HP-config SIGNATURES,
ElasticNet-reweighted per deployment) reach near-best performance on each HELD-OUT
reservoir vs that reservoir's own explicit-best greedy ensemble? If menu-on-held-out
sits on the y=x diagonal vs explicit-best, the menu is unbiased / not overfit to the
reservoirs used to pick it.

Reuses greedy_deploy_select.py's exact functions: hp_signature/*_band, load_pool_models,
fit_stack, greedy_select, knee_n.

For each held-out reservoir R (D=30000):
  1. EXPLICIT-BEST(R): greedy_select on R's OWN pooled models (all hp_strategies) with
     diversity -> knee-N ensemble -> test-oracle Pearson on genomic (+snv/ood).
  2. MENU (from the OTHER reservoirs): greedy_select per other-reservoir, collect the
     signatures at/before each one's knee, AGGREGATE = signatures selected in >=2 of the
     other reservoirs (fallback: top-K by frequency up to median knee-N). Never sees R.
  3. MENU-ON-HELD-OUT(R): in R's cell, pick R's best-val model per menu signature,
     fit_stack their val preds on R's val_labels, eval test-oracle Pearson genomic
     (+snv/ood) using each model's PER-SET test preds re-weighted by the same stack.
  4. Record explicit_best(R) vs menu_on_heldout(R) and the gap.

CPU/BLAS-bound. Run: OMP_NUM_THREADS=4 uv run --no-sync python scripts/analysis/loso_menu_fairness.py
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from greedy_deploy_select import (  # noqa: E402
    fit_stack,
    greedy_select,
    hp_signature,
    knee_n,
    load_pool_models,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
BAKE = REPO / "outputs/hp_step1_bakeoff_e100"
D = 30000
RESERVOIRS = [
    "genomic",
    "dinuc_shuffle",
    "evoaug_heavy",
    "gc_matched",
    "motif_planted_v2",
    "phylogenetic_zoonomia",
    "uncertainty_guided",
]
SEEDDIR = "seed42_0"
MAX_N = 20
PREFILTER = 120
FRAC = 0.90
MENU_MIN_VOTES = 2  # a signature enters the menu if selected in >=2 other reservoirs

# extra per-set eval keys: (labels_key, npz_pred_key)
EVAL_SETS = {
    "genomic": ("oracle_genomic", "test_pred_genomic"),
    "snv": ("oracle_snv_ref", "test_pred_snv_ref"),
    "ood": ("oracle_ood", "test_pred_ood"),
}


def cell_dir(r):
    return BAKE / f"k562_{r}_d{D}" / SEEDDIR


def load_persets(models, labels_npz_path):
    """Attach per-set (val already present) test preds to each model, plus the per-set
    oracle labels. Returns dict set-> (oracle_labels, {model_id: pred}). Models that lack
    a per-set pred or whose length mismatches are dropped for THAT set."""
    lz = np.load(labels_npz_path)
    out = {}
    for s, (lab_key, _pred_key) in EVAL_SETS.items():
        if lab_key in lz.files:
            out[s] = {"oracle": lz[lab_key].astype(np.float64), "preds": {}}
    return out


def eval_stack_on_sets(chosen_models, val_y, sets, cell):
    """Fit the positive ElasticNet stack on chosen models' VAL preds -> apply the SAME
    linear stack to each per-set test-pred matrix -> oracle Pearson per set.

    We refit via fit_stack per set (val_X constant, test_X = that set's per-set preds),
    so the val-fit weights are identical across sets and only the test matrix changes."""
    val_X = np.vstack([m["val"] for m in chosen_models])
    results = {}
    for s, (lab_key, pred_key) in EVAL_SETS.items():
        oracle = sets.get(s)
        if oracle is None:
            continue
        try:
            test_cols = []
            ok = True
            for m in chosen_models:
                p = m.get("_perset", {}).get(pred_key)
                if p is None or p.shape[0] != oracle["oracle"].shape[0]:
                    ok = False
                    break
                test_cols.append(p)
            if not ok:
                continue
            test_X = np.vstack(test_cols)
            _vpred, tpred = fit_stack(val_X, val_y, test_X)
            mm = np.isfinite(tpred) & np.isfinite(oracle["oracle"])
            if mm.sum() > 3:
                results[s] = float(pearsonr(tpred[mm], oracle["oracle"][mm])[0])
        except Exception:
            continue
    return results


def load_cell(r):
    """load_pool_models on the whole cell dir picks up ALL hp_strategy subdirs
    (glob '*/r*_meta.json'), and grabs the first labels.npz (shared val within a cell)."""
    cell = cell_dir(r)
    if not cell.exists():
        raise FileNotFoundError(f"missing cell {cell}")
    models, labels = load_pool_models(cell)
    # attach per-set test preds by re-reading each model's npz via id
    # load_pool_models set id = meta model_id or npz.stem; we re-glob to map.
    import glob as _g

    metas = sorted(_g.glob(str(cell / "*/r*_meta.json")))
    id_to_npz = {}
    for mp in metas:
        npz = Path(mp.replace("_meta.json", ".npz"))
        try:
            mj = json.loads(Path(mp).read_text())
        except Exception:
            continue
        mid = mj.get("model_id", npz.stem)
        id_to_npz[mid] = npz
        id_to_npz[npz.stem] = npz
    persets = None
    lf = sorted(cell.glob("*/labels.npz"))
    if lf:
        persets = load_persets(models, lf[0])
    for m in models:
        m["_perset"] = {}
        npz = id_to_npz.get(m["id"])
        if npz is None or not npz.exists():
            continue
        try:
            d = np.load(npz)
            for _s, (_lk, pk) in EVAL_SETS.items():
                if pk in d.files:
                    m["_perset"][pk] = d[pk].astype(np.float64)
        except Exception:
            continue
    return models, labels, persets


def greedy_knee_chosen(models, labels):
    """Run greedy_select (diversity=True) -> knee-N -> the chosen models at/before knee."""
    rng = np.random.default_rng(0)
    curve, chosen = greedy_select(models, labels, MAX_N, PREFILTER, True, rng)
    n = knee_n(curve, FRAC)
    n = max(1, min(n, len(chosen)))
    return curve, chosen[:n], n


def main():
    # 1) load all available reservoir cells once
    cells = {}
    skipped = {}
    for r in RESERVOIRS:
        try:
            models, labels, persets = load_cell(r)
            if len(models) < 3:
                skipped[r] = f"only {len(models)} valid models"
                continue
            cells[r] = {
                "models": models,
                "labels": labels,
                "persets": persets,
            }
            print(f"[load] {r}: {len(models)} valid models", flush=True)
        except Exception as e:
            skipped[r] = f"load error: {e}"
            print(f"[skip] {r}: {e}", flush=True)

    # 2) precompute per-reservoir greedy knee selection (both for explicit-best AND
    #    for building menus from the OTHERS)
    per_res = {}
    for r, c in cells.items():
        try:
            curve, chosen, nstar = greedy_knee_chosen(c["models"], c["labels"])
            sigs = [tuple(m["sig"]) for m in chosen]
            per_res[r] = {"curve": curve, "chosen": chosen, "nstar": nstar, "sigs": sigs}
            print(f"[greedy] {r}: knee N*={nstar}, sigs={sigs}", flush=True)
        except Exception as e:
            skipped[r] = f"greedy error: {e}"
            print(f"[skip-greedy] {r}: {e}", flush=True)

    active = [r for r in cells if r in per_res]

    def build_menu(exclude_r):
        """Aggregate signatures from all OTHER reservoirs. Rule: a signature enters the
        menu if selected at/before knee in >= MENU_MIN_VOTES other reservoirs. Fallback:
        top-K by vote up to the median other-reservoir knee-N if the >=2 rule is empty."""
        from collections import Counter

        votes = Counter()
        knees = []
        for r in active:
            if r == exclude_r:
                continue
            for sig in set(per_res[r]["sigs"]):  # count each reservoir once per sig
                votes[sig] += 1
            knees.append(per_res[r]["nstar"])
        menu = [sig for sig, v in votes.items() if v >= MENU_MIN_VOTES]
        rule = f">={MENU_MIN_VOTES}-vote"
        if not menu:
            k = int(round(np.median(knees))) if knees else 1
            menu = [sig for sig, _ in votes.most_common(max(1, k))]
            rule = f"fallback top-{max(1,k)} by vote"
        return menu, dict(votes), rule

    def apply_menu(r, menu):
        """In R's cell, pick R's best-val model per menu signature, stack, eval."""
        c = cells[r]
        by_sig = {}
        for m in c["models"]:
            sig = tuple(m["sig"])
            if sig in menu:
                if sig not in by_sig or m.get("solo_val_r", -1) > by_sig[sig].get(
                    "solo_val_r", -1
                ):
                    # solo_val_r may not be set yet; compute
                    pass
        # compute solo_val_r for selection (best-val per sig)
        val_y = c["labels"]["val_labels"]
        for m in c["models"]:
            if "solo_val_r" not in m:
                mm = np.isfinite(m["val"]) & np.isfinite(val_y)
                m["solo_val_r"] = (
                    pearsonr(m["val"][mm], val_y[mm])[0] if mm.sum() > 3 else -1.0
                )
        by_sig = {}
        for m in c["models"]:
            sig = tuple(m["sig"])
            if sig in menu:
                if sig not in by_sig or m["solo_val_r"] > by_sig[sig]["solo_val_r"]:
                    by_sig[sig] = m
        chosen = list(by_sig.values())
        if not chosen:
            return None, 0
        res = eval_stack_on_sets(chosen, val_y, c["persets"], r)
        return res, len(chosen)

    records = {}
    for r in active:
        try:
            # explicit-best: eval knee ensemble on per-set oracles
            eb_chosen = per_res[r]["chosen"]
            eb = eval_stack_on_sets(
                eb_chosen, cells[r]["labels"]["val_labels"], cells[r]["persets"], r
            )
            menu, votes, rule = build_menu(r)
            mo, msize = apply_menu(r, menu)
            rec = {
                "explicit_best": eb,
                "explicit_best_n": len(eb_chosen),
                "menu_on_heldout": mo,
                "menu_size": len(menu),
                "menu_matched_in_R": msize,
                "menu_rule": rule,
            }
            if eb.get("genomic") is not None and mo and mo.get("genomic") is not None:
                rec["gap_genomic"] = mo["genomic"] - eb["genomic"]
            records[r] = rec
            gg = rec.get("gap_genomic")
            print(
                f"[loso] {r}: explicit={eb.get('genomic'):.4f} "
                f"menu={ (mo or {}).get('genomic') } gap={gg} "
                f"menu_size={len(menu)} matched={msize}",
                flush=True,
            )
        except Exception as e:
            skipped[r] = f"loso error: {e}\n{traceback.format_exc()}"
            print(f"[skip-loso] {r}: {e}", flush=True)

    # aggregate
    gaps = [
        records[r]["gap_genomic"]
        for r in records
        if records[r].get("gap_genomic") is not None
    ]
    agg = {
        "mean_gap_genomic": float(np.mean(gaps)) if gaps else None,
        "worst_gap_genomic": float(np.min(gaps)) if gaps else None,  # most negative
        "median_gap_genomic": float(np.median(gaps)) if gaps else None,
        "n_reservoirs": len(records),
    }

    out = {
        "config": {
            "D": D,
            "seeddir": SEEDDIR,
            "max_n": MAX_N,
            "prefilter": PREFILTER,
            "knee_frac": FRAC,
            "menu_min_votes": MENU_MIN_VOTES,
            "eval_sets": list(EVAL_SETS.keys()),
        },
        "per_reservoir": {
            r: {
                "explicit_best": records[r]["explicit_best"].get("genomic"),
                "menu_on_heldout": (records[r]["menu_on_heldout"] or {}).get("genomic"),
                "gap": records[r].get("gap_genomic"),
                "menu_size": records[r]["menu_size"],
                "menu_matched_in_R": records[r]["menu_matched_in_R"],
                "explicit_best_n": records[r]["explicit_best_n"],
                "menu_rule": records[r]["menu_rule"],
                "explicit_best_allsets": records[r]["explicit_best"],
                "menu_on_heldout_allsets": records[r]["menu_on_heldout"],
            }
            for r in records
        },
        "aggregate": agg,
        "skipped": skipped,
    }
    jpath = BAKE / "loso_menu_fairness.json"
    jpath.write_text(json.dumps(out, indent=2))
    print(f"wrote {jpath}", flush=True)

    # ---- figure ----
    rs = [r for r in RESERVOIRS if r in records and records[r].get("gap_genomic") is not None]
    eb = np.array([records[r]["explicit_best"]["genomic"] for r in rs])
    mo = np.array([records[r]["menu_on_heldout"]["genomic"] for r in rs])
    gaps_arr = mo - eb

    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(rs))
    w = 0.38
    ax1.bar(x - w / 2, eb, w, label="explicit-best (R's own)", color="#4C72B0")
    ax1.bar(x + w / 2, mo, w, label="menu-on-held-out", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.replace("_", "\n") for r in rs], fontsize=12)
    ax1.set_ylabel("test-oracle Pearson (genomic)", fontsize=15)
    ax1.set_title("Per-held-out-reservoir: explicit-best vs menu", fontsize=16)
    ax1.legend(fontsize=13)
    lo = min(eb.min(), mo.min())
    hi = max(eb.max(), mo.max())
    ax1.set_ylim(lo - 0.03, hi + 0.02)
    for i, (a, b) in enumerate(zip(eb, mo)):
        ax1.text(i + w / 2, b + 0.002, f"{b - a:+.3f}", ha="center", fontsize=10)

    ax2.scatter(eb, mo, s=120, color="#55A868", zorder=3)
    for i, r in enumerate(rs):
        ax2.annotate(r, (eb[i], mo[i]), fontsize=10, xytext=(4, 4),
                     textcoords="offset points")
    dlo = min(eb.min(), mo.min()) - 0.01
    dhi = max(eb.max(), mo.max()) + 0.01
    ax2.plot([dlo, dhi], [dlo, dhi], "k--", lw=1.5, label="y = x (unbiased)")
    ax2.set_xlim(dlo, dhi)
    ax2.set_ylim(dlo, dhi)
    ax2.set_xlabel("explicit-best oracle Pearson", fontsize=15)
    ax2.set_ylabel("menu-on-held-out oracle Pearson", fontsize=15)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=13)
    meang = float(np.mean(gaps_arr))
    worstg = float(np.min(gaps_arr))
    verdict = "UNBIASED (on diagonal)" if meang >= -0.008 else "BIASED (below diagonal)"
    ax2.set_title(
        f"Menu fairness: mean gap {meang:+.4f}, worst {worstg:+.4f}\n{verdict}",
        fontsize=15,
    )
    fig.suptitle(
        "LOSO-CV menu fairness — reservoir-agnostic deploy menu vs per-reservoir best "
        f"(D={D})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ppath = BAKE / "loso_menu_fairness.png"
    fig.savefig(ppath, dpi=140)
    print(f"wrote {ppath}", flush=True)


if __name__ == "__main__":
    main()
