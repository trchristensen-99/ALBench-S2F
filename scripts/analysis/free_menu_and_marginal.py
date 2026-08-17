"""FREE (no-retraining) menu selection + reservoir-marginal curve, from EXISTING bake-off preds.

Task 1: empirical general-val greedy over ALL reservoirs' pooled models -> the menu (configs).
Task 2: same, but over K=1..N reservoir subsets -> ensemble-vs-K curve (diminishing returns =
        how many reservoirs' HP-opt to run).
Also emits with-X / without-X recipes for the bias RETRAINING experiment.

Selection is on a general val = held-out slice of the common battery, equal-weighted across
{genomic, snv_ref, ood}; per-set standardized so no regime dominates. Honest scope: this is a
SELECTION heuristic on single-reservoir-trained preds, NOT a measure of retraining generality.
"""
import os, json, glob, itertools
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BAKE = f"{REPO}/outputs/hp_step1_bakeoff_e100"
RECIPE_DIR = f"{REPO}/configs/deploy_recipes"
RES = ["genomic", "motif_planted_v2", "evoaug_heavy", "dinuc_shuffle", "gc_matched",
       "phylogenetic_zoonomia", "uncertainty_guided", "diversity_guided"]
D = 30000
SETS = [("genomic", "oracle_genomic", "test_pred_genomic"),
        ("snv", "oracle_snv_ref", "test_pred_snv_ref"),
        ("ood", "oracle_ood", "test_pred_ood")]
CFG_KEYS = ["lr", "batch_size", "conv_dropout", "dense_dropout", "n_layers", "width_base",
            "width_jitter", "block_class", "ks", "pct_start", "optimizer", "weight_decay",
            "use_shift_aug", "shift_max", "use_evoaug", "activation", "loss"]
EN_KW = dict(l1_ratio=[0.1, 0.5, 0.9, 1.0], positive=True, cv=3, max_iter=5000, n_jobs=1)
MAX_POOL, MAX_SIZE = 40, 8

# ---- load: per reservoir, models with per-set battery preds + config; common oracle ----
oracle_v, oracle_e, val_idx, eval_idx = {}, {}, {}, {}
by_res = {}
for r in RES:
    cell = f"{BAKE}/k562_{r}_d{D}/seed42_0"
    lf = sorted(glob.glob(f"{cell}/*/labels.npz"))
    if not lf:
        continue
    lz = np.load(lf[0])
    if not oracle_v:  # set up common oracle + split once (battery shared across cells)
        for sname, ok, pk in SETS:
            if ok not in lz.files:
                continue
            y = lz[ok].astype(np.float64)
            n = len(y); h = n // 2
            val_idx[sname] = np.arange(h); eval_idx[sname] = np.arange(h, n)
            oracle_v[sname] = y[val_idx[sname]]; oracle_e[sname] = y[eval_idx[sname]]
    models = []
    for mp in sorted(glob.glob(f"{cell}/*/r*_meta.json")):
        try:
            mj = json.load(open(mp))
        except Exception:
            continue
        vp = mj.get("val_pearson")
        if vp is None or not np.isfinite(vp) or "hp" not in mj:
            continue
        npz = mp.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        try:
            d = np.load(npz)
        except Exception:
            continue
        preds = {}
        ok_all = True
        for sname, ok, pk in SETS:
            if sname not in val_idx or pk not in d.files:
                ok_all = False; break
            preds[sname] = d[pk].astype(np.float64)
        if not ok_all:
            continue
        models.append(dict(res=r, hp=mj["hp"], vp=float(vp), preds=preds))
    if models:
        by_res[r] = models
        print(f"[load] {r}: {len(models)} models", flush=True)
ACTIVE = list(by_res)
SNAMES = list(val_idx)
print(f"active={ACTIVE}  sets={SNAMES}", flush=True)


def single_score(m):  # general val score of one model (mean per-set pearson on val half)
    ps = []
    for s in SNAMES:
        p = m["preds"][s][val_idx[s]]
        ps.append(pearsonr(p, oracle_v[s])[0])
    return float(np.mean(ps))


def fit_general(models):
    Xv, Yv, stats = [], [], {}
    for s in SNAMES:
        P = np.column_stack([m["preds"][s][val_idx[s]] for m in models])
        y = oracle_v[s]
        mu, sd = P.mean(0), P.std(0) + 1e-9
        ym, ysd = y.mean(), y.std() + 1e-9
        stats[s] = (mu, sd, ym, ysd)
        Xv.append((P - mu) / sd); Yv.append((y - ym) / ysd)
    en = ElasticNetCV(**EN_KW).fit(np.vstack(Xv), np.concatenate(Yv))
    return en, stats


def score_half(en, stats, models, idx, oracle):
    ps = []
    for s in SNAMES:
        mu, sd, ym, ysd = stats[s]
        P = np.column_stack([m["preds"][s][idx[s]] for m in models])
        pred = en.predict((P - mu) / sd)
        ps.append(pearsonr(pred, oracle[s])[0])
    return float(np.mean(ps))


def greedy_general(pool):
    pool = sorted(pool, key=single_score, reverse=True)[:MAX_POOL]
    selected, best_val, best_en, best_stats, remaining = [], -np.inf, None, None, list(pool)
    while remaining and len(selected) < MAX_SIZE:
        best = None
        for c in remaining:
            trial = selected + [c]
            en, stats = fit_general(trial)
            vs = score_half(en, stats, trial, val_idx, oracle_v)
            if best is None or vs > best[0]:
                best = (vs, c, en, stats)
        vs, c, en, stats = best
        if vs <= best_val + 1e-4 and selected:
            break
        selected.append(c); remaining.remove(c); best_val = vs; best_en, best_stats = en, stats
    ev = score_half(best_en, best_stats, selected, eval_idx, oracle_e) if selected else None
    return selected, best_val, ev


def recipe_of(models):
    seen, out = set(), []
    for m in models:
        hp = m["hp"]
        k = (hp.get("block_class"), hp.get("optimizer"), hp.get("n_layers"), hp.get("width_base"))
        if k in seen:
            continue
        seen.add(k); out.append({kk: hp[kk] for kk in CFG_KEYS if kk in hp})
    return out


# ---- Task 1: full menu ----
allpool = [m for r in ACTIVE for m in by_res[r]]
menu, mv, me = greedy_general(allpool)
print(f"\n=== TASK1 MENU (all reservoirs): {len(menu)} members, val={mv:.4f} eval={me:.4f} ===", flush=True)
for m in menu:
    hp = m["hp"]
    print(f"  [{m['res']:<18}] {hp.get('block_class')}/{hp.get('optimizer')} "
          f"L{hp.get('n_layers')} w{hp.get('width_base')} lr{hp.get('lr'):.1e}", flush=True)
json.dump(recipe_of(menu), open(f"{RECIPE_DIR}/free_menu_d{D}.json", "w"), indent=1)

# ---- Task 2: marginal vs K reservoirs ----
print("\n=== TASK2 marginal (eval vs #reservoirs) ===", flush=True)
agg = []
for K in range(1, len(ACTIVE)):
    subs = list(itertools.combinations(ACTIVE, K))[:30]
    evs = []
    for S in subs:
        pool = [m for r in S for m in by_res[r]]
        _, _, ev = greedy_general(pool)
        if ev is not None:
            evs.append(ev)
    if evs:
        agg.append([K, float(np.mean(evs)), float(np.std(evs)), len(evs)])
        g = agg[-1][1] - agg[-2][1] if len(agg) > 1 else 0.0
        print(f"K={K}: eval={np.mean(evs):.4f}±{np.std(evs):.4f} marginal={g:+.4f} (n={len(evs)})", flush=True)

# ---- emit with-X / without-X recipes for the bias retraining experiment ----
BIAS = {"dinuc_shuffle": (["genomic", "motif_planted_v2", "evoaug_heavy"],
                          ["genomic", "motif_planted_v2", "dinuc_shuffle"]),
        "gc_matched": (["genomic", "motif_planted_v2", "evoaug_heavy"],
                       ["genomic", "motif_planted_v2", "gc_matched"])}
bias_out = {}
for X, (without, withX) in BIAS.items():
    if not all(r in by_res for r in without + withX):
        continue
    mw, _, ewo = greedy_general([m for r in without for m in by_res[r]])
    mi, _, ewi = greedy_general([m for r in withX for m in by_res[r]])
    json.dump(recipe_of(mw), open(f"{RECIPE_DIR}/bias_without_{X}_d{D}.json", "w"), indent=1)
    json.dump(recipe_of(mi), open(f"{RECIPE_DIR}/bias_with_{X}_d{D}.json", "w"), indent=1)
    bias_out[X] = dict(without_srcs=without, with_srcs=withX,
                       without_eval_free=ewo, with_eval_free=ewi,
                       n_without=len(mw), n_with=len(mi))
    print(f"[bias {X}] free-eval without={ewo:.4f} (n={len(mw)}) | with={ewi:.4f} (n={len(mi)})", flush=True)

json.dump(dict(menu_val=mv, menu_eval=me, marginal=agg, bias=bias_out, sets=SNAMES),
          open(f"{BAKE}/free_menu_and_marginal.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ks = [a[0] for a in agg]; ms = [a[1] for a in agg]; ss = [a[2] for a in agg]
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.errorbar(ks, ms, yerr=ss, color="#1d4ed8", lw=2.6, marker="o", capsize=3)
ax.set_xlabel("# reservoir strategies' HP-opt pooled (K)")
ax.set_ylabel("general ensemble eval (mean over genomic/SNV/OOD)")
ax.set_title("Diminishing returns: general ensemble vs # reservoirs (D=30k, FREE selection)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{BAKE}/free_marginal.png", dpi=140)
print("WROTE free_marginal.png + free_menu_d30000.json + bias recipes", flush=True)
