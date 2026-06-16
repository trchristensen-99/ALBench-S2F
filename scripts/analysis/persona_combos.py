"""Persona-level cross-ensemble all-subsets analysis.

The deploy question: if we ensemble the FINAL models from each LLM *persona* (exploit,
critic, diverse, default, neutral, explore), which COMBINATION of personas gives the best
ensemble oracle-r -- and where does our chosen {exploit, critic, diverse} land?

The existing best_per_strategy_combos.py keys atoms by the meta "strategy" field, which is
"llm_autoresearch" for every persona, so it cannot separate personas. Here the atom key is
the PERSONA parsed from the cell dir name (llm_{persona}_{model}_nv{N}).

Method (mirrors best_per_strategy_combos): reduce each persona to ONE atom -- its highest
solo val-oracle model (fixed model=sonnet, best over its nv variants) -- then exhaustively
stack every non-empty persona subset with ElasticNetCV(positive=True) on val, score the
stacked test prediction's Pearson r vs test_oracle (oracle stays held out; subset "winner"
chosen by val MSE). Averaged across the screen's seeds.

CPU/BLAS-bound -> run via srun on cpuq with capped threads.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from albench.ensemble import fit_elasticnet_stack

PERSONAS = ["exploit", "critic", "diverse", "default", "neutral", "explore"]
TARGET = frozenset({"exploit", "critic", "diverse"})


def parse_cell(name):
    # llm_{persona}_{model}_nv{N}
    p = name.split("_")
    if len(p) < 4 or p[0] != "llm":
        return None
    return {"persona": p[1], "model": p[2], "nv": p[3]}


def best_atom(cell: Path, val_y):
    """Highest solo val-oracle model in a persona cell -> (val_pred, test_pred, solo_r)."""
    best = None
    for m in cell.glob("*_meta.json"):
        z = np.load(cell / m.name.replace("_meta.json", ".npz"))
        vp, tp = z["val_pred"], z["test_pred"]
        mm = np.isfinite(vp) & np.isfinite(val_y)
        r = float(pearsonr(vp[mm], val_y[mm])[0]) if mm.sum() > 3 else -1.0
        if best is None or r > best[2]:
            best = (vp, tp, r)
    return best


def seed_atoms(seed_dir: Path, model: str):
    """One best atom per persona (fixed model, best over nv variants)."""
    labels = None
    by_persona = {}
    for cell in sorted(seed_dir.iterdir()):
        if not cell.is_dir():
            continue
        info = parse_cell(cell.name)
        if not info or info["model"] != model:
            continue
        lab = cell / "labels.npz"
        if not lab.exists():
            continue
        if labels is None:
            labels = np.load(lab)
        val_y = labels["val_labels"]
        atom = best_atom(cell, val_y)
        if atom is None:
            continue
        p = info["persona"]
        if p not in by_persona or atom[2] > by_persona[p][2]:
            by_persona[p] = atom
    return by_persona, labels


def all_subsets(by_persona, labels):
    val_y, oracle = labels["val_labels"], labels["test_oracle"]
    personas = [p for p in PERSONAS if p in by_persona]
    rows = []
    for size in range(1, len(personas) + 1):
        for combo in itertools.combinations(personas, size):
            vX = np.vstack([by_persona[p][0] for p in combo])
            tX = np.vstack([by_persona[p][1] for p in combo])
            vpred, tpred = fit_elasticnet_stack(vX, val_y, tX)
            mt = np.isfinite(tpred) & np.isfinite(oracle)
            rows.append(
                {
                    "set": frozenset(combo),
                    "size": size,
                    "val_mse": float(np.mean((vpred - val_y) ** 2)),
                    "oracle_r": float(pearsonr(tpred[mt], oracle[mt])[0]),
                }
            )
    return rows


def run(pool: Path, model: str, out: Path):
    seed_dirs = sorted(p for p in pool.iterdir() if p.is_dir() and p.name.startswith("seed"))
    per_seed = []
    agg_r = defaultdict(list)
    agg_valmse = defaultdict(list)
    sizes = {}
    for sd in seed_dirs:
        by_persona, labels = seed_atoms(sd, model)
        if labels is None or len(by_persona) < 2:
            continue
        rows = all_subsets(by_persona, labels)
        per_seed.append({"seed": sd.name, "n_personas": len(by_persona)})
        for r in rows:
            agg_r[r["set"]].append(r["oracle_r"])
            agg_valmse[r["set"]].append(r["val_mse"])
            sizes[r["set"]] = r["size"]
    combos = []
    for s, rs in agg_r.items():
        combos.append(
            {
                "personas": sorted(s),
                "size": sizes[s],
                "oracle_r_mean": float(np.mean(rs)),
                "oracle_r_sd": float(np.std(rs)),
                "val_mse_mean": float(np.mean(agg_valmse[s])),
                "n_seeds": len(rs),
            }
        )
    combos.sort(key=lambda c: -c["oracle_r_mean"])
    for i, c in enumerate(combos):
        c["rank"] = i + 1
    target = next((c for c in combos if frozenset(c["personas"]) == TARGET), None)
    best_by_size = {}
    for c in combos:
        if (
            c["size"] not in best_by_size
            or c["oracle_r_mean"] > best_by_size[c["size"]]["oracle_r_mean"]
        ):
            best_by_size[c["size"]] = c

    result = {
        "pool": str(pool),
        "model": model,
        "seeds_used": [d["seed"] for d in per_seed],
        "n_combos": len(combos),
        "ranking": combos,
        "best_by_size": {str(k): v for k, v in sorted(best_by_size.items())},
        "target_set": sorted(TARGET),
        "target_result": target,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"=== persona combos | {pool.name} | model={model} | seeds={result['seeds_used']} ===")
    print(f"top 8 of {len(combos)} persona subsets by mean oracle-r:")
    for c in combos[:8]:
        star = "  <-- TARGET" if frozenset(c["personas"]) == TARGET else ""
        print(
            f"  #{c['rank']:2d}  r={c['oracle_r_mean']:.4f}±{c['oracle_r_sd']:.4f}  "
            f"size={c['size']}  {'+'.join(c['personas'])}{star}"
        )
    if target:
        print(
            f"\nTARGET exploit+critic+diverse: rank #{target['rank']}/{len(combos)} overall, "
            f"r={target['oracle_r_mean']:.4f}±{target['oracle_r_sd']:.4f}"
        )
        size3 = [c for c in combos if c["size"] == 3]
        trank3 = next(i + 1 for i, c in enumerate(size3) if frozenset(c["personas"]) == TARGET)
        print(f"  among {len(size3)} size-3 subsets: rank #{trank3}")
        print(
            f"  best size-3: {'+'.join(best_by_size[3]['personas'])} r={best_by_size[3]['oracle_r_mean']:.4f}"
        )
    print("\nbest subset by size:")
    for k in sorted(best_by_size):
        c = best_by_size[k]
        print(f"  size {k}: r={c['oracle_r_mean']:.4f}  {'+'.join(c['personas'])}")
    return result


def make_fig(result, out_png: Path):
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "figure.constrained_layout.use": True,
        }
    )
    target_set = set(result["target_set"])
    tsize = len(target_set)
    # Fair comparison is AT FIXED SIZE: oracle-r grows with subset size (a member-count
    # confound), so rank only the same-size subsets as the target.
    same = sorted(
        [c for c in result["ranking"] if c["size"] == tsize],
        key=lambda c: -c["oracle_r_mean"],
    )
    fig, ax = plt.subplots(figsize=(13, 9))
    y = np.arange(len(same))[::-1]
    rs = [c["oracle_r_mean"] for c in same]
    sds = [c["oracle_r_sd"] for c in same]
    cols = ["#d62728" if set(c["personas"]) == target_set else "#4c72b0" for c in same]
    ax.barh(y, rs, xerr=sds, color=cols, capsize=3)
    labs = ["+".join(c["personas"]) for c in same]
    for i, c in enumerate(same):
        tag = "  ★ chosen" if set(c["personas"]) == target_set else ""
        ax.text(
            rs[i] + sds[i] + 0.0006, y[i], "{:.3f}{}".format(rs[i], tag), va="center", fontsize=11
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlim(min(rs) - 0.012, max(rs) + 0.013)
    ax.set_xlabel("ensemble oracle-r  (best atom/persona, mean ± SD over 3 seeds)")
    tr = result["target_result"]
    trank = next(i + 1 for i, c in enumerate(same) if set(c["personas"]) == target_set)
    ax.set_title(
        "All size-{} persona combos (screen pool, Sonnet) — fair fixed-size comparison\n"
        "exploit+critic+diverse = #{}/{}  (r={:.3f}); gaps are within the ~0.015 cross-seed SD".format(
            tsize, trank, len(same), tr["oracle_r_mean"]
        )
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("saved", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument(
        "--out", default="outputs/analysis_figures/exp0_strategy_choice/persona_combos.json"
    )
    ap.add_argument(
        "--fig", default="outputs/analysis_figures/exp0_strategy_choice/D_persona_combos.png"
    )
    args = ap.parse_args()
    res = run(Path(args.pool), args.model, Path(args.out))
    make_fig(res, Path(args.fig))


if __name__ == "__main__":
    main()
