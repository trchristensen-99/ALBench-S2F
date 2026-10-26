#!/usr/bin/env python
"""Analyses of the Oct-23 launches: prompt-style arms, multi-agent loop pilot, precision A/B.

prompts   default vs explore / reflect / explore_reflect prompt styles, 9 paired runs on genomic 30k pools r1..r9
agent     default loop vs the multi-agent loop (proposer, coder, evaluator, reflector), 5 paired runs on pools r1..r5, with LLM cost per run
precision the 33-architecture menu retrained under bf16 / fp16 / tf32 against the fp32 baseline (paired by architecture)

  python scripts/analysis/analyze_oct25.py --out outputs/justification_oct22 [--only prompts,agent,precision]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct22 as a  # noqa: E402
import make_justification as mj  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EV = a.EV
O22, MR_ = a.O22, a.MR_
DATA: dict = {}
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, TEAL = (
    a.BLUE,
    a.ORANGE,
    a.GREEN,
    a.RED,
    a.GREY,
    a.PURPLE,
    a.TEAL,
)


def run_stats(d, first_n=20):
    """One search run -> ensemble r on each eval set (all first_n models, NNLS), mean/max single-model val r, mean pairwise val-pred correlation."""
    c = mj.load_cell(d)
    if c is None:
        return None
    n = min(first_n, c["V"].shape[1])
    cols = list(range(n))
    r = a.nnls_ens(c["V"], c["yv"], c["T"], c["yt"], cols)
    V = c["V"][:, cols]
    cc = np.corrcoef(V.T)
    return dict(
        ens=[r[e] for e in EV],
        mean_val=float(np.mean(c["val"][:n])),
        max_val=float(np.max(c["val"][:n])),
        n_models=n,
        diversity=float(1 - cc[np.triu_indices(n, 1)].mean()),
        cell=c,
        cols=cols,
    )


def pooled_triples(stats_list, triples):
    out = []
    for tr in triples:
        parts = [
            (stats_list[i]["cell"], stats_list[i]["cols"]) for i in tr if stats_list[i] is not None
        ]
        if len(parts) < len(tr):
            out.append(None)
            continue
        d = a.stack(parts)
        r = a.nnls_ens(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))
        out.append([r[e] for e in EV])
    return out


def paired(x, y, n_boot=4000, seed=0):
    """paired differences y - x over runs: mean, bootstrap 95% interval, t-test p"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    p = float(stats.ttest_rel(y, x).pvalue) if len(d) > 2 else float("nan")
    return dict(
        mean=float(d.mean()),
        lo=float(np.percentile(bs, 2.5)),
        hi=float(np.percentile(bs, 97.5)),
        p=p,
        sd_diff=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        n=len(d),
        wins=int((d > 0).sum()),
    )


# ---------------------------------------------------------------------------------------------------------------------- prompts
def an_prompts(out):
    arms = {
        "default": "genomic_30k",
        "explore": "genomic_30k_explore",
        "reflect": "genomic_30k_reflect",
        "explore + reflect": "genomic_30k_explore_reflect",
    }
    R = range(1, 10)
    st = {}
    for arm, key in arms.items():
        st[arm] = [run_stats(f"{O22}/{key}/rep{r}") for r in R]
        print(
            f"[prompts] {arm}: runs {sum(s is not None for s in st[arm])}, mean single-run ensemble {np.mean([s['ens'][0] for s in st[arm] if s]):.4f}",
            flush=True,
        )
    triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    res = {"arms": {}, "diffs": {}}
    for arm in arms:
        ens = np.array([s["ens"] for s in st[arm]])
        pt = pooled_triples(st[arm], triples)
        res["arms"][arm] = dict(
            ens=ens.tolist(),
            mean=ens.mean(0).tolist(),
            sd=ens.std(0, ddof=1).tolist(),
            mean_val=float(np.mean([s["mean_val"] for s in st[arm]])),
            max_val=float(np.mean([s["max_val"] for s in st[arm]])),
            diversity=float(np.mean([s["diversity"] for s in st[arm]])),
            pooled_triples=pt,
        )
    for arm in list(arms)[1:]:
        res["diffs"][arm] = {
            e: paired([s["ens"][i] for s in st["default"]], [s["ens"][i] for s in st[arm]])
            for i, e in enumerate(EV)
        }
        res["diffs"][arm]["pooled3_genomic"] = paired(
            [p[0] for p in res["arms"]["default"]["pooled_triples"]],
            [p[0] for p in res["arms"][arm]["pooled_triples"]],
        )
        res["diffs"][arm]["mean_val"] = paired(
            [s["mean_val"] for s in st["default"]], [s["mean_val"] for s in st[arm]]
        )
        d = res["diffs"][arm]["genomic"]
        print(
            f"[prompts] {arm} - default: single-run ensemble genomic {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] p={d['p']:.3f} wins {d['wins']}/{d['n']}; mean val {res['diffs'][arm]['mean_val']['mean']:+.4f}",
            flush=True,
        )
    DATA["prompts"] = res
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    cols = {"default": GREY, "explore": ORANGE, "reflect": TEAL, "explore + reflect": PURPLE}
    for ax, (ei, e) in zip(axs, enumerate(EV)):
        for k, arm in enumerate(arms):
            v = np.array(res["arms"][arm]["ens"])[:, ei]
            ax.scatter(
                np.full(len(v), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)),
                v,
                s=16,
                color=cols[arm],
                alpha=0.7,
            )
            ax.hlines(v.mean(), k - 0.3, k + 0.3, color=cols[arm], lw=2.2)
        for r_ in range(9):
            ax.plot(
                range(4),
                [res["arms"][arm]["ens"][r_][ei] for arm in arms],
                color=GREY,
                alpha=0.15,
                lw=0.8,
            )
        ax.set_xticks(range(4))
        ax.set_xticklabels(["default", "explore", "reflect", "explore\n+ reflect"], fontsize=8.5)
        ax.set_title(
            {"genomic": "genomic test", "ood": "designed set", "random_32k": "random sequences"}[e]
        )
    axs[0].set_ylabel("single-run ensemble r (20 models)")
    fig.suptitle(
        "Prompt styles, 9 paired runs on genomic 30k pools (lines join the same pool; bars are means)",
        fontsize=10.5,
        y=1.02,
    )
    a.savefig(fig, "prompt_styles", out)


# -------------------------------------------------------------------------------------------------------------------- agent pilot
def an_agent(out):
    R = range(1, 6)
    base = [run_stats(f"{O22}/genomic_30k/rep{r}") for r in R]
    ag = [run_stats(f"{O22}/genomic_30k_agent/rep{r}") for r in R]
    ok = [i for i in range(5) if base[i] and ag[i]]
    res = {
        "n_pairs": len(ok),
        "default": [base[i]["ens"] for i in ok],
        "agent": [ag[i]["ens"] for i in ok],
    }
    res["diff"] = {
        e: paired([base[i]["ens"][k] for i in ok], [ag[i]["ens"][k] for i in ok])
        for k, e in enumerate(EV)
    }
    res["mean_val"] = dict(
        default=float(np.mean([base[i]["mean_val"] for i in ok])),
        agent=float(np.mean([ag[i]["mean_val"] for i in ok])),
        diff=paired([base[i]["mean_val"] for i in ok], [ag[i]["mean_val"] for i in ok]),
    )
    res["max_val"] = dict(
        default=float(np.mean([base[i]["max_val"] for i in ok])),
        agent=float(np.mean([ag[i]["max_val"] for i in ok])),
    )
    res["diversity"] = dict(
        default=float(np.mean([base[i]["diversity"] for i in ok])),
        agent=float(np.mean([ag[i]["diversity"] for i in ok])),
    )
    # pooled over all five runs
    res["pooled5"] = {}
    for nm, lst in (("default", base), ("agent", ag)):
        d = a.stack([(lst[i]["cell"], lst[i]["cols"]) for i in ok])
        r = a.nnls_ens(d["V"], d["yv"], d["T"], d["yt"], range(d["V"].shape[1]))
        res["pooled5"][nm] = [r[e] for e in EV]
    # LLM cost
    cost = []
    for r_ in R:
        f = f"{O22}/genomic_30k_agent/rep{r_}/agent_logs/summary.json"
        if os.path.exists(f):
            cost.append(json.load(open(f)))
    res["cost_summaries"] = cost
    DATA["agent"] = res
    d = res["diff"]["genomic"]
    print(
        f"[agent] pairs {len(ok)}: agent - default single-run ensemble genomic {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] wins {d['wins']}/{d['n']}; mean val {res['mean_val']['default']:.4f} -> {res['mean_val']['agent']:.4f}; pooled5 {res['pooled5']['default'][0]:.4f} vs {res['pooled5']['agent'][0]:.4f}; diversity {res['diversity']['default']:.3f} vs {res['diversity']['agent']:.3f}",
        flush=True,
    )
    if cost:
        print("[agent] cost keys:", list(cost[0].keys())[:12], flush=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for k in range(len(ok)):
        ax.plot([0, 1], [res["default"][k][0], res["agent"][k][0]], "o-", color=GREY, alpha=0.7)
    ax.plot(
        [0, 1],
        [np.mean([x[0] for x in res["default"]]), np.mean([x[0] for x in res["agent"]])],
        "s-",
        color=BLUE,
        lw=2.4,
        ms=7,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["default loop", "multi-agent loop"])
    ax.set_ylabel("single-run ensemble r (genomic test)")
    ax.set_title("Multi-agent loop pilot (same pools)")
    a.savefig(fig, "agent_pilot", out)


# -------------------------------------------------------------------------------------------------------------------- precision
def per_entry(d):
    lab = np.load(f"{d}/labels.npz")
    yt, yv = lab["oracle_genomic"].astype(float), lab["val_labels"].astype(float)
    out = {}
    for f in glob.glob(f"{d}/r*_menu_*_meta.json"):
        m = json.load(open(f))
        z = np.load(f.replace("_meta.json", ".npz"))
        tp, vp = z["test_pred_genomic"].astype(float), z["val_pred"].astype(float)
        if not np.isfinite(tp).all() or np.abs(tp).max() > 1e3:
            continue
        out[m["menu_entry"]] = dict(
            test=mj.pear(tp, yt),
            val=mj.pear(vp, yv),
            time=m.get("train_time_sec"),
            epochs=m.get("epochs_trained"),
            opt=(m.get("train_config") or {}).get("optimizer"),
        )
    return out


def an_precision(out):
    base = per_entry(f"{MR_}/ind30k_genomic_r1")
    res = {"fp32": dict(n=len(base))}
    bd = a.menu_matrix([f"{MR_}/ind30k_genomic_r1"])
    r0, _ = a.full_enet(bd)
    res["fp32"].update(
        ensemble=r0,
        mean_test=float(np.mean([v["test"] for v in base.values()])),
        mean_time=float(np.mean([v["time"] for v in base.values() if v["time"]]))
        if any(v["time"] for v in base.values())
        else None,
    )
    for mode in ("tf32", "bf16", "fp16"):
        d = f"{MR_}/prec30k_{mode}_genomic_r1"
        pe = per_entry(d)
        common = sorted(set(pe) & set(base))
        dt = np.array([pe[k]["test"] - base[k]["test"] for k in common])
        dv = np.array([pe[k]["val"] - base[k]["val"] for k in common])
        t_ratio = [
            base[k]["time"] / pe[k]["time"] for k in common if base[k]["time"] and pe[k]["time"]
        ]
        dm = a.menu_matrix([d])
        rr, _ = a.full_enet(dm)
        by_opt = {}
        for k in common:
            by_opt.setdefault(base[k]["opt"], []).append(pe[k]["test"] - base[k]["test"])
        res[mode] = dict(
            n=len(pe),
            n_common=len(common),
            mean_dtest=float(dt.mean()),
            se=float(dt.std(ddof=1) / np.sqrt(len(dt))),
            mean_dval=float(dv.mean()),
            ensemble=rr,
            ensemble_diff={e: rr[e] - r0[e] for e in EV},
            median_time_ratio=float(np.median(t_ratio)) if t_ratio else None,
            mean_time_ratio=float(np.mean(t_ratio)) if t_ratio else None,
            n_worse_than_minus_0p02=int((dt < -0.02).sum()),
            by_optimizer={k_: dict(n=len(v), mean=float(np.mean(v))) for k_, v in by_opt.items()},
            missing_entries=sorted(set(base) - set(pe)),
        )
        s = res[mode]
        print(
            f"[precision] {mode}: {s['n_common']} paired models, test r change {s['mean_dtest']:+.4f} (SE {s['se']:.4f}), val change {s['mean_dval']:+.4f}; ensemble genomic {rr['genomic']:.4f} vs fp32 {r0['genomic']:.4f}; time ratio fp32/mode median {s['median_time_ratio']}; by optimizer {s['by_optimizer']}; missing {s['missing_entries']}",
            flush=True,
        )
    sp = f"{os.path.dirname(a.__file__)}/../../outputs/justification_oct22/precision_speed.json"
    sp = "outputs/justification_oct22/precision_speed.json"
    if os.path.exists(sp):
        rows = json.load(open(sp))["rows"]
        res["speed"] = {}
        for m in ("tf32", "bf16", "fp16"):
            r_ = [
                x["fp32_2"]["sec"] / x[m]["sec"]
                for x in rows
                if x.get(m) and x.get("fp32_2") and x[m].get("ok")
            ]
            res["speed"][m] = dict(
                median=float(np.median(r_)), min=float(min(r_)), max=float(max(r_)), n=len(r_)
            )
        print("[precision] controlled speedup vs fp32:", res["speed"], flush=True)
    DATA["precision"] = res
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, mode in zip(axs, ("tf32", "bf16", "fp16")):
        pe = per_entry(f"{MR_}/prec30k_{mode}_genomic_r1")
        common = sorted(set(pe) & set(base))
        x, y = [base[k]["test"] for k in common], [pe[k]["test"] for k in common]
        ax.scatter(x, y, s=14, color={"tf32": TEAL, "bf16": BLUE, "fp16": RED}[mode], alpha=0.8)
        lo, hi = min(x + y) - 0.01, max(x + y) + 0.01
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
        ax.set_xlabel("fp32 test r")
        ax.set_ylabel(f"{mode} test r")
        ax.set_title(f"{mode}: {res[mode]['mean_dtest']:+.4f} per model")
    fig.suptitle(
        "Same 33 architectures, same pool: reduced precision against fp32 (genomic test r per model)",
        fontsize=10.5,
        y=1.02,
    )
    a.savefig(fig, "precision_ab", out)


# ----------------------------------------------------------------------------------------------------------------- validation variants
def an_valvar(out):
    """Parse the validation-variant analysis log (scripts/val_variants/analyze_val_variants.py) into data.json."""
    import re

    logs = sorted(glob.glob("outputs/val_variants/analysis_phase2/log_*.txt")) or sorted(
        glob.glob("outputs/val_variants/analysis_phase1/log_*.txt")
    )
    txt = open(logs[-1]).read()
    res = {"pools": {}, "contrasts": {}}
    for blk in re.split(r"===== pool ", txt)[1:]:
        pool = blk.split(":", 1)[0].strip()
        arms = {}
        for m in re.finditer(
            r"^\s+(V0c|V0x|V0|V1|V2|V3)\s+((?:r\d\d=[0-9.]+\(k=\d+/\d+\)[ \t]*)+)", blk, re.M
        ):
            arms[m.group(1)] = [float(x) for x in re.findall(r"r\d\d=([0-9.]+)", m.group(2))]
        sd = re.search(r"V0 seed SD of mean18 = ([0-9.]+)", blk)
        rank = {}
        for m in re.finditer(
            r"^\s+(\S+)\|on_V(\d)_minus_on_V0\s+mean18 delta Spearman = ([+-][0-9.]+)\s+95% CI \[([+-][0-9.]+), ([+-][0-9.]+)\]",
            blk,
            re.M,
        ):
            rank.setdefault(m.group(1), {})[f"V{m.group(2)}"] = [
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
            ]
        own = {
            m.group(1): float(m.group(2))
            for m in re.finditer(
                r"^\s+(V0c?x?|V1|V2|V3)\|own\s+mean18 Spearman = ([0-9.]+)", blk, re.M
            )
        }
        ceil = re.search(r"ceiling_test_seed0_vs_seed1\s+mean18 Spearman = ([0-9.]+)", blk)
        res["pools"][pool] = dict(
            arms=arms,
            v0_seed_sd=float(sd.group(1)) if sd else None,
            rank_delta=rank,
            own_spearman=own,
            noise_ceiling=float(ceil.group(1)) if ceil else None,
        )
    for m in re.finditer(r"^\s+(\S+) minus (\S+): (V0=.*)$", txt, re.M):
        res["contrasts"][f"{m.group(1)} minus {m.group(2)}"] = {
            k: float(v) for k, v in re.findall(r"(V0c?x?|V1|V2|V3)=([+-][0-9.]+)", m.group(3))
        }
    DATA["valvar"] = res
    for pool, v in res["pools"].items():
        print(
            f"[valvar] {pool}: "
            + ", ".join(f"{k} {np.mean(x):.4f}" for k, x in v["arms"].items())
            + f" (V0 seed SD {v['v0_seed_sd']})",
            flush=True,
        )
    print("[valvar] contrasts:", res["contrasts"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--only", default="prompts,agent,precision,valvar")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prev = (
        json.load(open(f"{args.out}/data.json")) if os.path.exists(f"{args.out}/data.json") else {}
    )
    for step, fn in (
        ("valvar", an_valvar),
        ("precision", an_precision),
        ("agent", an_agent),
        ("prompts", an_prompts),
    ):
        if step in args.only.split(","):
            try:
                fn(args.out)
            except Exception as ex:
                import traceback

                traceback.print_exc()
                print(f"[{step}] FAILED: {ex}", flush=True)
            prev.update(DATA)
            json.dump(prev, open(f"{args.out}/data.json", "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
