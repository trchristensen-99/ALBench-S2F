#!/usr/bin/env python
"""Does the AutoResearch loop explore early and exploit late, and does what it optimises (V0c validation r) carry over to the other evaluation sets?

Reads `data_loop_models.json` (scripts/analysis/extract_loop_models.py). Three readouts, each paired with the question that motivated it:

  A. Proposals and picks by round. The six proposals, the two the coordinator picked and the models that trained are described against what is already trained in the same run:
       text novelty      1 - max TF-IDF cosine of idea+design to an earlier trained model            (proposals, picks)
       incumbent cosine  cosine of idea+design to the best-validation model so far                   (proposals, picks)
       new-feature rate  share containing an architectural feature (the loop's own taxonomy, `agent_loop._FEATURES`) no trained model has yet  (picks from the prose, trained models from the source)
     plus, for trained models, the share whose nearest earlier model (token 3-gram Jaccard of the code, numerals blanked so width and depth changes do not count) is one of the three best so far
     against the share expected by chance, the share of LegNet-derived designs, and the share with a training recipe not used earlier in the run.
     "Early" is the first round(0.4 * rounds) rounds, the manager schedule's explore phase; runs without a manager are split at the same fraction so the curves are comparable.
     Per-run means first, then mean and SE over runs, because rounds of one run are not independent.
  B. What the search buys on the other sets: rank agreement between validation r and r on each of the 18 test sets across the models of a run (all models and the ten best), the regret of
     taking the validation winner, and how much of the gain between the early and final best-by-validation model and between the stack after ~6 rounds and the final stack shows up on each set.
  C. Selection optimism: validation r against genomic test r (chr7/13), whether the validation winners sit above the line fitted on the other models.

    python scripts/analysis/analyze_loop_proposals.py [--data outputs/justification_oct22/data_loop_models.json] [--out outputs/justification_oct22] [--valvar outputs/val_variants/analysis_phase2/results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

# the loop's own regexes (albench/arch_search/agent_loop.py, _FEATURES) applied to trained models' source
SRC_FEATURES = {
    "squeeze-excitation": r"class\s+\w*(SE|Squeeze)\w*|SELayer|Squeeze",
    "self-attention": r"MultiheadAttention|TransformerEncoder|nn\.Transformer|scaled_dot_product_attention",
    "recurrent": r"nn\.(LSTM|GRU|RNN)\b",
    "state-space": r"[Mm]amba|[Ss]tate.?[Ss]pace|selective.?scan",
    "U-Net-like": r"ConvTranspose|F\.interpolate|nn\.Upsample",
    "dilated conv": r"dilation\s*=\s*(?!1\b)\d+",
    "attention pooling": r"AttnPool|AttentionPool|[Aa]ttn[_A-Za-z]*\s*=",
    "concat skips": r"torch\.(cat|concat)\(",
    "layer/group norm": r"nn\.(LayerNorm|GroupNorm|InstanceNorm1d|RMSNorm)",
    "depthwise/grouped conv": r"groups\s*=\s*(?!1\b)",
    "gating": r"F\.glu|nn\.GLU|\*\s*torch\.sigmoid|\*\s*F\.sigmoid|\.sigmoid\(\)\s*\*",
    "top-k pooling": r"\.topk\(|torch\.topk",
    "flatten readout": r"nn\.Flatten|\.flatten\(1\)|\.view\([^)]*-1\)|\.reshape\([^)]*-1\)",
    "global max pool": r"AdaptiveMaxPool1d|amax\(|\.max\(dim|\.max\(-1\)|adaptive_max_pool1d",
    "windowed/MIL": r"\.unfold\(|windows",
    "multi-branch": r"ModuleList",
    "second-order": r"einsum|bmm|torch\.matmul",
}
# the same taxonomy read from the proposal's prose, for ideas that were never coded (skipped by the coordinator)
TXT_FEATURES = {
    "squeeze-excitation": r"squeeze|excitation|\bSE\b",
    "self-attention": r"self-?attention|transformer|multi-?head|\bMHA\b|attention (layer|block)|local[- ]window attention|relative-position attention",
    "recurrent": r"\bLSTM|\bGRU|\bRNN|recurrent",
    "state-space": r"mamba|state.?space|\bSSM\b|\bS4\b",
    "U-Net-like": r"u-?net|encoder.?decoder|upsampl|transposed conv",
    "dilated conv": r"dilated|dilation",
    "attention pooling": r"attention.?pool|attn.?pool|learned query|attention-weighted|attentive pool",
    "concat skips": r"dense(ly)? (connect|block)|densenet|concat",
    "layer/group norm": r"layer ?norm|group ?norm|rmsnorm|instance ?norm",
    "depthwise/grouped conv": r"depthwise|grouped conv|separable|shufflenet|resnext",
    "gating": r"\bglu\b|gated|gating|sigmoid gate",
    "top-k pooling": r"top-?k",
    "flatten readout": r"flatten|position-?resolved|per-?position (readout|head)",
    "global max pool": r"global max|max-?pool(ing)? over (all )?positions",
    "windowed/MIL": r"windowed|multiple.?instance|unfold|sliding window",
    "multi-branch": r"multi-?kernel|parallel (branch|conv)|inception|multi-?scale|multi-?branch",
    "second-order": r"bilinear|outer.?product|second-?order|einsum|pairwise|factori[sz]ation|volterra",
}
# the coordinator's coverage table is built from SRC_FEATURES["self-attention"], which misses attention written by hand (softmax over a matmul/einsum score)
CUSTOM_ATTENTION = (r"softmax", r"einsum|bmm|torch\.matmul|\s@\s")
# the classes the explore directive asks the coordinator to keep covered (agent_loop._ARCH_CLASSES), the rest is plain convolution
CLASSES = (
    "self-attention",
    "recurrent",
    "state-space",
    "U-Net-like",
    "dilated conv",
    "attention pooling",
)
LEGNET = r"legnet|effblock|reference (backbone|mpra)"
PHASE_FRAC = 0.4


def src_feats(src: str) -> frozenset[str]:
    f = {k for k, pat in SRC_FEATURES.items() if re.search(pat, src)}
    if all(re.search(p, src) for p in CUSTOM_ATTENTION):
        f.add("self-attention")
    return frozenset(f)


def txt_feats(text: str) -> frozenset[str]:
    return frozenset(k for k, pat in TXT_FEATURES.items() if re.search(pat, text, flags=re.I))


def classes_of(f: frozenset[str]) -> frozenset[str]:
    return frozenset(c for c in CLASSES if c in f) or frozenset({"plain conv"})


def shingles(src: str, n: int = 3) -> set[int]:
    """Token n-grams of the code with comments, docstrings and numbers removed: width, depth and learning-rate edits then leave the set unchanged."""
    src = re.sub(r"#.*", "", src)
    src = re.sub(r'""".*?"""', "", src, flags=re.S)
    src = re.sub(r"\b\d+(\.\d+)?(e-?\d+)?\b", "N", src)
    toks = re.findall(r"\w+|[^\s\w]", src)
    return {hash(tuple(toks[i : i + n])) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cosines(vec, i: int, js: list[int]) -> np.ndarray:
    """Cosine of text i to each text in js (rows are L2-normalised TF-IDF)."""
    return (vec[js] @ vec[i].T).toarray().ravel()


def recipe_key(tc) -> str:
    """Optimizer, lr, weight decay and batch size of a training config (epochs and schedule excluded: they are not what a recipe 'bet' changes)."""
    try:
        c = tc if isinstance(tc, dict) else json.loads(str(tc).replace("'", '"'))
        return f"{c.get('optimizer')}|{c.get('lr')}|{c.get('weight_decay')}|{c.get('batch_size')}"
    except Exception:
        return str(tc)


def mean(x) -> float:
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def sem(x) -> float:
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def group_of(run: dict) -> str:
    m = {
        "schedule_ens": "manager schedule",
        "schedule_fb": "schedule + reference check",
        "quota": "quota manager",
    }.get(run["config"].get("manager"), "no manager")
    return m if run["standard_input"] or m != "no manager" else "no manager (older harness)"


def n_rounds_of(run: dict) -> int:
    return run["config"].get("rounds") or (max(p["round"] for p in run["plans"]) + 1)


def phase_of(rd: int, n_rounds: int) -> str:
    return "early" if rd < round(n_rounds * PHASE_FRAC) else "late"


def round_records(run: dict, vec, idx: dict) -> list[dict]:
    """One record per round of one run: proposals, picks and trained models against everything trained in earlier rounds."""
    nr = n_rounds_of(run)
    models = [m for m in run["models"] if m["val_r"] is not None]
    out, prev = [], []
    seen_recipes: set[str] = set()
    for pl in sorted(run["plans"], key=lambda p: p["round"]):
        rd = pl["round"]
        if not pl["items"]:
            continue
        rec = {"round": rd, "x": rd / max(1, nr - 1), "phase": phase_of(rd, nr), "n_rounds": nr}
        prev_idx = [idx[(run["run"], m["model_id"])] for m in prev]
        prev_txt_feats = (
            frozenset().union(*[txt_feats(m["idea"] + " " + m["design"]) for m in prev])
            if prev
            else frozenset()
        )
        best = max(prev, key=lambda m: m["val_r"]) if prev else None
        rows = []
        for it in pl["items"]:
            ti = idx[(run["run"], f"r{rd}_s{it['slot']}")]
            text = it["idea"] + " " + it["design"]
            f = txt_feats(text)
            rows.append(
                {
                    "pick": not it["skipped"] and it["status"] not in (None, "skipped"),
                    "nov": 1 - float(cosines(vec, ti, prev_idx).max()) if prev else np.nan,
                    "inc": float(cosines(vec, ti, [idx[(run["run"], best["model_id"])]])[0])
                    if best
                    else np.nan,
                    "classes": classes_of(f),
                    "new_feat": float(bool(f - prev_txt_feats)) if prev else np.nan,
                    "legnet": float(bool(re.search(LEGNET, text, flags=re.I))),
                }
            )
        for tag, rr in (("prop", rows), ("pick", [r for r in rows if r["pick"]])):
            rec[f"{tag}_nov"] = mean(r["nov"] for r in rr)
            rec[f"{tag}_inc"] = mean(r["inc"] for r in rr)
            rec[f"{tag}_new_feat"] = mean(r["new_feat"] for r in rr)
            rec[f"{tag}_legnet"] = mean(r["legnet"] for r in rr)
            rec[f"{tag}_nclass"] = (
                len(frozenset().union(*[r["classes"] for r in rr])) if rr else np.nan
            )
        tr = [m for m in models if m["round"] == rd]
        if prev and tr:
            sh_prev = [shingles(m["source"]) for m in prev]
            f_prev = frozenset().union(*[src_feats(m["source"]) for m in prev])
            top3 = {m["model_id"] for m in sorted(prev, key=lambda m: -m["val_r"])[:3]}
            par, new_f, new_r = [], [], []
            for m in tr:
                sh = shingles(m["source"])
                par.append(
                    float(
                        prev[int(np.argmax([jaccard(sh, s) for s in sh_prev]))]["model_id"] in top3
                    )
                )
                new_f.append(float(bool(src_feats(m["source"]) - f_prev)))
                new_r.append(float(recipe_key(m["train_config"]) not in seen_recipes))
            rec.update(
                parent_top3=mean(par),
                parent_chance=min(3, len(prev)) / len(prev),
                trained_new_feat=mean(new_f),
                trained_new_recipe=mean(new_r),
            )
        if tr:
            rec["val_mean"] = mean(m["val_r"] for m in tr)
            rec["new_best"] = float(max(m["val_r"] for m in tr) > best["val_r"]) if best else np.nan
        seen_recipes |= {recipe_key(m["train_config"]) for m in tr}
        out.append(rec)
        prev += tr
    return out


KEYS = (
    "prop_nov", "pick_nov", "prop_inc", "pick_inc", "prop_nclass", "pick_nclass", "prop_new_feat", "pick_new_feat",
    "prop_legnet", "pick_legnet", "trained_new_feat", "trained_new_recipe", "parent_top3", "parent_chance", "val_mean", "new_best",
)  # fmt: skip


def table_a(runs: list[dict], vec, idx: dict) -> dict:
    by_phase: dict = defaultdict(list)
    curves: dict = defaultdict(list)
    for run in runs:
        nr = n_rounds_of(run)
        recs = round_records(run, vec, idx)
        if len(recs) < round(nr * PHASE_FRAC) + 2:
            continue  # the late phase has not started (runs still going)
        g = group_of(run)
        for ph in ("early", "late"):
            sub = [r for r in recs if r["phase"] == ph]
            by_phase[(g, ph)].append({k: mean(r.get(k) for r in sub) for k in KEYS})
        curves[g] += recs
    res = {"phase": {}, "curves": {}}
    for (g, ph), runs_ in sorted(by_phase.items()):
        res["phase"][f"{g}|{ph}"] = {
            "n_runs": len(runs_),
            **{k: [mean(r[k] for r in runs_), sem(r[k] for r in runs_)] for k in KEYS},
        }
    for g, recs in curves.items():
        rounds = sorted({r["round"] for r in recs})
        res["curves"][g] = [
            {
                "round": rd,
                "n_runs": sum(r["round"] == rd for r in recs),
                **{k: mean(r.get(k) for r in recs if r["round"] == rd) for k in KEYS},
            }
            for rd in rounds
        ]
    return res


def table_b(runs: list[dict], sets: list[str]) -> dict:
    sets = [*sets, "snv_effect"]
    per_run = []
    for run in runs:
        ms = [m for m in run["models"] if m.get("finite") and m.get("r") and m["val_r"] is not None]
        if len(ms) < 10:
            continue
        val = np.array([m["val_r"] for m in ms])
        top10 = sorted(ms, key=lambda m: -m["val_r"])[:10]
        best = top10[0]
        early = [m for m in ms if m["round"] <= 2]
        first = max(early, key=lambda m: m["val_r"]) if early else None
        st = run.get("stack", {})
        cum = st.get("cum", {})
        k_mid = min(cum, key=lambda k: abs(int(k) - 6)) if cum else None
        per_run.append(
            {
                "rho": {s: float(spearmanr(val, [m["r"][s] for m in ms])[0]) for s in sets},
                "rho_top10": {
                    s: float(spearmanr([m["val_r"] for m in top10], [m["r"][s] for m in top10])[0])
                    for s in sets
                },
                "regret": {s: max(m["r"][s] for m in ms) - best["r"][s] for s in sets},
                "best_gain": {s: best["r"][s] - first["r"][s] for s in sets} if first else {},
                "stack_gain_late": {
                    s: st["all"][s] - cum[k_mid][s]
                    for s in sets
                    if k_mid and s in st.get("all", {}) and s in cum[k_mid]
                },
                "stack_all": st.get("all", {}),
            }
        )
    agg = {}
    for s in sets:
        agg[s] = {
            "rho": [mean(r["rho"][s] for r in per_run), sem(r["rho"][s] for r in per_run)],
            "rho_top10": [
                mean(r["rho_top10"][s] for r in per_run),
                sem(r["rho_top10"][s] for r in per_run),
            ],
            "regret": mean(r["regret"][s] for r in per_run),
            "best_gain": [
                mean(r["best_gain"].get(s) for r in per_run),
                sem(r["best_gain"].get(s) for r in per_run),
            ],
            "stack_gain_late": [
                mean(r["stack_gain_late"].get(s) for r in per_run),
                sem(r["stack_gain_late"].get(s) for r in per_run),
            ],
            "stack_all": mean(r["stack_all"].get(s) for r in per_run),
        }
    return {"n_runs": len(per_run), "per_set": agg}


def table_c(runs: list[dict]) -> dict:
    """Winner's-curse check: fit test r on validation r over all models except the three validation winners, and ask where the winners sit relative to that line.
    The same residual is taken for early- and late-round models, since a model trained late has been steered by more validation feedback."""
    gap, win, early_res, late_res = [], [], [], []
    for run in runs:
        ms = [
            m
            for m in run["models"]
            if m["val_r"] is not None and m["test_genomic_r"] is not None and m["val_r"] > 0.3
        ]
        if len(ms) < 15:
            continue
        order = sorted(range(len(ms)), key=lambda i: -ms[i]["val_r"])
        top = set(order[:3])
        rest = [i for i in range(len(ms)) if i not in top]
        b, a = np.polyfit(
            [ms[i]["val_r"] for i in rest], [ms[i]["test_genomic_r"] for i in rest], 1
        )

        def resid(i: int, a=a, b=b, ms=ms) -> float:
            return ms[i]["test_genomic_r"] - (a + b * ms[i]["val_r"])

        nr = n_rounds_of(run)
        gap.append(float(np.mean([m["val_r"] - m["test_genomic_r"] for m in ms])))
        win.append(float(np.mean([resid(i) for i in top])))
        early_res.append(mean(resid(i) for i in rest if phase_of(ms[i]["round"], nr) == "early"))
        late_res.append(mean(resid(i) for i in rest if phase_of(ms[i]["round"], nr) == "late"))
    return {
        "n_runs": len(gap),
        "val_minus_test": mean(gap),
        "winners_vs_line": [mean(win), sem(win)],
        "early_models_vs_line": [mean(early_res), sem(early_res)],
        "late_models_vs_line": [mean(late_res), sem(late_res)],
    }


def table_d(runs: list[dict]) -> dict:
    """Who carries the final stack: weight share of early-phase against late-phase models, concentration, and the architectural features of the weighted members."""
    phase_w, phase_n, eff, top1 = defaultdict(list), defaultdict(list), [], []
    feat_models, feat_weight = defaultdict(list), defaultdict(list)
    for run in runs:
        w = run.get("stack", {}).get("all", {}).get("weights")
        if not w or n_rounds_of(run) < 10:
            continue
        nr = n_rounds_of(run)
        ms = {m["model_id"]: m for m in run["models"] if m["model_id"] in w}
        tot = sum(w.values())
        if tot <= 0 or len(ms) < 10:
            continue
        for ph in ("early", "late"):
            ids = [i for i, m in ms.items() if phase_of(m["round"], nr) == ph]
            phase_w[ph].append(sum(w[i] for i in ids) / tot)
            phase_n[ph].append(len(ids) / len(ms))
        ws = np.array([w[i] / tot for i in ms])
        eff.append(float(1 / np.sum(ws**2)))
        top1.append(float(ws.max()))
        for f in SRC_FEATURES:
            has = [f in src_feats(m["source"]) for m in ms.values()]
            feat_models[f].append(float(np.mean(has)))
            feat_weight[f].append(float(sum(w[i] / tot for i, h in zip(ms, has) if h)))
    return {
        "n_runs": len(eff),
        "weight_share": {ph: [mean(v), sem(v)] for ph, v in phase_w.items()},
        "model_share": {ph: [mean(v), sem(v)] for ph, v in phase_n.items()},
        "effective_members": [mean(eff), sem(eff)],
        "top_member_weight": [mean(top1), sem(top1)],
        "features": {
            f: {"model_share": mean(feat_models[f]), "weight_share": mean(feat_weight[f])}
            for f in SRC_FEATURES
        },
    }


def table_features(runs: list[dict]) -> dict:
    """Share of trained models containing each feature, in early and late rounds, and the share of runs that ever trained one."""
    early, late, ever = defaultdict(list), defaultdict(list), defaultdict(list)
    for run in runs:
        nr = n_rounds_of(run)
        ms = [m for m in run["models"] if m["val_r"] is not None]
        if nr < 10 or len(ms) < 10:
            continue
        fs = [(phase_of(m["round"], nr), src_feats(m["source"])) for m in ms]
        for f in SRC_FEATURES:
            e = [f in x for ph, x in fs if ph == "early"]
            lt = [f in x for ph, x in fs if ph == "late"]
            if e and lt:
                early[f].append(float(np.mean(e)))
                late[f].append(float(np.mean(lt)))
            ever[f].append(float(any(f in x for _, x in fs)))
    return {
        f: {"early": mean(early[f]), "late": mean(late[f]), "runs_ever": mean(ever[f])}
        for f in SRC_FEATURES
    }


def table_valvar(path: str) -> dict:
    """The validation-set test of Oct 25-26: ensembles fitted and early-stopped on V0c (genomic chr19/21/X) against V1 (perturbed genomic), V2 (diverse reservoirs) and V3 (equal thirds)."""
    r = json.load(open(path))
    sets = [
        s for s in r["genomic"]["ensemble"]["V3"]["0"] if not s.startswith("_") and s != "snv_delta"
    ]
    out = {}
    for arm in ("V1", "V2", "V3"):
        rows = [
            {s: r[p]["ensemble"][arm][k][s] - r[p]["ensemble"]["V0c"][k][s] for s in sets}
            for p in ("genomic", "random", "motif_ct_enriched")
            for k in r[p]["ensemble"][arm]
            if k in r[p]["ensemble"]["V0c"]
        ]
        m18 = [float(np.mean(list(x.values()))) for x in rows]
        out[arm] = {
            "n_pool_seeds": len(rows),
            "mean18": [float(np.mean(m18)), sem(m18)],
            "genomic": float(np.mean([x["genomic"] for x in rows])),
            "ood": [float(np.mean([x["ood"] for x in rows])), sem([x["ood"] for x in rows])],
            "ood_positive": int(sum(x["ood"] > 0 for x in rows)),
            "random": float(np.mean([x["random_32k"] for x in rows])),
        }
    out["noise_sd_ood_v0_seeds"] = r["genomic"]["noise_sd_V0_seeds"]["ood"]
    return out


def stack_growth(runs: list[dict]) -> dict:
    """Mean stack r after k rounds and the increment per two rounds, over complete 15-round managed runs (explore = rounds 0-5)."""
    full = [r for r in runs if "stack" in r and n_rounds_of(r) == 15 and len(r["plans"]) >= 15]
    ks = ["2", "4", "6", "8", "10", "12", "all"]
    out = {"n_runs": len(full), "sets": {}}
    for s in ("genomic", "ood", "random_32k"):
        at = {
            k: [
                (r["stack"]["all"] if k == "all" else r["stack"]["cum"].get(k, {})).get(s)
                for r in full
            ]
            for k in ks
        }
        inc = {}
        for a, b in zip(ks[:-1], ks[1:]):
            v = [y - x for x, y in zip(at[a], at[b]) if x is not None and y is not None]
            inc[f"{a}-{b}"] = [mean(v), sem(v)]
        out["sets"][s] = {"at": {k: mean(v) for k, v in at.items()}, "increment": inc}
    return out


def make_figures(res: dict, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = res["A"]["curves"]
    ink, blue, orange, grey = "#222222", "#2b6cb0", "#c05621", "#8a8a8a"
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    grp = curves.get("manager schedule", [])
    base = curves.get("no manager", [])

    def line(a, rows, key, color, label, ls="-"):
        pts = [
            (r["round"], r[key])
            for r in rows
            if r.get(key) is not None and np.isfinite(r[key]) and r["n_runs"] >= 6
        ]
        if pts:
            a.plot(*zip(*pts), ls, color=color, lw=2, marker="o", ms=3.5, label=label)

    def shade(a):
        a.axvspan(-0.5, 5.5, color="#e9eef5", zorder=0)
        a.set_xlim(-0.5, 14.5)
        a.set_xlabel("round (shaded: explore phase of the schedule)")
        a.spines[["top", "right"]].set_visible(False)

    a = ax[0, 0]
    line(a, grp, "prop_nov", blue, "6 proposals")
    line(a, grp, "pick_nov", orange, "2 picks")
    line(a, base, "pick_nov", grey, "picks, no manager", "--")
    a.set_ylabel("novelty vs trained (1 - max cosine)")
    a.set_title("Ideas stay different early, then narrow")
    a.legend(frameon=False, fontsize=8)
    shade(a)
    a = ax[0, 1]
    line(a, grp, "pick_new_feat", blue, "picks with an untried feature")
    line(a, grp, "trained_new_feat", orange, "trained (from source)")
    line(a, base, "pick_new_feat", grey, "picks, no manager", "--")
    a.set_ylabel("share of the round")
    a.set_title("New mechanisms run out after round 3")
    a.legend(frameon=False, fontsize=8)
    shade(a)
    a = ax[0, 2]
    line(a, grp, "trained_new_recipe", blue, "new recipe (opt/lr/wd/batch)")
    line(a, grp, "pick_legnet", orange, "picks derived from LegNet")
    line(a, base, "pick_legnet", grey, "same, no manager", "--")
    a.set_ylabel("share of the round")
    a.set_title("Recipes explored early, LegNet family late")
    a.legend(frameon=False, fontsize=8)
    shade(a)
    a = ax[1, 0]
    line(a, grp, "parent_top3", blue, "nearest earlier model is a top-3")
    line(a, grp, "parent_chance", grey, "expected by chance", ":")
    line(a, base, "parent_top3", orange, "no manager", "--")
    a.set_ylabel("share of trained models")
    a.set_title("Late models descend from the winners")
    a.legend(frameon=False, fontsize=8)
    shade(a)
    a = ax[1, 1]
    line(a, grp, "new_best", blue, "new best validation r")
    line(a, base, "new_best", grey, "no manager", "--")
    a.set_ylabel("share of runs with a new best this round")
    a.set_title("Exploit pays for ~3 rounds")
    a.legend(frameon=False, fontsize=8)
    shade(a)
    a = ax[1, 2]
    g = res["stack_growth"]
    labels = ["2-4", "4-6", "6-8", "8-10", "10-12", "12-15"]
    w = 0.27
    for j, (s, c, nm) in enumerate(
        (("genomic", ink, "genomic"), ("ood", orange, "OOD"), ("random_32k", blue, "random"))
    ):
        inc = g["sets"][s]["increment"]
        keys = ["2-4", "4-6", "6-8", "8-10", "10-12", "12-all"]
        a.bar(
            np.arange(6) + (j - 1) * w,
            [inc[k][0] for k in keys],
            w,
            yerr=[inc[k][1] for k in keys],
            color=c,
            label=nm,
            error_kw={"lw": 0.8},
        )
    a.set_xticks(range(6), labels)
    a.set_xlabel("rounds added to the stack")
    a.set_ylabel("stack test r gained")
    a.set_title(f"Stack gain per round pair ({g['n_runs']} full runs)")
    a.axvline(1.5, color=grey, lw=0.8, ls=":")
    a.legend(frameon=False, fontsize=8)
    a.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{out}/fig_loop_explore_exploit.png", dpi=180)
    plt.close(fig)

    B = res["B"]["per_set"]
    names = [
        "genomic",
        "random_32k",
        "ood",
        "snv_effect",
        "sub_high",
        "ins_high",
        "del_high",
        "dinuc_shuffle",
        "gc_sweep",
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    x = np.arange(len(names))
    ax[0].bar(
        x - 0.2, [B[n]["rho"][0] for n in names], 0.4, color=blue, label="all models of a run"
    )
    ax[0].bar(
        x + 0.2,
        [B[n]["rho_top10"][0] for n in names],
        0.4,
        color=orange,
        label="the 10 best by validation",
    )
    ax[0].set_xticks(x, names, rotation=35, ha="right")
    ax[0].set_ylabel("Spearman(validation r, test r)")
    ax[0].set_title("Validation r ranks the models on every set except OOD and GC among the best")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    ax[1].bar(
        x - 0.2,
        [B[n]["best_gain"][0] for n in names],
        0.4,
        color=blue,
        yerr=[B[n]["best_gain"][1] for n in names],
        label="best-by-validation model, rounds 0-2 to end",
        error_kw={"lw": 0.8},
    )
    ax[1].bar(
        x + 0.2,
        [B[n]["stack_gain_late"][0] for n in names],
        0.4,
        color=orange,
        yerr=[B[n]["stack_gain_late"][1] for n in names],
        label="stack, round ~6 to end",
        error_kw={"lw": 0.8},
    )
    ax[1].set_xticks(x, names, rotation=35, ha="right")
    ax[1].set_ylabel("test r gained")
    ax[1].set_title("What the search gained, per evaluation set (no stack bar for the SNV effect)")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{out}/fig_loop_other_sets.png", dpi=180)
    plt.close(fig)


def print_report(res: dict) -> None:
    cols = (
        "prop_nov pick_nov prop_inc pick_inc prop_nclass prop_new_feat pick_new_feat prop_legnet pick_legnet "
        "trained_new_feat trained_new_recipe parent_top3 parent_chance val_mean new_best"
    ).split()
    print(
        "A. phase means over runs (each cell: mean over runs of the run's mean over the phase's rounds)"
    )
    print(f"{'group|phase':44s} {'runs':>4s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
    for k, v in res["A"]["phase"].items():
        print(f"{k:44s} {v['n_runs']:4d} " + " ".join(f"{v[c][0]:11.3f}" for c in cols))
    B = res["B"]
    print(f"\nB. other eval sets ({B['n_runs']} standard-input runs)")
    print(
        f"{'set':14s} {'rho all':>11s} {'rho top10':>11s} {'regret':>8s} {'best-model gain':>17s} {'stack gain':>17s} {'stack r':>8s}"
    )
    for s, v in B["per_set"].items():
        print(
            f"{s:14s} {v['rho'][0]:5.2f}±{v['rho'][1]:.2f} {v['rho_top10'][0]:5.2f}±{v['rho_top10'][1]:.2f} {v['regret']:8.4f} "
            f"{v['best_gain'][0]:+8.4f}±{v['best_gain'][1]:.4f} {v['stack_gain_late'][0]:+8.4f}±{v['stack_gain_late'][1]:.4f} {v['stack_all']:8.4f}"
        )
    print("\nC. selection optimism:", json.dumps(res["C"]))
    D = res["D"]
    print(
        f"\nD. final stack of the managed runs ({D['n_runs']} runs): weight share early {D['weight_share']['early'][0]:.2f} vs model share {D['model_share']['early'][0]:.2f}; "
        f"late {D['weight_share']['late'][0]:.2f} vs {D['model_share']['late'][0]:.2f}; effective members {D['effective_members'][0]:.1f}, top member {D['top_member_weight'][0]:.2f}"
    )
    print(
        f"{'feature':26s} {'early':>6s} {'late':>6s} {'runs ever':>9s} {'model share':>11s} {'weight share':>12s}"
    )
    for f, v in res["features"].items():
        print(
            f"{f:26s} {v['early']:6.2f} {v['late']:6.2f} {v['runs_ever']:9.2f} {D['features'][f]['model_share']:11.2f} {D['features'][f]['weight_share']:12.2f}"
        )
    if "valvar" in res:
        print(
            "\nValidation variants (ensemble test r, arm minus V0c, paired over pool x seed):",
            json.dumps(res["valvar"]),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/justification_oct22/data_loop_models.json")
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--valvar", default="outputs/val_variants/analysis_phase2/results.json")
    a = ap.parse_args()
    d = json.load(open(a.data))
    runs = [r for r in d["runs"] if len(r["plans"]) >= 5 and len(r["models"]) >= 10]
    print(
        f"{len(runs)} runs; groups:",
        {g: sum(group_of(r) == g for r in runs) for g in sorted({group_of(r) for r in runs})},
    )
    # one TF-IDF space over every proposal and every trained design, so cosines are comparable across runs
    texts, idx = [], {}
    for run in runs:
        for pl in run["plans"]:
            for it in pl["items"]:
                idx[(run["run"], f"r{pl['round']}_s{it['slot']}")] = len(texts)
                texts.append(it["idea"] + " " + it["design"])
        for m in run["models"]:
            idx[(run["run"], m["model_id"])] = len(texts)
            texts.append(m["idea"] + " " + m["design"])
    vec = (
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        .fit_transform(texts)
        .tocsr()
    )
    std = [r for r in runs if r["standard_input"]]
    managed = [r for r in std if group_of(r) in ("manager schedule", "schedule + reference check")]
    res = {
        "A": table_a(runs, vec, idx),
        "B": table_b(std, d["sets"]),
        "C": table_c(std),
        "D": table_d(managed),
        "features": table_features(managed),
        "stack_growth": stack_growth(managed),
    }
    if os.path.exists(a.valvar):
        res["valvar"] = table_valvar(a.valvar)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_loop_proposals.json", "w"), indent=1)
    make_figures(res, a.out)
    print_report(res)


if __name__ == "__main__":
    main()
