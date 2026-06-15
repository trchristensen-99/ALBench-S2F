"""Rank the Phase-0 LLM prompt-screen variants and recommend the top-3 for the
Phase-1 strategy bake-off.

The screen (`scripts/submit_llm_prompt_screen.py` → outputs/hp_llm_screen_e100) runs a
WIDE matrix of LLM proposer variants {style × model × novel-axes} at one cheap cell
(genomic D=30k × 2 seeds). This script reads every variant's trained-model trajectory
and ranks variants on the plan's axis — ORACLE-Pearson vs GPU-SECONDS — then picks a
TOP-3 with a DIVERSITY spread (distinct prompt styles, not 3 near-identical variants),
so the LLM family enters the bake-off represented but not over-ticketed.

For each (cell × seed × variant) dir we build the CUMULATIVE MEAN-ENSEMBLE curve in
trajectory order (the deploy mode): ensemble val-Pearson and ensemble oracle-test-Pearson
vs cumulative GPU-seconds (Σ train_time_sec). Per variant we then summarize across seeds:
  - final ensemble oracle-Pearson (what the variant's whole pool achieves),
  - ensemble oracle-Pearson at a COMMON GPU-second budget B (B = the min total GPU-seconds
    across all variant×seed runs present — the fair apples-to-apples efficiency point),
  - GPU-seconds to reach 95% of the variant's own final ensemble oracle-Pearson (speed).
Ranking is by oracle@B (efficiency) with final-oracle as the tiebreak. The top-3 is then
chosen greedily under a distinct-style constraint so the carried variants spread.

oracle test = per_set_metrics 'genomic' (chr-split ag_s2_chrsplit_v1); val = chr_val.

Read-heavy (loads r*.npz preds) — run on a compute node, not the bamdev4 login node.

Usage:
  python scripts/analysis/rank_llm_screen.py
  python scripts/analysis/rank_llm_screen.py --out_root outputs/hp_llm_screen_e100 --budget_frac 1.0
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

# variant label = llm_<style>_<modelshort>_nv<novel>  (submit_llm_prompt_screen.py)
_VARIANT_RE = re.compile(r"^llm_(?P<style>[a-z]+)_(?P<model>opus|sonnet|haiku)_nv(?P<novel>\d)$")
# Map the short model tag back to the full id the bake-off launcher needs.
_MODEL_FULL = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def _r(pred: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(pearsonr(pred[m], y[m])[0])


def variant_curve(vdir: Path, cache_dir: Path) -> dict | None:
    """Cumulative mean-ensemble val + oracle-Pearson vs cumulative GPU-seconds for one
    (cell × seed × variant) dir, in trajectory order. Cached per-path so reruns are fast."""
    tag = "__".join(vdir.parts[-3:])
    cache = cache_dir / f"{tag}.json"
    # Cache is keyed on the on-disk meta count: a variant that has trained MORE models
    # since the cache was written (e.g. ranked mid-run, now complete) must be recomputed,
    # else the ranking silently reflects a stale partial trajectory.
    n_meta_now = len(list(vdir.glob("r*_meta.json")))
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("n_meta") == n_meta_now:
            return cached

    lab = vdir / "labels.npz"
    if not lab.exists():
        return None
    lz = np.load(lab)
    val_y, test_oracle = lz["val_labels"], lz["test_oracle"]

    # Order models along the SEARCH TRAJECTORY: (round, within-round index).
    items: list[tuple[int, int, Path, dict]] = []
    for m in sorted(vdir.glob("r*_meta.json")):
        try:
            meta = json.loads(m.read_text())
        except Exception:
            continue
        if meta.get("error") or "val_pearson" not in meta:
            continue
        npz = m.with_name(m.name.replace("_meta.json", ".npz"))
        if not npz.exists():
            continue
        idx_match = re.search(r"_(\d+)$", meta.get("model_id", m.stem))
        items.append(
            (int(meta.get("round", 0)), int(idx_match.group(1)) if idx_match else 0, npz, meta)
        )
    items.sort(key=lambda t: (t[0], t[1]))
    if len(items) < 2:
        return None

    cum_gpu = 0.0
    val_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    steps: list[dict] = []
    for _, _, npz, meta in items:
        d = np.load(npz)
        val_preds.append(d["val_pred"])
        test_preds.append(d["test_pred"])
        cum_gpu += float(meta.get("train_time_sec", 0.0))
        steps.append(
            {
                "n_models": len(val_preds),
                "gpu_sec": cum_gpu,
                "ens_val_r": _r(np.mean(np.stack(val_preds), axis=0), val_y),
                "ens_oracle_r": _r(np.mean(np.stack(test_preds), axis=0), test_oracle),
                "single_val_r": float(meta["val_pearson"]),
                "single_oracle_r": float(
                    meta.get("per_set_metrics", {}).get("genomic", {}).get("pearson", float("nan"))
                ),
                "hp": {
                    k: meta.get("hp", {}).get(k)
                    for k in (
                        "optimizer",
                        "n_layers",
                        "width_base",
                        "lr",
                        "block_class",
                        "activation",
                    )
                },
            }
        )
    out = {"dir": str(vdir), "n_meta": n_meta_now, "steps": steps}
    cache.write_text(json.dumps(out, indent=2))
    return out


def _interp_at(xs: list[float], ys: list[float], x: float) -> float:
    """ys at cumulative-budget x (np.interp clamps outside the observed range)."""
    a = np.asarray(xs, float)
    b = np.asarray(ys, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2:
        return float("nan")
    return float(np.interp(x, a[keep], b[keep]))


def _gpu_to_frac(xs: list[float], ys: list[float], frac: float) -> float:
    """GPU-seconds at which the cumulative-best of ys first reaches frac×(its final)."""
    a = np.asarray(xs, float)
    b = np.maximum.accumulate(np.asarray(ys, float))  # cumulative-best (monotone)
    if not np.isfinite(b).any():
        return float("nan")
    target = frac * b[np.isfinite(b)][-1]
    hit = next((i for i, v in enumerate(b) if np.isfinite(v) and v >= target), len(b) - 1)
    return float(a[hit])


def collect(out_root: Path, cache_dir: Path) -> dict[str, list[dict]]:
    """{variant_label: [per-seed curve summary, ...]} over all cells present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_variant: dict[str, list[dict]] = {}
    for vdir in sorted(out_root.glob("k562_*_d*/seed*/llm_*")):
        if not vdir.is_dir():
            continue
        curve = variant_curve(vdir, cache_dir)
        if curve is None:
            continue
        steps = curve["steps"]
        last = steps[-1]
        xs = [s["gpu_sec"] for s in steps]
        per_variant.setdefault(vdir.name, []).append(
            {
                "cell": vdir.parts[-3],
                "seed": vdir.parts[-2],
                "n_models": last["n_models"],
                "total_gpu_sec": last["gpu_sec"],
                "final_ens_oracle_r": last["ens_oracle_r"],
                "final_ens_val_r": last["ens_val_r"],
                "best_single_oracle_r": float(np.nanmax([s["single_oracle_r"] for s in steps])),
                "best_single_val_r": float(np.nanmax([s["single_val_r"] for s in steps])),
                "done": (Path(curve["dir"]) / ".screen_done").exists(),
                "_xs": xs,
                "_ens_oracle": [s["ens_oracle_r"] for s in steps],
                "_ens_val": [s["ens_val_r"] for s in steps],
                "_hps": [s["hp"] for s in steps],
            }
        )
    return per_variant


def _hp_fingerprint(seed_runs: list[dict]) -> dict:
    """Modal/median HP signature across all this variant's trained configs (for the
    user to eyeball search-behavior diversity beyond the prompt axes)."""
    hps = [hp for run in seed_runs for hp in run["_hps"]]
    if not hps:
        return {}

    def modal(key):
        vals = [str(h.get(key)) for h in hps if h.get(key) is not None]
        return Counter(vals).most_common(1)[0][0] if vals else None

    def med(key):
        vals = [h.get(key) for h in hps if isinstance(h.get(key), (int, float))]
        return float(np.median(vals)) if vals else None

    return {
        "optimizer": modal("optimizer"),
        "block_class": modal("block_class"),
        "n_layers_med": med("n_layers"),
        "width_base_med": med("width_base"),
        "lr_med": med("lr"),
    }


def summarize(per_variant: dict[str, list[dict]]) -> tuple[list[dict], float]:
    # Common budget B = min total GPU-seconds across ALL variant×seed runs present, so
    # every variant has reached B and oracle@B is a fair efficiency comparison.
    all_totals = [run["total_gpu_sec"] for runs in per_variant.values() for run in runs]
    budget = float(min(all_totals)) if all_totals else 0.0

    recs = []
    for label, runs in per_variant.items():
        m = _VARIANT_RE.match(label)
        oracle_at_b = [_interp_at(r["_xs"], r["_ens_oracle"], budget) for r in runs]
        speed = [_gpu_to_frac(r["_xs"], r["_ens_oracle"], 0.95) for r in runs]
        recs.append(
            {
                "variant": label,
                "style": m.group("style") if m else "?",
                "model": m.group("model") if m else "?",
                "novel": int(m.group("novel")) if m else 0,
                "n_seeds": len(runs),
                "n_models_mean": float(np.mean([r["n_models"] for r in runs])),
                "complete": all(r["done"] for r in runs),
                "oracle_at_budget": float(np.nanmean(oracle_at_b)),
                "final_ens_oracle_r": float(np.nanmean([r["final_ens_oracle_r"] for r in runs])),
                "final_ens_val_r": float(np.nanmean([r["final_ens_val_r"] for r in runs])),
                "best_single_oracle_r": float(
                    np.nanmean([r["best_single_oracle_r"] for r in runs])
                ),
                "gpu_sec_to_95pct": float(np.nanmean(speed)),
                "total_gpu_sec_mean": float(np.nanmean([r["total_gpu_sec"] for r in runs])),
                "hp_fingerprint": _hp_fingerprint(runs),
            }
        )
    # Rank: efficiency (oracle@B) first, final-oracle as tiebreak.
    recs.sort(key=lambda r: (r["oracle_at_budget"], r["final_ens_oracle_r"]), reverse=True)
    for i, r in enumerate(recs):
        r["rank"] = i + 1
    return recs, budget


def pick_top3_diverse(recs: list[dict], k: int = 3, pick_model: str | None = None) -> list[dict]:
    """Greedy: walk the ranked list and take a variant only if its prompt STYLE is new,
    so the carried set spreads (not 3 near-identical). Backfill by pure rank if fewer
    than k distinct styles exist. If pick_model is set, only that proposer model is
    eligible (the deploy family is locked to one model — Sonnet downstream — while the
    other model stays in the table purely as the comparison axis)."""
    elig = [r for r in recs if pick_model is None or r["model"] == pick_model]
    chosen: list[dict] = []
    seen_styles: set[str] = set()
    for r in elig:
        if len(chosen) >= k:
            break
        if r["style"] not in seen_styles:
            chosen.append(r)
            seen_styles.add(r["style"])
    if len(chosen) < k:
        for r in elig:
            if len(chosen) >= k:
                break
            if r not in chosen:
                chosen.append(r)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(REPO / "outputs/hp_llm_screen_e100"))
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--report_dir", default=str(REPO / "outputs/analysis_figures/llm_screen"))
    ap.add_argument(
        "--pick_model",
        default="sonnet",
        help="lock the top-K to this proposer model (deploy is Sonnet-only); '' to allow any model",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else report_dir / "curve_cache"

    per_variant = collect(out_root, cache_dir)
    if not per_variant:
        raise SystemExit(f"no usable variant trajectories under {out_root}")
    recs, budget = summarize(per_variant)
    top3 = pick_top3_diverse(recs, pick_model=args.pick_model or None)

    n_complete = sum(r["complete"] for r in recs)
    print(f"=== LLM prompt-screen ranking ({len(recs)} variants; {n_complete} fully done) ===")
    print(f"common GPU-second budget B = {budget:,.0f}s (min total across variant×seed runs)")
    if n_complete < len(recs):
        print("  [PARTIAL] screen still running — ranking reflects trajectories so far.\n")
    hdr = (
        f"{'#':>2} {'variant':28} {'seeds':>5} {'mdl':>4} {'oracle@B':>9} "
        f"{'final_or':>8} {'final_val':>9} {'gpu→95%':>8} {'done':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        print(
            f"{r['rank']:>2} {r['variant']:28} {r['n_seeds']:>5} {r['n_models_mean']:>4.0f} "
            f"{r['oracle_at_budget']:>9.4f} {r['final_ens_oracle_r']:>8.4f} "
            f"{r['final_ens_val_r']:>9.4f} {r['gpu_sec_to_95pct']:>8.0f} "
            f"{'yes' if r['complete'] else 'no':>5}"
        )

    print("\n=== recommended TOP-3 (rank + distinct-style spread) ===")
    for r in top3:
        fp = r["hp_fingerprint"]
        print(
            f"  {r['variant']}  (style={r['style']} model={r['model']} novel={r['novel']})  "
            f"oracle@B={r['oracle_at_budget']:.4f}  "
            f"hp≈[{fp.get('optimizer')}/{fp.get('block_class')}/"
            f"L{fp.get('n_layers_med')}/w{fp.get('width_base_med')}]"
        )

    # Ready-to-paste LLM_VARIANTS block for submit_step1_bakeoff.py. NOTE: the bake-off
    # launcher currently sets LLM_MODEL + LLM_PROMPT_STYLE but NOT LLM_ALLOW_NOVEL_AXES —
    # if any pick has novel=1, thread a novel field through job_script before launch.
    print("\n=== paste into submit_step1_bakeoff.py LLM_VARIANTS (verify novel axis!) ===")
    any_novel = False
    for r in top3:
        full = _MODEL_FULL.get(r["model"], r["model"])
        any_novel = any_novel or r["novel"] == 1
        nv_note = "   # novel=1 → launcher needs LLM_ALLOW_NOVEL_AXES wired" if r["novel"] else ""
        print(f'    ("llm_{r["style"]}", "llm_autoresearch", "{full}", "{r["style"]}"),{nv_note}')
    if any_novel:
        print("  WARNING: a recommended variant uses novel-axes — bake-off launcher must pass it.")

    report = {
        "out_root": str(out_root),
        "common_budget_gpu_sec": budget,
        "n_variants": len(recs),
        "n_complete": n_complete,
        "ranking": recs,
        "recommended_top3": [r["variant"] for r in top3],
    }
    (report_dir / "screen_ranking.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {report_dir / 'screen_ranking.json'}")


if __name__ == "__main__":
    main()
