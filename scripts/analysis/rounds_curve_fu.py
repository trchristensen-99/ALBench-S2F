"""Per-round cumulative ensemble-oracle curves for the Phase-0 followup n2 cells.

For each cell: accumulate configs round-by-round (round field in meta json), refit the
ElasticNet stack on the cumulative pool, measure ensemble oracle-Pearson. Average across
the 3 hp-seeds per variant, then report the overall mean curve and the improvement
(curve - curve[round1]) vs number of autoresearch rounds.

NOTE: ensemble size grows with rounds (2 configs/round), so part of any rise is just a
bigger ensemble, not better proposals -- this is the rounds/size confound, flagged inline.
CPU/BLAS-bound -- run via srun cpuq with capped threads, never the login node.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from albench.ensemble import fit_elasticnet_stack

ROOT = Path("outputs/hp_llm_ablation_followup_e100/k562_genomic_d30000")
SEEDS = ["seed42_0", "seed43_1", "seed44_2"]
VARIANTS = ["llm_default_ctxnone", "llm_diverse_ctxnone_n2"]  # both span 12 rounds x 2/round


def cell_curve(cell):
    lab = np.load(cell / "labels.npz")
    vy, orc = lab["val_labels"], lab["test_oracle"]
    byround = defaultdict(list)
    for m in cell.glob("*_meta.json"):
        d = json.load(open(m))
        z = np.load(m.parent / m.name.replace("_meta.json", ".npz"))
        byround[int(d["round"])].append((z["val_pred"], z["test_pred"]))
    accV, accT, curve = [], [], []
    for r in sorted(byround):
        for vp, tp in byround[r]:
            accV.append(vp)
            accT.append(tp)
        _, tpred = fit_elasticnet_stack(np.vstack(accV), vy, np.vstack(accT))
        mt = np.isfinite(tpred) & np.isfinite(orc)
        curve.append(float(pearsonr(tpred[mt], orc[mt])[0]))
    return curve


def main():
    per_variant = {}
    for v in VARIANTS:
        curves = []
        for s in SEEDS:
            cell = ROOT / s / v
            if (cell / ".ablation_done").exists():
                curves.append(cell_curve(cell))
        L = min(len(c) for c in curves)
        per_variant[v] = np.array([c[:L] for c in curves])

    print("round = autoresearch rounds completed (2 configs/round, cumulative ensemble)\n")
    header = "{:>5} {:>11}".format("round", "ens_mem")
    for v in VARIANTS:
        header += " {:>22}".format(v.replace("llm_", ""))
    header += " {:>13} {:>13}".format("OVERALL_mean", "improv_vs_r1")
    print(header)

    L = min(per_variant[v].shape[1] for v in VARIANTS)
    overall0 = None
    for k in range(L):
        means = {v: float(per_variant[v][:, k].mean()) for v in VARIANTS}
        overall = float(np.mean([means[v] for v in VARIANTS]))
        if k == 0:
            overall0 = overall
        line = "{:>5} {:>11}".format(k + 1, 2 * (k + 1))
        for v in VARIANTS:
            line += " {:>22.4f}".format(means[v])
        line += " {:>13.4f} {:>+13.4f}".format(overall, overall - overall0)
        print(line)

    print("\nper-variant improvement vs round1 (last round - round1):")
    for v in VARIANTS:
        c = per_variant[v].mean(0)
        print(
            "  {:28s} r1={:.4f} -> r{}={:.4f}  (+{:.4f})".format(
                v, c[0], len(c), c[-1], c[-1] - c[0]
            )
        )

    print("\nmarginal per-round gain (overall mean, delta from previous round):")
    ov = np.mean([per_variant[v].mean(0) for v in VARIANTS], axis=0)
    for k in range(1, len(ov)):
        flag = "  <- below 0.008 noise floor" if abs(ov[k] - ov[k - 1]) < 0.008 else ""
        print("  r{:>2}->r{:<2}  {:+.4f}{}".format(k, k + 1, ov[k] - ov[k - 1], flag))

    _save_fig(per_variant, L, ov)


def _save_fig(per_variant, L, ov):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = np.arange(1, L + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for v in VARIANTS:
        c = per_variant[v].mean(0)[:L]
        sd = per_variant[v].std(0)[:L]
        ax1.plot(rounds, c, marker="o", label=v.replace("llm_", ""))
        ax1.fill_between(rounds, c - sd, c + sd, alpha=0.15)
    ax1.plot(rounds, ov[:L], marker="s", color="k", lw=2, label="overall mean")
    ax1.set_xlabel("autoresearch rounds completed (2 configs/round)")
    ax1.set_ylabel("cumulative ensemble oracle-Pearson")
    ax1.set_title("AutoResearch: ensemble oracle-r vs rounds (k562 genomic D=30k)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    improv = ov[:L] - ov[0]
    ax2.plot(rounds, improv, marker="s", color="C3", lw=2)
    for frac, lbl in [(0.75, "75%"), (0.90, "90%")]:
        y = frac * improv[-1]
        kx = int(np.argmax(improv >= y)) + 1
        ax2.axhline(y, ls="--", color="gray", alpha=0.5)
        ax2.annotate(
            "{} of total @ round {}".format(lbl, kx),
            (kx, y),
            textcoords="offset points",
            xytext=(5, -12),
            fontsize=9,
        )
    ax2.set_xlabel("autoresearch rounds completed")
    ax2.set_ylabel("overall improvement vs round 1")
    ax2.set_title("Marginal value of added rounds (saturating)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = Path("outputs/analysis_figures/llm_followup")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "rounds_curve.png", dpi=140)
    print("\nsaved figure -> {}".format(out / "rounds_curve.png"))


if __name__ == "__main__":
    main()
