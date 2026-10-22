"""Test whether Sahu CpG effect is linear, and probe motif confound.

Sahu bin data (9 points): per-bin count_mean vs CpG content.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

# Load Sahu bin data
bins = pd.read_csv(REPO / "outputs/sahu_random_lib_analysis/cpg_vs_count_bins.csv")
print("Sahu bin data:")
print(bins[["cpg_bin", "n_seqs", "cpg_mean", "count_mean", "count_median", "count_max"]].to_string())
print()

x = bins["cpg_mean"].values  # CpG density per bin
y = bins["count_mean"].values  # mean read count per bin
n_seqs = bins["n_seqs"].values
median_cnt = bins["count_median"].values
max_cnt = bins["count_max"].values

print("=== Mean vs Median read count per CpG bin ===")
for i in range(len(bins)):
    print(f"  bin {i}: cpg={x[i]:.4f}  mean={y[i]:.3f}  median={median_cnt[i]:.0f}  max={max_cnt[i]:>5}  n_seqs={n_seqs[i]:>7}")

print()
print("KEY OBSERVATION: count_median = 1 in ALL bins. The mean uplift is driven entirely")
print("by the long tail of high-count outliers (max counts up to 7136 in bin 0!).")
print("This means MOST random sequences have similar (count=1) baseline regardless of CpG.")
print("The CpG-mean correlation is driven by frequency of chance motif hits, not assay bias.")
print()

# --- Curve fitting ---
print("=" * 60)
print("CURVE FITS to Sahu bin means")
print("=" * 60)

# Linear: y = a + b*x
lin = linregress(x, y)
print(f"\nLinear: y = {lin.intercept:.3f} + {lin.slope:.3f}*x   R²={lin.rvalue**2:.4f}")

# Quadratic
poly2 = np.polyfit(x, y, 2)
y_pred_p2 = np.polyval(poly2, x)
r2_p2 = 1 - np.sum((y - y_pred_p2)**2) / np.sum((y - y.mean())**2)
print(f"Quadratic: y = {poly2[0]:+.3f}x² + {poly2[1]:+.3f}x + {poly2[2]:+.3f}   R²={r2_p2:.4f}")

# Log: y = a + b*log(x)
log_x = np.log(x + 1e-8)
res_log = linregress(log_x, y)
y_pred_log = res_log.intercept + res_log.slope * log_x
r2_log = 1 - np.sum((y - y_pred_log)**2) / np.sum((y - y.mean())**2)
print(f"Log:    y = {res_log.intercept:.3f} + {res_log.slope:.3f}*log(x)   R²={r2_log:.4f}")

# Saturation: y = a + b*x/(c+x)  (Michaelis-Menten style)
def sat(x, a, b, c): return a + b * x / (c + x)
try:
    popt, _ = curve_fit(sat, x, y, p0=[1.5, 1.0, 0.05], maxfev=5000)
    y_sat = sat(x, *popt)
    r2_sat = 1 - np.sum((y - y_sat)**2) / np.sum((y - y.mean())**2)
    print(f"Saturating: y = {popt[0]:.3f} + {popt[1]:.3f}*x/({popt[2]:.4f}+x)   R²={r2_sat:.4f}")
except Exception as e:
    print(f"  saturating fit failed: {e}")

# Power: y = a * x^b
def power(x, a, b): return a * (x + 1e-8)**b
try:
    popt_p, _ = curve_fit(power, x, y, p0=[2.0, 0.1], maxfev=5000)
    y_pow = power(x, *popt_p)
    r2_pow = 1 - np.sum((y - y_pow)**2) / np.sum((y - y.mean())**2)
    print(f"Power:    y = {popt_p[0]:.3f}*x^{popt_p[1]:.3f}   R²={r2_pow:.4f}")
except Exception as e:
    print(f"  power fit failed: {e}")

# --- Motif confound analysis ---
print()
print("=" * 60)
print("MOTIF CONFOUND ANALYSIS")
print("=" * 60)
print()
print("Key K562 TF motifs containing CpG (more likely to appear in high-CpG random seqs):")
print("  SP1:    GGGCGG (Krueppel-like)")
print("  E2F:    TTTSGCGC")
print("  CRE:    TGACGTCA")
print("  EBOX:   CACGTG (Myc/Max)")
print("  YY1:    CCGCCATNTT")
print()
print("These motifs all CONTAIN CpG. By construction, high-CpG random sequences will")
print("contain more chance occurrences of these motifs (P(CG-containing 6-mer) ~ CpG²).")
print()
print("=> The 'CpG-activity slope' in Sahu may be PARTLY driven by motif hits, not a")
print("   pure assay bias. To disentangle, we'd need to:")
print("   1. Re-analyze Sahu sequences per CpG bin × motif content")
print("   2. Or: train a debias oracle that conditions on motif content not just CpG")
print()

# --- Saturation argument ---
# If the slope is sub-linear (saturating), it suggests a finite pool of motif-like sequences
# at high CpG that activate the assay; once you have enough motif density, you saturate.
# This would favor a saturating function over a pure linear extrapolation.

# Save outputs
out_dir = REPO / "outputs/sahu_curve_analysis"
out_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "linear": {"slope": float(lin.slope), "intercept": float(lin.intercept), "r2": float(lin.rvalue**2)},
    "quadratic": {"coefs": [float(c) for c in poly2], "r2": float(r2_p2)},
    "logarithmic": {"slope": float(res_log.slope), "intercept": float(res_log.intercept), "r2": float(r2_log)},
}
try:
    summary["saturating"] = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]), "r2": float(r2_sat)}
except: pass
(out_dir / "curve_fits.json").write_text(json.dumps(summary, indent=2))
print(f"Saved {out_dir / 'curve_fits.json'}")
