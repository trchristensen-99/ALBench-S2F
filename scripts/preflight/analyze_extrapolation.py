"""Fit γ_k power law from main-sweep + extension results, score
extrapolation against the held-out D points.

For each (test_set, method) cell:
  1. Load measured MSE at all training D points (main sweep + extension
     fitting points).
  2. Fit a shared power law per test_set:
       MSE_k(D) = a * (D * γ_k) ** (-α) + ε
     with shared (a, α, ε) across methods, only γ_k varies.
  3. For each held-out D in extrapolation_test_d, compute
     |fitted_log_MSE - measured_log_MSE|.
  4. Apply acceptance thresholds from configs/extrapolation_design.yaml.

Outputs:
  results/extrapolation/<test_set>/<method>/extrap_check.json
  results/extrapolation/SUMMARY.md (pass/warn/fail per cell)

Usage:
  uv run --no-sync python scripts/preflight/analyze_extrapolation.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / "configs" / "extrapolation_design.yaml"


def _power_law(D: np.ndarray, a: float, alpha: float, eps: float) -> np.ndarray:
    """Power-law form (γ_k folded into a per-method via the fit)."""
    return a * np.power(D, -alpha) + eps


def _fit_per_test_set(
    fitting_d: np.ndarray, mse_per_method: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit shared (a, α, ε) + per-method γ_k.

    Strategy: a power law where each method has its own effective-D
    multiplier. We parameterize as MSE_m(D) = a * (D * gamma_m)^(-alpha) + eps,
    which is equivalent to MSE_m(D) = (a / gamma_m**alpha) * D**(-alpha) + eps.
    So we can fit per-method `a_m = a / gamma_m**alpha`, then γ_m = (a / a_m)**(1/alpha).

    Pin gamma_random ≡ 1 by definition (a_random = a).
    """
    # 1. Fit random's curve to find (a, α, ε)
    if "random" not in mse_per_method:
        raise ValueError("Need 'random' MSE curve to anchor γ_random ≡ 1")
    y_random = np.asarray(mse_per_method["random"], dtype=np.float64)
    valid = np.isfinite(y_random)
    try:
        popt, _ = curve_fit(
            _power_law,
            fitting_d[valid],
            y_random[valid],
            p0=[max(y_random[valid]) - min(y_random[valid]), 0.3, min(y_random[valid])],
            bounds=([0, 0.05, 0], [np.inf, 2.0, np.inf]),
            maxfev=10_000,
        )
    except Exception as e:
        return {}, {"error": str(e)}
    a, alpha, eps = popt

    # 2. Per method: fit a_m using the same (alpha, eps), then γ_m
    gammas: dict[str, float] = {"random": 1.0}
    a_per_method: dict[str, float] = {"random": float(a)}
    for method, y in mse_per_method.items():
        if method == "random":
            continue
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(y)
        if valid.sum() < 3:
            gammas[method] = float("nan")
            continue
        try:
            (a_m,), _ = curve_fit(
                lambda D, a_m: a_m * np.power(D, -alpha) + eps,
                fitting_d[valid],
                y[valid],
                p0=[a],
                bounds=([0], [np.inf]),
                maxfev=10_000,
            )
            a_per_method[method] = float(a_m)
            gammas[method] = float((a / a_m) ** (1.0 / alpha))
        except Exception:
            gammas[method] = float("nan")

    fit_summary = {"a": float(a), "alpha": float(alpha), "eps": float(eps)}
    return gammas, fit_summary


def _check_extrapolation(
    fit_summary: dict[str, float],
    gammas: dict[str, float],
    extrap_d: np.ndarray,
    measured_extrap: dict[str, np.ndarray],
    thresholds: dict,
) -> dict[str, dict]:
    """Score predicted vs measured MSE at extrapolation points.

    measured_extrap[method]: array of measured MSEs at extrap_d points.
    """
    a = fit_summary["a"]
    alpha = fit_summary["alpha"]
    eps = fit_summary["eps"]
    out = {}
    for method, gamma in gammas.items():
        if not np.isfinite(gamma):
            out[method] = {"verdict": "no_fit", "gamma_k": gamma}
            continue
        if method not in measured_extrap:
            out[method] = {"verdict": "no_measured", "gamma_k": gamma}
            continue
        a_m = a / (gamma**alpha)
        predicted = a_m * np.power(extrap_d, -alpha) + eps
        measured = np.asarray(measured_extrap[method], dtype=np.float64)
        # Compare in log-MSE space (the way the law was fit)
        log_pred = np.log(np.maximum(predicted, 1e-12))
        log_meas = np.log(np.maximum(measured, 1e-12))
        errs = np.abs(log_pred - log_meas)
        max_err = float(np.max(errs))
        if max_err < thresholds["good"]:
            verdict = "pass"
        elif max_err < thresholds["warn"]:
            verdict = "warn"
        else:
            verdict = "fail"
        out[method] = {
            "verdict": verdict,
            "gamma_k": float(gamma),
            "max_log_err": max_err,
            "predicted_mse": [float(p) for p in predicted],
            "measured_mse": [float(m) for m in measured],
            "extrap_d": [int(d) for d in extrap_d],
        }
    return out


def _load_results(
    fitting_d: list[int], extrap_d: list[int], methods: list[str], main_sweep_dir: Path
) -> tuple[dict, dict]:
    """Walk main_sweep_dir, gather per (method, D) test MSE.

    Expected layout:
      results/exp1_1/d_init0/<arch>/<method>/d<D>/seed<seed>/result.json
        and / or
      results/exp1_extension/d<D>/<arch>/<method>/seed<seed>/result.json

    For γ_k fitting we use the lower envelope across archs (per
    pre-registration) and average across seeds within an arch first.
    """
    fitting_set = set(fitting_d)
    extrap_set = set(extrap_d)
    fit_per_arch_method_d: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    extrap_per_arch_method_d: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for f in main_sweep_dir.rglob("result.json"):
        d = json.loads(f.read_text())
        path_parts = f.parts
        # Try to parse from parts: d_init*/arch/method/dN/seedN/result.json
        # or extension: dN/arch/method/seedN/result.json
        try:
            arch = next(p for p in path_parts if p in ("legnet", "dream_rnn", "dream_attn"))
        except StopIteration:
            continue
        method = next((p for p in path_parts if p in methods), None)
        if method is None:
            continue
        d_str = next((p for p in path_parts if p.startswith("d") and p[1:].isdigit()), None)
        if d_str is None:
            continue
        d_train = int(d_str[1:])
        test_mse = float(d.get("test_mse_at_best_val", float("nan")))
        if d_train in fitting_set:
            fit_per_arch_method_d[(arch, method, d_train)].append(test_mse)
        elif d_train in extrap_set:
            extrap_per_arch_method_d[(arch, method, d_train)].append(test_mse)
    # Lower envelope across archs (per pre-reg)
    fit_curves: dict[str, np.ndarray] = {}
    for method in methods:
        per_d = []
        for d in fitting_d:
            arch_means = []
            for arch in ("legnet", "dream_rnn", "dream_attn"):
                seeds = fit_per_arch_method_d.get((arch, method, d), [])
                if seeds:
                    arch_means.append(np.mean(seeds))
            per_d.append(min(arch_means) if arch_means else float("nan"))
        fit_curves[method] = np.array(per_d)
    extrap_curves: dict[str, np.ndarray] = {}
    for method in methods:
        per_d = []
        for d in extrap_d:
            arch_means = []
            for arch in ("legnet", "dream_rnn", "dream_attn"):
                seeds = extrap_per_arch_method_d.get((arch, method, d), [])
                if seeds:
                    arch_means.append(np.mean(seeds))
            per_d.append(min(arch_means) if arch_means else float("nan"))
        extrap_curves[method] = np.array(per_d)
    return fit_curves, extrap_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--main_sweep_dir",
        default=str(REPO / "results" / "exp1_1"),
        help="Root dir of main sweep results.",
    )
    args = ap.parse_args()

    if not DESIGN.exists():
        raise SystemExit(f"missing {DESIGN}")
    design = yaml.safe_load(DESIGN.read_text())
    fitting_d = design["fitting_d_grid"]
    extrap_d = design["extrapolation_test_d"]
    methods = design["extension_methods"] + ["random"]
    thresholds = design["acceptance_thresholds"]

    main_sweep_dir = Path(args.main_sweep_dir)
    if not main_sweep_dir.exists():
        raise SystemExit(f"main sweep dir {main_sweep_dir} not found — run main sweep first")

    fit_curves, extrap_curves = _load_results(fitting_d, extrap_d, methods, main_sweep_dir)
    if not any(np.isfinite(v).any() for v in fit_curves.values()):
        raise SystemExit("no measured MSE found in main sweep results")

    fit_d_arr = np.array(fitting_d)
    extrap_d_arr = np.array(extrap_d)

    gammas, fit_summary = _fit_per_test_set(fit_d_arr, fit_curves)
    if "error" in fit_summary:
        raise SystemExit(f"fit failed: {fit_summary['error']}")

    print(
        f"Fit (a, α, ε) = ({fit_summary['a']:.4f}, {fit_summary['alpha']:.4f}, {fit_summary['eps']:.4f})"
    )
    for m, g in sorted(gammas.items()):
        print(f"  γ_{m} = {g:.4f}")

    extrap = _check_extrapolation(fit_summary, gammas, extrap_d_arr, extrap_curves, thresholds)

    out_dir = REPO / "results" / "extrapolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fit": fit_summary,
        "gammas": gammas,
        "extrap_check": extrap,
        "fitting_d_grid": fitting_d,
        "extrap_d": extrap_d,
        "thresholds": thresholds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_dir / 'summary.json'}")

    # Pretty-print verdicts
    print("\nExtrapolation verdicts:")
    for method, c in extrap.items():
        verdict = c.get("verdict", "n/a")
        max_err = c.get("max_log_err", float("nan"))
        print(f"  {method:30s} {verdict:8s} max_log_err={max_err:.3f}")


if __name__ == "__main__":
    main()
