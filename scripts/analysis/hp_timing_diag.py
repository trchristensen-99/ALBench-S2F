"""Per-strategy HP-pick comparison to explain wildly different per-model times
at D=300k. Reads r*_meta.json and reports the HP axes most likely to drive cost
(epochs_trained, channels, batch_size, lr, block depth/size, kernel)."""

import glob
import json
import os

import numpy as np

ROOT = "outputs/hp_step1_bakeoff_e100/k562_genomic_d300000"
CELLS = [
    ("evo_batch_s0", "seed42_0/evo_batch"),
    ("evo_single_s1", "seed43_1/evo_single"),
    ("optuna_tpe_s0", "seed42_0/optuna_tpe"),
    ("optuna_gp_s0", "seed42_0/optuna_gp"),
    ("evo_adaptive_s0", "seed42_0/evo_adaptive"),
    ("evo_exploit_s0", "seed42_0/evo_exploit"),
    ("llm_explore_s0", "seed42_0/llm_explore_nv1"),
    ("llm_diverse_s0", "seed42_0/llm_diverse_nv1"),
]


def med(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def fmt(x, w=8):
    if x is None:
        return "-".rjust(w)
    if isinstance(x, float):
        return (f"{x:.3g}" if abs(x) < 100 else f"{x:.0f}").rjust(w)
    return str(x).rjust(w)


def total_params_proxy(hp):
    bs = hp.get("block_sizes") or hp.get("blocks") or []
    if isinstance(bs, list) and bs and isinstance(bs[0], (int, float)):
        return int(sum(bs))
    ch = hp.get("channels") or hp.get("base_channels") or 0
    nl = hp.get("num_layers") or hp.get("n_layers") or len(bs) if bs else 0
    return int(ch) * int(nl)


def main():
    print(
        f"{'cell':20s} {'n':>3s} {'mean_s':>8s} {'p90_s':>8s} {'ep_med':>7s} {'param_med':>10s} {'bs_med':>7s} {'lr_med':>9s} {'depth_med':>10s}"
    )
    samples = {}
    for name, p in CELLS:
        metas = sorted(glob.glob(os.path.join(ROOT, p, "r*_meta.json")))
        rows = []
        for m in metas:
            try:
                d = json.load(open(m))
            except Exception:
                continue
            hp = d.get("hp", {}) or {}
            bs = d.get("block_sizes", None)
            if bs is not None and "block_sizes" not in hp:
                hp = dict(hp)
                hp["block_sizes"] = bs
            rows.append((d.get("train_time_sec", 0), d.get("epochs_trained", 0), hp))
        if not rows:
            continue
        t = np.array([r[0] for r in rows])
        ep = np.array([r[1] for r in rows])
        param = med([total_params_proxy(r[2]) for r in rows])
        bs_v = med([r[2].get("batch_size") for r in rows])
        lr_v = med([r[2].get("lr") or r[2].get("learning_rate") for r in rows])
        depth = med([len(r[2].get("block_sizes") or []) or r[2].get("num_layers") for r in rows])
        print(
            f"{name:20s} {len(rows):3d} {t.mean():8.0f} {np.percentile(t, 90):8.0f} {np.median(ep):7.0f}  {fmt(param, 9)} {fmt(bs_v, 7)} {fmt(lr_v, 9)} {fmt(depth, 10)}"
        )
        samples[name] = rows
    print("\n=== Spearman: per-model train_time vs HP axis (pooled across cells) ===")
    from scipy.stats import spearmanr

    all_rows = [r for v in samples.values() for r in v]
    t = np.array([r[0] for r in all_rows])
    for key, getter in [
        ("epochs_trained", lambda r: r[1]),
        ("param_proxy", lambda r: total_params_proxy(r[2])),
        ("batch_size", lambda r: r[2].get("batch_size")),
        ("lr", lambda r: r[2].get("lr") or r[2].get("learning_rate")),
        ("depth", lambda r: len(r[2].get("block_sizes") or []) or r[2].get("num_layers")),
    ]:
        v = [getter(r) for r in all_rows]
        mask = np.array([x is not None for x in v])
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(t[mask], np.array([v[i] for i in range(len(v)) if mask[i]]))
        print(f"  {key:18s} rho={rho:+.3f}  n={int(mask.sum())}")

    print("\n=== two slowest + two fastest models, with hp diff ===")
    flat = [(r[0], r[1], r[2], name) for name, rows in samples.items() for r in rows]
    flat.sort(key=lambda x: x[0])
    for label, items in [("FAST", flat[:2]), ("SLOW", flat[-2:])]:
        for t, ep, hp, name in items:
            print(
                f"  [{label}] {name:18s} t={t:6.0f}s  ep={ep:3d}  hp={ {k: hp[k] for k in sorted(hp) if k in ('block_sizes', 'batch_size', 'lr', 'channels', 'num_layers', 'dropout', 'kernel_size', 'first_kernel', 'first_channels', 'base_channels', 'head_dim', 'schedule')} }"
            )


if __name__ == "__main__":
    main()
