"""Two cleaner coord-plot variants per cell:
  *_sample.png : random 150-trial subsample (representative)
  *_top.png    : top-150 trials by val_pearson (best-performing subset)
"""
import glob, json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 20, 'axes.labelsize': 14,
    'xtick.labelsize': 13, 'ytick.labelsize': 11, 'figure.titlesize': 22,
})

E100 = 'outputs/hp_step1_bakeoff_e100'
OUT = 'outputs/analysis/hp_coords_v3'
os.makedirs(OUT, exist_ok=True)
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']

CELLS_BY_RD = {
    ('genomic', 30000):  ['seed42_0', 'seed43_1', 'seed44_2'],
    ('motif_planted_v2', 30000): ['seed42_0', 'seed43_1'],
    ('dinuc_shuffle', 30000):    ['seed42_0', 'seed43_1'],
    ('genomic', 300000): ['seed42_0', 'seed43_1', 'seed44_2'],
    ('motif_planted_v2', 300000): ['seed42_0'],
    ('dinuc_shuffle', 300000):    ['seed42_0'],
}

HP_AXES = [
    ('lr', 'log'), ('batch_size', 'log'), ('weight_decay', 'log'),
    ('conv_dropout', 'linear'), ('dense_dropout', 'linear'),
    ('width_base', 'log'), ('ks', 'linear'), ('n_layers', 'linear'),
]
OUTCOMES = ['val_loss', 'val_pearson']

def collect(cell, seeds):
    rows = []
    for sd in seeds:
        for s in K5:
            for m in sorted(glob.glob(os.path.join(E100, cell, sd, s, 'r*_meta.json')))[:75]:
                try: d = json.load(open(m))
                except: continue
                vp = d.get('val_pearson')
                if vp is None: continue
                hist = d.get('val_trajectory') or {}
                vloss = None
                vl = hist.get('val_loss') if isinstance(hist, dict) else None
                if isinstance(vl, list) and vl:
                    try: vloss = float(min(vl))
                    except: pass
                if vloss is None: vloss = d.get('val_mse')
                rows.append({'hp': d.get('hp', {}), 'val_pearson': float(vp), 'val_loss': vloss})
    return rows

def make_plot(rows, title, fpath):
    n = len(rows)
    if not n: return
    best_vp = max(r['val_pearson'] for r in rows)
    n_axes = len(HP_AXES) + len(OUTCOMES)
    M = np.full((n, n_axes), np.nan)
    raw_cols = []
    raw_vp = np.array([r['val_pearson'] for r in rows])
    for j, (key, scale) in enumerate(HP_AXES):
        vals = []
        for r in rows:
            v = r['hp'].get(key)
            if isinstance(v, bool): v = int(v)
            try: v = float(v) if v is not None else np.nan
            except: v = np.nan
            vals.append(v)
        v = np.array(vals)
        raw_cols.append(v.copy())
        v_t = np.log10(np.where(v > 0, v, np.nan)) if scale == 'log' else v
        lo, hi = np.nanmin(v_t), np.nanmax(v_t)
        M[:, j] = (v_t - lo) / max(hi - lo, 1e-9)
    for k, ok in enumerate(OUTCOMES):
        j = len(HP_AXES) + k
        v = np.array([r[ok] if r[ok] is not None else np.nan for r in rows], dtype=float)
        raw_cols.append(v.copy())
        finite = np.isfinite(v)
        if finite.sum() < 2: M[:, j] = 0.5; continue
        lo, hi = np.nanmin(v), np.nanmax(v)
        M[:, j] = (v - lo) / max(hi - lo, 1e-9)
    fig, ax = plt.subplots(figsize=(15, 7))
    xs = np.arange(n_axes)
    cmap = cm.plasma
    norm = Normalize(vmin=np.percentile(raw_vp, 5), vmax=np.percentile(raw_vp, 95))
    order = np.argsort(raw_vp)
    for i in order:
        if np.all(np.isnan(M[i])): continue
        ax.plot(xs, M[i], color=cmap(norm(raw_vp[i])), alpha=0.6, lw=1.2)
    # Ticks
    for j, (key, scale) in enumerate(HP_AXES):
        v = raw_cols[j]
        f = v[np.isfinite(v)]
        if not len(f): continue
        if scale == 'log':
            ticks_val = 10 ** np.linspace(np.log10(f.min()), np.log10(f.max()), 5)
        else:
            ticks_val = np.linspace(f.min(), f.max(), 5)
        for tn, tv in zip(np.linspace(0,1,5), ticks_val):
            lbl = f'{tv:.1e}' if scale == 'log' else (f'{tv:.2g}' if (f.max()-f.min()) < 10 else f'{int(tv)}')
            ax.text(j-0.02, tn, lbl, ha='right', va='center', fontsize=9)
    for k, ok in enumerate(OUTCOMES):
        j = len(HP_AXES) + k
        v = raw_cols[j]; f = v[np.isfinite(v)]
        if not len(f): continue
        for tn, tv in zip(np.linspace(0,1,5), np.linspace(f.min(), f.max(), 5)):
            ax.text(j-0.02, tn, f'{tv:.3f}', ha='right', va='center', fontsize=9, color='darkred' if ok=='val_pearson' else 'k')
    for j in xs: ax.axvline(j, color='gray', alpha=0.5, lw=0.7)
    ax.axvline(len(HP_AXES) - 0.5, color='gray', alpha=0.8, lw=1.5, linestyle='--')
    labels = [k for k, _ in HP_AXES] + OUTCOMES
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=0)
    for tl, lbl in zip(ax.get_xticklabels(), labels):
        if lbl in OUTCOMES: tl.set_color('darkred')
    ax.set_xlim(-0.5, n_axes - 0.5); ax.set_ylim(-0.05, 1.05); ax.set_yticks([])
    ax.set_title(title, fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('val_pearson', rotation=270, labelpad=18)
    plt.tight_layout(); plt.savefig(fpath, dpi=140, bbox_inches='tight'); plt.close()
    print(f'  saved {fpath} (n={n}, best={best_vp:.4f})')

rng = np.random.default_rng(0)
for (R, D), seeds in CELLS_BY_RD.items():
    cell = f'k562_{R}_d{D}'
    rows = collect(cell, seeds)
    if not rows: continue
    Dlbl = f'{D//1000}k'
    # Random subsample
    sample_n = min(150, len(rows))
    idx = rng.choice(len(rows), size=sample_n, replace=False)
    make_plot([rows[i] for i in idx],
              f'{R}  D={Dlbl}  ({sample_n} random sample of {len(rows)} trials)',
              os.path.join(OUT, f'coord_{R}_d{D}_sample.png'))
    # Top-K by val_pearson
    top_rows = sorted(rows, key=lambda r: -r['val_pearson'])[:min(150, len(rows))]
    make_plot(top_rows,
              f'{R}  D={Dlbl}  (top {len(top_rows)} of {len(rows)} by val_pearson)',
              os.path.join(OUT, f'coord_{R}_d{D}_top.png'))
print('done')
