"""One combined PNG per (R,D): left = random 100-trial subsample, right = top-100 by val_pearson."""
import glob, json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 18, 'axes.labelsize': 12,
                     'xtick.labelsize': 12, 'ytick.labelsize': 10, 'figure.titlesize': 20})

E100 = 'outputs/hp_step1_bakeoff_e100'
OUT = 'outputs/analysis/hp_coords_combined'
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
HP_AXES = [('lr','log'),('batch_size','log'),('weight_decay','log'),
           ('conv_dropout','linear'),('dense_dropout','linear'),
           ('width_base','log'),('ks','linear'),('n_layers','linear')]
OUTCOMES = ['val_loss', 'val_pearson']
SAMPLE_N = 100

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
                vl = hist.get('val_loss') if isinstance(hist, dict) else None
                vloss = float(min(vl)) if isinstance(vl, list) and vl else d.get('val_mse')
                rows.append({'hp': d.get('hp', {}), 'val_pearson': float(vp), 'val_loss': vloss})
    return rows

def render(ax, rows, title, norm, cmap):
    n = len(rows)
    if not n: ax.set_title(f'{title} (empty)'); return
    n_axes = len(HP_AXES) + len(OUTCOMES)
    M = np.full((n, n_axes), np.nan)
    raw_cols = []
    raw_vp = np.array([r['val_pearson'] for r in rows])
    for j,(key,scale) in enumerate(HP_AXES):
        vals = []
        for r in rows:
            v = r['hp'].get(key)
            if isinstance(v, bool): v = int(v)
            try: v = float(v) if v is not None else np.nan
            except: v = np.nan
            vals.append(v)
        v = np.array(vals); raw_cols.append(v.copy())
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
    xs = np.arange(n_axes)
    order = np.argsort(raw_vp)
    for i in order:
        if np.all(np.isnan(M[i])): continue
        ax.plot(xs, M[i], color=cmap(norm(raw_vp[i])), alpha=0.55, lw=1.0)
    # axis tick labels
    for j, (key, scale) in enumerate(HP_AXES):
        v = raw_cols[j]; f = v[np.isfinite(v)]
        if not len(f): continue
        if scale == 'log':
            ticks_val = 10 ** np.linspace(np.log10(f.min()), np.log10(f.max()), 5)
        else:
            ticks_val = np.linspace(f.min(), f.max(), 5)
        for tn, tv in zip(np.linspace(0,1,5), ticks_val):
            lbl = f'{tv:.1e}' if scale == 'log' else (f'{tv:.2g}' if (f.max()-f.min()) < 10 else f'{int(tv)}')
            ax.text(j-0.02, tn, lbl, ha='right', va='center', fontsize=8)
    for k, ok in enumerate(OUTCOMES):
        j = len(HP_AXES) + k
        v = raw_cols[j]; f = v[np.isfinite(v)]
        if not len(f): continue
        for tn, tv in zip(np.linspace(0,1,5), np.linspace(f.min(), f.max(), 5)):
            ax.text(j-0.02, tn, f'{tv:.3f}', ha='right', va='center', fontsize=8,
                    color='darkred' if ok=='val_pearson' else 'k')
    for j in xs: ax.axvline(j, color='gray', alpha=0.5, lw=0.7)
    ax.axvline(len(HP_AXES)-0.5, color='gray', alpha=0.8, lw=1.5, linestyle='--')
    labels = [k for k,_ in HP_AXES] + OUTCOMES
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=0)
    for tl, lbl in zip(ax.get_xticklabels(), labels):
        if lbl in OUTCOMES: tl.set_color('darkred')
    ax.set_xlim(-0.5, n_axes-0.5); ax.set_ylim(-0.05, 1.05); ax.set_yticks([])
    ax.set_title(title, fontweight='bold', fontsize=15)

rng = np.random.default_rng(0)
for (R, D), seeds in CELLS_BY_RD.items():
    cell = f'k562_{R}_d{D}'
    rows = collect(cell, seeds)
    if not rows: continue
    Dlbl = f'{D//1000}k'
    raw_vp_all = np.array([r['val_pearson'] for r in rows])
    cmap = cm.plasma
    norm = Normalize(vmin=np.percentile(raw_vp_all, 5), vmax=np.percentile(raw_vp_all, 95))
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    # left: random subsample
    sn = min(SAMPLE_N, len(rows))
    idx = rng.choice(len(rows), size=sn, replace=False)
    sample_rows = [rows[i] for i in idx]
    render(axes[0], sample_rows, f'random {sn} of {len(rows)} trials', norm, cmap)
    # right: top SAMPLE_N
    top_rows = sorted(rows, key=lambda r: -r['val_pearson'])[:min(SAMPLE_N, len(rows))]
    best_vp = top_rows[0]['val_pearson']
    render(axes[1], top_rows, f'top {len(top_rows)} by val_pearson (best={best_vp:.4f})', norm, cmap)
    fig.suptitle(f'{R}  D={Dlbl}  ({len(rows)} total trials, K=5 strategies × 75 model budget × {len(seeds)} seed(s))', fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.01, aspect=40, shrink=0.85)
    cbar.set_label('val_pearson  (colorbar shared across both panels)', rotation=270, labelpad=18)
    plt.savefig(os.path.join(OUT, f'coord_{R}_d{D}_combined.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  saved combined for {R}/{Dlbl}  (n={len(rows)})')
print('done')
