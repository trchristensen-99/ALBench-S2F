"""Parallel coordinates per (reservoir, D) — pool seeds. Match ~/Downloads/hp_coord_plots/ style.
- One figure per (reservoir × D) cell
- ALL trials shown, colored by val_pearson
- Numeric tick labels per axis
- Sep line before val_loss + val_pearson_r outcome axes
- Title shows trial count + best Pearson
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

CELLS_BY_RD = {  # (reservoir, D) → list of seed dirs
    ('genomic', 30000):  ['seed42_0', 'seed43_1', 'seed44_2'],
    ('motif_planted_v2', 30000): ['seed42_0', 'seed43_1'],
    ('dinuc_shuffle', 30000):    ['seed42_0', 'seed43_1'],
    ('genomic', 300000): ['seed42_0', 'seed43_1', 'seed44_2'],
    ('motif_planted_v2', 300000): ['seed42_0'],
    ('dinuc_shuffle', 300000):    ['seed42_0'],
}

# Axes: HPs (8) then dashed-line separator then outcomes (2)
HP_AXES = [
    ('lr', 'log', None),
    ('batch_size', 'log', None),
    ('weight_decay', 'log', None),
    ('conv_dropout', 'linear', (0, 0.5)),
    ('dense_dropout', 'linear', (0, 0.5)),
    ('width_base', 'log', None),
    ('ks', 'linear', (3, 11)),
    ('n_layers', 'linear', (2, 12)),
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
                hp = d.get('hp', {})
                # try to read val_loss from training history
                vloss = None
                hist = d.get('val_trajectory')
                if isinstance(hist, dict):
                    vl = hist.get('val_loss')
                    if isinstance(vl, list) and vl:
                        # best (min) val_loss
                        try: vloss = float(min(vl))
                        except: pass
                if vloss is None:
                    vloss = d.get('best_val_loss')
                rows.append({'hp': hp, 'val_pearson': float(vp), 'val_loss': vloss})
    return rows

for (R, D), seeds in CELLS_BY_RD.items():
    cell = f'k562_{R}_d{D}'
    rows = collect(cell, seeds)
    if not rows:
        print(f'SKIP {cell}: no rows'); continue
    n = len(rows)
    best_vp = max(r['val_pearson'] for r in rows)
    # Build matrix: rows = trials, cols = axes (HP + outcomes)
    n_axes = len(HP_AXES) + len(OUTCOMES)
    M = np.full((n, n_axes), np.nan)
    raw_vp = np.array([r['val_pearson'] for r in rows])
    # Normalize each axis to [0,1] for plotting; keep raw for tick labels
    raw_cols = []
    for j, (key, scale, _bounds) in enumerate(HP_AXES):
        vals = []
        for r in rows:
            v = r['hp'].get(key)
            if isinstance(v, bool): v = int(v)
            try: v = float(v) if v is not None else np.nan
            except: v = np.nan
            vals.append(v)
        v = np.array(vals)
        raw_cols.append(v.copy())
        if scale == 'log':
            v_t = np.log10(np.where(v > 0, v, np.nan))
        else:
            v_t = v
        # Normalize
        lo, hi = np.nanmin(v_t), np.nanmax(v_t)
        if hi - lo < 1e-9: M[:, j] = 0.5
        else: M[:, j] = (v_t - lo) / (hi - lo)
    # Outcome columns
    for k, ok in enumerate(OUTCOMES):
        j = len(HP_AXES) + k
        v = np.array([r[ok] if r[ok] is not None else np.nan for r in rows], dtype=float)
        raw_cols.append(v.copy())
        if np.isfinite(v).sum() < 2:
            M[:, j] = 0.5
        else:
            lo, hi = np.nanmin(v), np.nanmax(v)
            if hi - lo < 1e-9: M[:, j] = 0.5
            else: M[:, j] = (v - lo) / (hi - lo)
    # Plot
    fig, ax = plt.subplots(figsize=(15, 7))
    xs = np.arange(n_axes)
    cmap = cm.plasma
    norm = Normalize(vmin=np.percentile(raw_vp, 5), vmax=np.percentile(raw_vp, 95))
    order = np.argsort(raw_vp)  # plot worst first → best on top
    for i in order:
        if np.all(np.isnan(M[i])): continue
        ax.plot(xs, M[i], color=cmap(norm(raw_vp[i])), alpha=0.5, lw=1.0)
    # Axes ticks: 5 ticks per axis with REAL values
    for j, (key, scale, _bounds) in enumerate(HP_AXES):
        v = raw_cols[j]
        vfinite = v[np.isfinite(v)]
        if not len(vfinite): continue
        if scale == 'log':
            lo_log, hi_log = np.log10(vfinite.min()), np.log10(vfinite.max())
            ticks_log = np.linspace(lo_log, hi_log, 5)
            ticks_val = 10 ** ticks_log
            ticks_norm = np.linspace(0, 1, 5)
            for tn, tv in zip(ticks_norm, ticks_val):
                ax.text(j-0.02, tn, f'{tv:.1e}', ha='right', va='center', fontsize=9)
        else:
            lo, hi = vfinite.min(), vfinite.max()
            ticks_val = np.linspace(lo, hi, 5)
            ticks_norm = np.linspace(0, 1, 5)
            for tn, tv in zip(ticks_norm, ticks_val):
                lbl = f'{tv:.2g}' if (hi - lo) < 10 else f'{int(tv)}'
                ax.text(j-0.02, tn, lbl, ha='right', va='center', fontsize=9)
    # Outcome ticks
    for k, ok in enumerate(OUTCOMES):
        j = len(HP_AXES) + k
        v = raw_cols[j]
        vfinite = v[np.isfinite(v)]
        if not len(vfinite): continue
        ticks_val = np.linspace(vfinite.min(), vfinite.max(), 5)
        for tn, tv in zip(np.linspace(0, 1, 5), ticks_val):
            lbl = f'{tv:.3f}'
            ax.text(j-0.02, tn, lbl, ha='right', va='center', fontsize=9, color='darkred' if ok=='val_pearson' else 'k')
    # Vertical lines
    for j in xs:
        ax.axvline(j, color='gray', alpha=0.5, lw=0.7)
    # Separator before outcomes
    ax.axvline(len(HP_AXES) - 0.5, color='gray', alpha=0.8, lw=1.5, linestyle='--')
    # Labels
    labels = [k for k, _, _ in HP_AXES] + OUTCOMES
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    for tl, lbl in zip(ax.get_xticklabels(), labels):
        if lbl in OUTCOMES: tl.set_color('darkred')
    ax.set_xlim(-0.5, n_axes - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    Dlbl = f'{D//1000}k' if D >= 1000 else str(D)
    ax.set_title(f'{R}  D={Dlbl}   ({n} trials,  best val_pearson={best_vp:.4f})',
                 fontweight='bold')
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('val_pearson  (darker = lower / lighter = higher)', rotation=270, labelpad=18)
    plt.tight_layout()
    fpath = os.path.join(OUT, f'coord_{R}_d{D}.png')
    plt.savefig(fpath, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  saved {fpath}  ({n} trials, best={best_vp:.4f})')

print('done')
