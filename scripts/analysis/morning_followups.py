"""Head-to-head ablation comparisons + ks-by-depth analysis."""
import glob, json, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 11,
                     'figure.titlesize': 16})

OUT = 'outputs/analysis/morning_followups'
os.makedirs(OUT, exist_ok=True)

def load_meta(d):
    rows = []
    for m in sorted(glob.glob(os.path.join(d, 'r*_meta.json'))):
        try: x = json.load(open(m))
        except: continue
        vp = x.get('val_pearson')
        if vp is None: continue
        # oracle_r from per_set_metrics['genomic']['pearson']
        psm = x.get('per_set_metrics', {})
        oracle_r = None
        if 'genomic' in psm:
            oracle_r = psm['genomic'].get('pearson') if isinstance(psm['genomic'], dict) else None
        rows.append({'val_pearson': float(vp), 'oracle_r': float(oracle_r) if oracle_r else None,
                     'hp': x.get('hp', {})})
    return rows

# ── HEAD-TO-HEAD ABLATIONS ──────────────────────────────────────────
COMPARISONS = [
    ('Weight Decay', [
        ('wd_PI (1e-8 to 1e-7)',   'outputs/overnight/wd_PI_1e-8_to_1e-7'),
        ('wd_EMP (1e-5 to 1e-3)', 'outputs/overnight/wd_EMP_1e-5_to_1e-3'),
    ]),
    ('Width jitter regime', [
        ('shrink [0.125, 1.0]', 'outputs/overnight/wj_shrink'),
        ('uniform [0.5, 2.0]',  'outputs/overnight/wj_uniform'),
        ('grow [1.0, 4.0]',     'outputs/overnight/wj_grow'),
    ]),
    ('Dropout', [
        ('zero ~0', 'outputs/overnight/drop_zero'),
        ('positive [0.1, 0.5]', 'outputs/overnight/drop_pos'),
    ]),
    ('Optimizer (forced)', [
        ('adam',  'outputs/overnight/opt_adam'),
        ('adamw', 'outputs/overnight/opt_adamw'),
        ('muon',  'outputs/overnight/opt_muon'),
    ]),
    ('lr_schedule (forced)', [
        ('plateau',  'outputs/overnight/sch_plateau'),
        ('onecycle', 'outputs/overnight/sch_onecycle'),
        ('cosine',   'outputs/overnight/sch_cosine'),
    ]),
    ('block_class (forced)', [
        ('eff',   'outputs/overnight/bc_eff'),
        ('plain', 'outputs/overnight/bc_plain'),
        ('ag',    'outputs/overnight/bc_ag'),
    ]),
    ('ks menu (with vs without 5)', [
        ('with ks=5  {5,7,11,15}', 'outputs/overnight/ks_with5'),
        ('no ks=5    {7,11,15}',    'outputs/overnight/ks_no5'),
    ]),
]

nrows = len(COMPARISONS)
fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.5 * nrows))
if nrows == 1: axes = [axes]
summary = []
for ax, (title, regimes) in zip(axes, COMPARISONS):
    bars = []
    labels = []
    for lbl, path in regimes:
        rows = load_meta(path)
        if not rows: continue
        # top-5 by val_pearson
        rows.sort(key=lambda r: -r['val_pearson'])
        top5 = rows[:5]
        # If oracle_r missing, fall back to val_pearson
        scores = [r['oracle_r'] if r['oracle_r'] is not None else r['val_pearson'] for r in top5]
        m = float(np.mean(scores))
        s = float(np.std(scores))
        bars.append((m, s, len(rows)))
        labels.append(f'{lbl}\n(n={len(rows)})')
        summary.append({'family': title, 'regime': lbl, 'n_models': len(rows),
                        'top5_mean': m, 'top5_std': s})
    xs = np.arange(len(bars))
    means = [b[0] for b in bars]
    stds = [b[1] for b in bars]
    ax.bar(xs, means, yerr=stds, capsize=5, color=['#5080d0','#d05050','#50d050','#d0a050','#a050d0'][:len(bars)], edgecolor='k')
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('top-5 mean score\n(oracle_r or val_pearson)')
    ax.set_title(f'{title}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Tight y-range
    if means:
        ymin = min(m - s for m, s in zip(means, stds)) - 0.005
        ymax = max(m + s for m, s in zip(means, stds)) + 0.005
        ax.set_ylim(ymin, ymax)
    for x, m in zip(xs, means):
        ax.text(x, m, f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('Head-to-head ablation comparisons (top-5 mean per regime)', fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ablation_head_to_head.png'), dpi=140, bbox_inches='tight')
plt.close()

# CSV
with open(os.path.join(OUT, 'ablation_summary.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=['family', 'regime', 'n_models', 'top5_mean', 'top5_std'])
    w.writeheader(); w.writerows(summary)

print(f'saved ablation_head_to_head.png + ablation_summary.csv')

# ── KS-BY-DEPTH ANALYSIS (existing data: ks is single per model) ─────
# Question: does the OPTIMAL single ks depend on n_layers?
print('\nks vs depth analysis')
E100 = 'outputs/hp_step1_bakeoff_e100'
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
CELLS_BY_D = {
    30000: [('k562_genomic_d30000', s) for s in ['seed42_0','seed43_1','seed44_2']]
         + [('k562_motif_planted_v2_d30000', s) for s in ['seed42_0','seed43_1']]
         + [('k562_dinuc_shuffle_d30000', s) for s in ['seed42_0','seed43_1']],
    300000: [('k562_genomic_d300000', s) for s in ['seed42_0','seed43_1','seed44_2']],
}
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, D in zip(axes, [30000, 300000]):
    # Bucket top-N models by n_layers, look at ks distribution per bucket
    pool = []
    for cell, seed in CELLS_BY_D[D]:
        sd = os.path.join(E100, cell, seed)
        if not os.path.isdir(sd): continue
        rows = []
        for s in K5:
            for m in sorted(glob.glob(os.path.join(sd, s, 'r*_meta.json')))[:75]:
                try: d = json.load(open(m))
                except: continue
                if 'val_pearson' not in d: continue
                hp = d.get('hp', {})
                rows.append({'val_pearson': float(d['val_pearson']),
                             'n_layers': hp.get('n_layers'), 'ks': hp.get('ks')})
        rows.sort(key=lambda r: -r['val_pearson'])
        pool.extend(rows[:30])
    # 2D heatmap: n_layers (rows) × ks (cols), normalized per row
    n_layers_vals = sorted({r['n_layers'] for r in pool if r['n_layers'] is not None})
    ks_vals = sorted({r['ks'] for r in pool if r['ks'] is not None})
    mat = np.zeros((len(n_layers_vals), len(ks_vals)))
    for r in pool:
        if r['n_layers'] is None or r['ks'] is None: continue
        i = n_layers_vals.index(r['n_layers'])
        j = ks_vals.index(r['ks'])
        mat[i, j] += 1
    # Normalize per row
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.where(row_sums > 0, mat / row_sums, 0)
    im = ax.imshow(mat_norm, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(ks_vals))); ax.set_xticklabels(ks_vals)
    ax.set_yticks(range(len(n_layers_vals))); ax.set_yticklabels(n_layers_vals)
    ax.set_xlabel('ks (kernel size)'); ax.set_ylabel('n_layers')
    ax.set_title(f'D={D//1000}k — ks distribution per depth (row-normalized)')
    for i in range(len(n_layers_vals)):
        for j in range(len(ks_vals)):
            if mat[i,j] > 0:
                ax.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                        color='white' if mat_norm[i,j] > 0.5 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, label='fraction of top models at this depth')
fig.suptitle('Does optimal ks depend on n_layers?  (current data: single ks per model — per-LAYER ks needs new ablation)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ks_vs_depth.png'), dpi=140, bbox_inches='tight')
plt.close()
print('saved ks_vs_depth.png')
