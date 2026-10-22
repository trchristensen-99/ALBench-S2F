"""Per-axis HP convergence analysis. For each HP axis, show:
- Distribution of values among TOP-K models at D=30k vs D=300k
- Convergence score: how tightly the top models cluster on this axis
- Recommendation: 'strong-optimum' axes need careful search; 'weak' axes can be fixed/simplified.
"""
import glob, json, os, numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 11,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 16})

E100 = 'outputs/hp_step1_bakeoff_e100'
OUT = 'outputs/analysis/hp_convergence'
os.makedirs(OUT, exist_ok=True)
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
TOP_K = 50    # top-K by val_pearson per cell
BUDGET = 75   # only first 75 models per strategy

CELLS = {
    30000:  [('k562_genomic_d30000', s) for s in ['seed42_0','seed43_1','seed44_2']]
          + [('k562_motif_planted_v2_d30000', s) for s in ['seed42_0','seed43_1']]
          + [('k562_dinuc_shuffle_d30000', s) for s in ['seed42_0','seed43_1']],
    300000: [('k562_genomic_d300000', s) for s in ['seed42_0','seed43_1','seed44_2']],
}

def load_top_per_cell(cell, seed, k=TOP_K):
    rows = []
    for s in K5:
        for m in sorted(glob.glob(os.path.join(E100, cell, seed, s, 'r*_meta.json')))[:BUDGET]:
            try: d = json.load(open(m))
            except: continue
            if 'val_pearson' not in d: continue
            rows.append((float(d['val_pearson']), d.get('hp', {})))
    rows.sort(key=lambda x: -x[0])
    return [hp for _, hp in rows[:k]]

# Pool top-K across all cells per D
pools = {D: [] for D in CELLS}
for D, cells in CELLS.items():
    for cell, seed in cells:
        sd = os.path.join(E100, cell, seed)
        if not os.path.isdir(sd): continue
        pools[D].extend(load_top_per_cell(cell, seed, TOP_K))
print(f'Loaded top-{TOP_K} per cell × {sum(len(c) for c in CELLS.values())} cells:')
for D, p in pools.items():
    print(f'  D={D}: {len(p)} models pooled')

# HP axes (key, label, type, log)
AXES = [
    ('lr',            'lr',            'continuous', True),
    ('batch_size',    'batch_size',    'continuous', True),
    ('weight_decay',  'weight_decay',  'continuous', True),
    ('conv_dropout',  'conv_dropout',  'continuous', False),
    ('dense_dropout', 'dense_dropout', 'continuous', False),
    ('n_layers',      'n_layers',      'int',        False),
    ('width_base',    'width_base',    'continuous', True),
    ('ks',            'ks',            'int',        False),
    ('block_class',   'block_class',   'cat',        False),
    ('optimizer',     'optimizer',     'cat',        False),
    ('lr_schedule',   'lr_schedule',   'cat',        False),
    ('pct_start',     'pct_start',     'continuous', False),
    ('use_evoaug',    'use_evoaug',    'cat',        False),
    ('use_shift_aug', 'use_shift_aug', 'cat',        False),
]

def convergence_score(vals, kind, log=False):
    """Score 0-1. 1 = tight optimum (good signal). 0 = uniform (no signal)."""
    if not vals: return None
    if kind == 'cat':
        counter = Counter(str(v) for v in vals)
        n = sum(counter.values())
        probs = np.array([c/n for c in counter.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(max(len(counter), 2))
        return float(1 - entropy / max_entropy) if max_entropy > 0 else 0
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5: return None
    if log:
        arr = np.log10(np.where(arr > 0, arr, np.nan))
        arr = arr[np.isfinite(arr)]
    if len(arr) < 5: return None
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    total = arr.max() - arr.min()
    if total < 1e-9: return 1.0
    return float(1 - iqr / total)

# Plot grid: 14 axes → 4×4 grid (2 unused)
n_axes = len(AXES)
ncols = 4
nrows = (n_axes + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
axes = axes.flatten()

D_COLORS = {30000: '#5080d0', 300000: '#d05050'}
D_LABELS = {30000: 'D=30k', 300000: 'D=300k'}

scores_summary = []
for i, (key, label, kind, log) in enumerate(AXES):
    ax = axes[i]
    score_30k = score_300k = None
    for D in [30000, 300000]:
        vals = [h.get(key) for h in pools[D] if h.get(key) is not None]
        if kind == 'cat':
            vals = [str(v) for v in vals if v is not None]
            counter = Counter(vals)
            keys_sorted = sorted(counter.keys())
            ys = [counter[k] / max(sum(counter.values()), 1) for k in keys_sorted]
            xs = np.arange(len(keys_sorted))
            shift = -0.18 if D == 30000 else 0.18
            ax.bar(xs + shift, ys, width=0.35, color=D_COLORS[D], label=D_LABELS[D], alpha=0.85, edgecolor='k')
            ax.set_xticks(xs); ax.set_xticklabels(keys_sorted, rotation=20, ha='right', fontsize=9)
        else:
            arr = np.array([v for v in vals if v is not None], dtype=float)
            arr = arr[np.isfinite(arr)]
            if log: arr = np.log10(np.where(arr > 0, arr, np.nan)); arr = arr[np.isfinite(arr)]
            if len(arr) > 5:
                ax.hist(arr, bins=20, alpha=0.55, color=D_COLORS[D], label=D_LABELS[D], edgecolor='k')
        if D == 30000:
            score_30k = convergence_score([h.get(key) for h in pools[D]], kind, log)
        else:
            score_300k = convergence_score([h.get(key) for h in pools[D]], kind, log)
    if log: ax.set_xlabel(f'log10({label})')
    else: ax.set_xlabel(label)
    title_score = f'30k:{score_30k:.2f}' if score_30k is not None else '30k:—'
    title_score += f' / 300k:{score_300k:.2f}' if score_300k is not None else ' / 300k:—'
    ax.set_title(f'{label}  [{title_score}]', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    scores_summary.append({'axis': label, 'score_30k': score_30k, 'score_300k': score_300k})

# Hide unused axes
for j in range(len(AXES), len(axes)):
    axes[j].axis('off')

fig.suptitle('HP convergence in top-50 models per cell — pooled across reservoirs × seeds.\n'
             'Score = 1−(IQR/total range); higher = top models converge to a tight value (stronger signal).',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'convergence_per_axis.png'), dpi=140, bbox_inches='tight')
plt.close()
print('saved convergence_per_axis.png')

# Ranking table
import csv
with open(os.path.join(OUT, 'convergence_ranking.csv'), 'w') as f:
    w = csv.writer(f)
    w.writerow(['axis', 'score_D=30k', 'score_D=300k', 'mean'])
    for r in sorted(scores_summary, key=lambda x: -(x.get('score_30k') or 0 + (x.get('score_300k') or 0))/2):
        s30 = r.get('score_30k'); s3h = r.get('score_300k')
        m = (s30 + s3h)/2 if s30 is not None and s3h is not None else (s30 or s3h)
        w.writerow([r['axis'], f'{s30:.3f}' if s30 is not None else '', f'{s3h:.3f}' if s3h is not None else '', f'{m:.3f}' if m is not None else ''])

print('saved convergence_ranking.csv')
print('\n=== Convergence ranking (mean across D) ===')
print(f'{"axis":20s} {"D=30k":>8s} {"D=300k":>8s} {"mean":>8s}')
for r in sorted(scores_summary, key=lambda x: -((x.get('score_30k') or 0) + (x.get('score_300k') or 0))/2):
    s30 = r.get('score_30k') or 0; s3h = r.get('score_300k') or 0
    print(f'{r["axis"]:20s} {s30:8.3f} {s3h:8.3f} {(s30+s3h)/2:8.3f}')
