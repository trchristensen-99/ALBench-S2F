"""Compare NEW HP space (newspace_d30k_*) vs existing OLD HP space results at D=30k.

Two angles:
1. Random-vs-random: NEW random sampling vs existing OLD random-strategy bakeoff
   (most apples-to-apples — same strategy, different HP space)
2. NEW random vs OLD K=5 BEST: how does random in new space compare to
   strategies in old space?
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'figure.titlesize': 16})

OUT = 'outputs/analysis/new_vs_old_d30k'
os.makedirs(OUT, exist_ok=True)
E100 = 'outputs/hp_step1_bakeoff_e100'

def load_meta(d, k=None):
    rows = []
    for m in sorted(glob.glob(os.path.join(d, 'r*_meta.json'))):
        try: x = json.load(open(m))
        except: continue
        vp = x.get('val_pearson')
        if vp is None: continue
        psm = x.get('per_set_metrics', {})
        oracle_r = psm.get('genomic', {}).get('pearson') if isinstance(psm.get('genomic'), dict) else None
        rows.append({'val_pearson': float(vp),
                     'oracle_r': float(oracle_r) if oracle_r is not None else None,
                     'hp': x.get('hp', {}),
                     'meta_path': m})
    if k is not None: rows = rows[:k]
    return rows

# NEW HP space — random sampling, 6 cells
NEW_CELLS = {
    'genomic_s42': 'outputs/overnight/newspace_d30k_genomic_seed42',
    'genomic_s43': 'outputs/overnight/newspace_d30k_genomic_seed43',
    'motif_s42':   'outputs/overnight/newspace_d30k_motif_seed42',
    'motif_s43':   'outputs/overnight/newspace_d30k_motif_seed43',
    'dinuc_s42':   'outputs/overnight/newspace_d30k_dinuc_seed42',
    'dinuc_s43':   'outputs/overnight/newspace_d30k_dinuc_seed43',
}

# OLD HP space K=5 — existing bakeoff, 3 cells overlap
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
OLD_CELLS = {
    'genomic_s42': ('k562_genomic_d30000', 'seed42_0'),
    'genomic_s43': ('k562_genomic_d30000', 'seed43_1'),
    'motif_s42':   ('k562_motif_planted_v2_d30000', 'seed42_0'),
    'motif_s43':   ('k562_motif_planted_v2_d30000', 'seed43_1'),
    'dinuc_s42':   ('k562_dinuc_shuffle_d30000', 'seed42_0'),
    'dinuc_s43':   ('k562_dinuc_shuffle_d30000', 'seed43_1'),
}

results = []
for cell_name, new_path in NEW_CELLS.items():
    new_rows = load_meta(new_path)
    if not new_rows: continue
    # NEW: top-5 by val_pearson
    new_rows.sort(key=lambda r: -r['val_pearson'])
    new_top5 = new_rows[:5]
    new_scores = [r['oracle_r'] if r['oracle_r'] else r['val_pearson'] for r in new_top5]
    new_top5_mean = float(np.mean(new_scores))
    new_best = float(max(r['oracle_r'] if r['oracle_r'] else r['val_pearson'] for r in new_rows))
    
    # OLD: pool all K=5 strategies, top-5 by val_pearson
    cell, seed = OLD_CELLS[cell_name]
    old_rows = []
    for s in K5:
        sd = os.path.join(E100, cell, seed, s)
        if os.path.isdir(sd):
            old_rows.extend(load_meta(sd, k=75))
    if not old_rows: continue
    old_rows.sort(key=lambda r: -r['val_pearson'])
    old_top5 = old_rows[:5]
    old_scores = [r['oracle_r'] if r['oracle_r'] else r['val_pearson'] for r in old_top5]
    old_top5_mean = float(np.mean(old_scores))
    old_best = float(max(r['oracle_r'] if r['oracle_r'] else r['val_pearson'] for r in old_rows))
    
    results.append({
        'cell': cell_name,
        'new_n': len(new_rows), 'old_n': len(old_rows),
        'new_top5_mean': new_top5_mean, 'old_top5_mean': old_top5_mean,
        'new_best': new_best, 'old_best': old_best,
    })

# Print table
print(f'{"cell":15s} {"NEW_n":>6s} {"OLD_n":>6s} {"NEW_top5":>10s} {"OLD_top5":>10s} {"Δ":>8s} {"NEW_best":>10s} {"OLD_best":>10s}')
for r in results:
    delta = r['new_top5_mean'] - r['old_top5_mean']
    print(f'{r["cell"]:15s} {r["new_n"]:6d} {r["old_n"]:6d} {r["new_top5_mean"]:10.4f} {r["old_top5_mean"]:10.4f} {delta:+8.4f} {r["new_best"]:10.4f} {r["old_best"]:10.4f}')

if results:
    new_means = [r['new_top5_mean'] for r in results]
    old_means = [r['old_top5_mean'] for r in results]
    print(f'\nMEAN          new_top5 = {np.mean(new_means):.4f}')
    print(f'MEAN          old_top5 = {np.mean(old_means):.4f}')
    print(f'MEAN          Δ        = {np.mean(new_means) - np.mean(old_means):+.4f}')
    print(f'(positive Δ = NEW HP space beats OLD; note: NEW uses random sampling, OLD uses K=5 strategies)')

# Plot
if results:
    fig, ax = plt.subplots(figsize=(12, 6))
    cells = [r['cell'] for r in results]
    xs = np.arange(len(cells))
    ax.bar(xs - 0.2, [r['new_top5_mean'] for r in results], width=0.4,
           label=f'NEW space (random, n=~{int(np.mean([r["new_n"] for r in results]))})',
           color='#5080d0', edgecolor='k')
    ax.bar(xs + 0.2, [r['old_top5_mean'] for r in results], width=0.4,
           label=f'OLD space (K=5, n=~{int(np.mean([r["old_n"] for r in results]))})',
           color='#d05050', edgecolor='k')
    ax.set_xticks(xs); ax.set_xticklabels(cells, rotation=30)
    ax.set_ylabel('top-5 mean (oracle_r or val_pearson)')
    ax.set_title('NEW HP space (random) vs OLD HP space (K=5 strategies)\n— D=30k, top-5 by val_pearson —')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate Δ
    for x, r in zip(xs, results):
        delta = r['new_top5_mean'] - r['old_top5_mean']
        ymax = max(r['new_top5_mean'], r['old_top5_mean'])
        ax.text(x, ymax + 0.005, f'Δ={delta:+.4f}', ha='center', fontsize=10, fontweight='bold',
                color='green' if delta >= 0 else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'new_vs_old_d30k.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print(f'\nsaved {OUT}/new_vs_old_d30k.png')
