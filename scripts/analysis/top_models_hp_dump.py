"""Dump full HP configs of TOP-5 single (by val_pearson) and TOP-5 ENet-weighted ensemble.
Output: CSV + a clean PNG table per cell."""
import glob, json, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

OUT = 'outputs/analysis/top_models_hp'
os.makedirs(OUT, exist_ok=True)
E100 = 'outputs/hp_step1_bakeoff_e100'
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']

CELLS = [
    ("k562_genomic_d10000", ["seed42_0", "seed43_1"]),
    ("k562_motif_planted_v2_d10000", ["seed42_0", "seed43_1"]),
    ("k562_dinuc_shuffle_d10000", ["seed42_0", "seed43_1"]),
    ("k562_genomic_d100000", ["seed42_0", "seed43_1"]),
    ("k562_genomic_d1000000", ["seed42_0"]),
    ("k562_motif_planted_v2_d300000", ["seed42_0"]),
    ("k562_dinuc_shuffle_d300000", ["seed42_0"]),
    ('k562_genomic_d30000', ['seed42_0', 'seed43_1', 'seed44_2']),
    ('k562_motif_planted_v2_d30000', ['seed42_0', 'seed43_1']),
    ('k562_dinuc_shuffle_d30000', ['seed42_0', 'seed43_1']),
    ('k562_genomic_d300000', ['seed42_0', 'seed43_1', 'seed44_2']),
]
HP_KEYS = ['lr', 'batch_size', 'weight_decay', 'conv_dropout', 'dense_dropout',
           'n_layers', 'width_base', 'ks', 'block_class', 'optimizer',
           'lr_schedule', 'pct_start', 'use_evoaug', 'use_shift_aug']

def load_all(cell_dir):
    lab_path = None
    for s in K5:
        p = os.path.join(cell_dir, s, 'labels.npz')
        if os.path.exists(p): lab_path = p; break
    if not lab_path: return None, None, None
    lab = np.load(lab_path); vy, oy = lab['val_labels'], lab['test_oracle']
    rows = []
    for s in K5:
        for m in sorted(glob.glob(os.path.join(cell_dir, s, 'r*_meta.json')))[:75]:
            try: d = json.load(open(m))
            except: continue
            if 'val_pearson' not in d: continue
            try: z = np.load(m.replace('_meta.json', '.npz'))
            except: continue
            if z['val_pred'].shape != vy.shape or z['test_pred'].shape != oy.shape: continue
            rows.append({'strategy': s, 'val_pearson': float(d['val_pearson']),
                         'hp': d.get('hp', {}), 'val_pred': z['val_pred'],
                         'test_pred': z['test_pred'], 'meta': d.get('meta_path', '')})
    return rows, vy, oy

# Per-cell: top-5 single + top-5 by ENet weight
all_rows_csv = []
for cell, seeds in CELLS:
    for seed in seeds:
        cd = os.path.join(E100, cell, seed)
        if not os.path.isdir(cd): continue
        rows, vy, oy = load_all(cd)
        if not rows: continue
        # Top-5 single
        rows_by_val = sorted(rows, key=lambda r: -r['val_pearson'])
        top5_single = rows_by_val[:5]
        # ENet weights
        V = np.array([r['val_pred'] for r in rows]).T
        T = np.array([r['test_pred'] for r in rows]).T
        try:
            en = ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
            en.fit(V, vy)
            weights = en.coef_
        except: continue
        # Top-5 by weight
        idx_by_w = np.argsort(-weights)
        top5_ensemble = [rows[i] for i in idx_by_w[:5]]
        # Write to CSV
        for kind, top in [('top5_single', top5_single), ('top5_ensemble', top5_ensemble)]:
            for rank, r in enumerate(top):
                row = {'cell': cell, 'seed': seed, 'kind': kind, 'rank': rank+1,
                       'strategy': r['strategy'], 'val_pearson': r['val_pearson']}
                if kind == 'top5_ensemble':
                    i = idx_by_w[rank]
                    row['enet_weight'] = float(weights[i])
                for k in HP_KEYS:
                    v = r['hp'].get(k)
                    if isinstance(v, list): v = str(v)
                    row[k] = v
                all_rows_csv.append(row)

# Save CSV
csv_path = os.path.join(OUT, 'top_models_hp.csv')
with open(csv_path, 'w') as f:
    if all_rows_csv:
        all_keys = []
        for r in all_rows_csv:
            for k in r.keys():
                if k not in all_keys: all_keys.append(k)
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader(); w.writerows(all_rows_csv)
print(f'saved {csv_path}')

# Compact summary per cell — focus on a few key HPs in a clean table
SUMMARY_KEYS = ['lr', 'batch_size', 'weight_decay', 'conv_dropout', 'dense_dropout',
                'n_layers', 'width_base', 'ks', 'block_class', 'optimizer']
def fmt(v):
    if v is None: return '—'
    if isinstance(v, float):
        if abs(v) < 1e-2 or abs(v) > 1e3: return f'{v:.2e}'
        return f'{v:.3f}' if abs(v) > 0.001 else '0'
    return str(v)

# Plot: one PNG per (D, kind), showing top-5 across all cells
for D in [30000, 300000]:
    relevant_cells = [(c, s) for c in {r['cell'] for r in all_rows_csv if str(D) in r['cell']}
                              for s in {r['seed'] for r in all_rows_csv if r['cell']==c}]
    for kind in ['top5_single', 'top5_ensemble']:
        subset = [r for r in all_rows_csv if str(D) in r['cell'] and r['kind']==kind]
        if not subset: continue
        n_rows = len(subset)
        fig, ax = plt.subplots(figsize=(20, max(6, 0.35 * n_rows)))
        ax.set_xlim(0, 100); ax.set_ylim(0, n_rows + 1.5); ax.axis('off')
        col_x = [0, 12, 22, 28, 34, 42, 50, 56, 62, 67, 72, 77, 84, 92, 100]
        headers = ['cell/seed', 'strategy', 'val_pearson'] + (['enet_w'] if kind == 'top5_ensemble' else []) + SUMMARY_KEYS
        # Header
        for i, h in enumerate(headers):
            x0 = col_x[i] if i < len(col_x)-1 else col_x[-2]
            x1 = col_x[i+1] if i+1 < len(col_x) else col_x[-1]
            ax.add_patch(Rectangle((x0, n_rows+0.3), x1-x0, 0.8, facecolor='#343a40', edgecolor='k'))
            ax.text((x0+x1)/2, n_rows+0.7, h, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        # Rows
        for i, r in enumerate(subset):
            y = n_rows - i - 1 + 0.3
            color = '#f0f8ff' if i % 2 == 0 else '#ffffff'
            for j, h in enumerate(headers):
                x0 = col_x[j] if j < len(col_x)-1 else col_x[-2]
                x1 = col_x[j+1] if j+1 < len(col_x) else col_x[-1]
                ax.add_patch(Rectangle((x0, y), x1-x0, 0.8, facecolor=color, edgecolor='gray', lw=0.3))
                if h == 'cell/seed': val = f'{r["cell"].replace("k562_", "").replace("_planted_v2", "").replace("_shuffle", "")}/{r["seed"]}'
                elif h == 'val_pearson': val = f'{r["val_pearson"]:.4f}'
                elif h == 'enet_w': val = f'{r.get("enet_weight", 0):.4f}'
                else: val = fmt(r.get(h))
                ax.text((x0+x1)/2, y+0.4, val, ha='center', va='center', fontsize=8, family='monospace')
        ax.set_title(f'{kind.replace("_", " ").title()} HPs at D={D//1000}k', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f'{kind}_d{D}.png'), dpi=140, bbox_inches='tight')
        plt.close()
        print(f'saved {kind}_d{D}.png ({n_rows} rows)')
