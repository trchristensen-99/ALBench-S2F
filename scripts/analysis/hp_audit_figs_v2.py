"""Regenerate A1/A2/A3 figures with large fonts + cleaner layout for PI-deck use."""
import glob, json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
    'figure.titlesize': 20,
})

E100 = 'outputs/hp_step1_bakeoff_e100'
OUT = 'outputs/analysis/hp_audit_v2'
os.makedirs(OUT, exist_ok=True)
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
BUDGET = 75; TOPN = 10

CELLS = [
    ('k562_genomic_d30000', 'seed42_0'),
    ('k562_genomic_d30000', 'seed44_2'),
    ('k562_genomic_d300000', 'seed42_0'),
    ('k562_genomic_d300000', 'seed44_2'),
    ('k562_motif_planted_v2_d30000', 'seed42_0'),
    ('k562_dinuc_shuffle_d30000', 'seed42_0'),
]
HP_KEYS = ['lr', 'batch_size', 'conv_dropout', 'dense_dropout', 'n_layers',
           'width_base', 'block_class', 'ks', 'weight_decay']

def load_cell(cell_dir):
    lab_path = None
    for s in K5:
        p = os.path.join(cell_dir, s, 'labels.npz')
        if os.path.exists(p): lab_path = p; break
    if not lab_path: return None, None, None
    lab = np.load(lab_path)
    vy, oy = lab['val_labels'], lab['test_oracle']
    rows = []
    for s in K5:
        sd = os.path.join(cell_dir, s)
        if not os.path.isdir(sd): continue
        metas = []
        for m in sorted(glob.glob(os.path.join(sd, 'r*_meta.json'))):
            try: d = json.load(open(m))
            except: continue
            if 'val_pearson' not in d: continue
            metas.append((int(d.get('round',-1)), float(d['val_pearson']), d, m))
        metas.sort(); metas = metas[:BUDGET]
        for r, vp, d, m in metas:
            try: z = np.load(m.replace('_meta.json', '.npz'))
            except: continue
            if z['val_pred'].shape != vy.shape or z['test_pred'].shape != oy.shape: continue
            rows.append({'strategy': s, 'round': r, 'val_pearson': vp, 'hp': d.get('hp', {}),
                         'val_pred': z['val_pred'], 'test_pred': z['test_pred']})
    return rows, vy, oy

def fit_ensemble(rows, vy, oy):
    V = np.array([r['val_pred'] for r in rows]).T
    T = np.array([r['test_pred'] for r in rows]).T
    en = ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return en.coef_, float(pearsonr(en.predict(T), oy)[0])

# Composite A1: parallel coordinates 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
cell_data = []
for ax, (cell, seed) in zip(axes.flat, CELLS):
    cd = os.path.join(E100, cell, seed)
    rows, vy, oy = load_cell(cd)
    if not rows: 
        ax.set_title(f'{cell}/{seed} (no data)'); continue
    w, score = fit_ensemble(rows, vy, oy)
    cell_data.append((cell, seed, rows, w, score))
    top_w_idx = np.argsort(-w)[:TOPN]
    keys = HP_KEYS
    xs = np.arange(len(keys))
    wmax = max(w[top_w_idx].max(), 1e-9)
    cmap = plt.cm.viridis
    # Normalize each HP axis to [0,1] for visual clarity
    for i in top_w_idx:
        vals = []
        for k in keys:
            v = rows[i]['hp'].get(k)
            if isinstance(v, bool): v = int(v)
            if isinstance(v, str):
                v = {'eff':0, 'ag':1, 'plain':2}.get(v, hash(v)%3)
            if v is None: v = 0
            try: v = float(v)
            except: v = 0
            # Normalize log for lr, wd
            if k in {'lr', 'weight_decay'} and v > 0: v = (np.log10(v) + 6) / 6  # roughly [0,1] from 1e-6 to 1e0
            elif k == 'batch_size' and v > 0: v = np.log2(v) / 12  # 1-4096
            elif k == 'width_base' and v > 0: v = np.log2(v) / 10  # 8-1024
            elif k == 'n_layers': v = v / 12
            elif k == 'ks': v = v / 11
            elif k in {'conv_dropout', 'dense_dropout'}: v = v / 0.5
            elif k == 'block_class': v = v / 2
            vals.append(v)
        c = cmap(w[i] / wmax)
        ax.plot(xs, vals, color=c, alpha=0.7, lw=2)
    ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=45, ha='right')
    short = cell.replace('k562_','').replace('_planted_v2','').replace('_shuffle','')
    Dlabel = '30k' if '30000' in cell else '300k'
    ax.set_title(f'{short}  {seed}\nensemble oracle_r = {score:.4f}', fontweight='bold')
    ax.set_ylim(-0.1, 1.1); ax.set_ylabel('normalized HP value')
    ax.grid(True, alpha=0.3)
fig.suptitle('A1 — HPs of top-10 ensemble members (color = ENet weight, dark→light = low→high)', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'A1_coords_all.png'), dpi=150, bbox_inches='tight')
plt.close()

# A2: round-index histograms (2x3 grid, each has 2 sub-bars side-by-side)
fig, axes = plt.subplots(2, 3, figsize=(22, 11))
for ax, (cell, seed, rows, w, score) in zip(axes.flat, cell_data):
    rounds = np.array([r['round'] for r in rows])
    top_w = np.argsort(-w)[:TOPN]
    top_v = np.argsort(-np.array([r['val_pearson'] for r in rows]))[:TOPN]
    bins = np.linspace(0, max(rounds)+1, 16)
    ax.hist(rounds[top_w], bins=bins, alpha=0.7, color='steelblue', label='top-10 by ENSEMBLE WEIGHT', edgecolor='k')
    ax.hist(rounds[top_v], bins=bins, alpha=0.5, color='salmon', label='top-10 by val_pearson', edgecolor='k')
    ax.set_xlabel('round index')
    ax.set_ylabel('count')
    short = cell.replace('k562_','').replace('_planted_v2','').replace('_shuffle','')
    ax.set_title(f'{short}  {seed}', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
fig.suptitle('A2 — Round-index distribution of top-10 models (ensemble vs single-best)', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'A2_rounds_all.png'), dpi=150, bbox_inches='tight')
plt.close()

# A3: scatter (val_pearson vs ENet weight) with overlap visualization
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
for ax, (cell, seed, rows, w, score) in zip(axes.flat, cell_data):
    vs = np.array([r['val_pearson'] for r in rows])
    top_w = set(np.argsort(-w)[:TOPN].tolist())
    top_v = set(np.argsort(-vs)[:TOPN].tolist())
    both = top_w & top_v
    only_w = top_w - top_v
    only_v = top_v - top_w
    other = set(range(len(rows))) - top_w - top_v
    for grp, color, label, sz in [(other,'lightgray','other models',20),
                                   (only_v,'salmon','single-only',150),
                                   (only_w,'steelblue','ensemble-only',150),
                                   (both,'darkviolet','both top',180)]:
        if not grp: continue
        idx = list(grp)
        ax.scatter(vs[idx], w[idx], c=color, label=label, s=sz, alpha=0.7,
                   edgecolors='k', linewidths=0.5)
    ax.set_xlabel('val_pearson (single model)')
    ax.set_ylabel('ENet weight')
    short = cell.replace('k562_','').replace('_planted_v2','').replace('_shuffle','')
    jaccard = len(both)/(len(both)+len(only_w)+len(only_v)) if (both|only_w|only_v) else 0
    sp = float(spearmanr(w, vs).statistic)
    ax.set_title(f'{short}  {seed}\njaccard={jaccard:.2f}  spearman={sp:.2f}', fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
fig.suptitle('A3 — Which models does the ensemble actually use?  (blue = picked by ensemble but NOT top-10 single; red = top-10 single but DROPPED by ensemble)', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'A3_overlap_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

print('saved 3 composite figures to', OUT)
