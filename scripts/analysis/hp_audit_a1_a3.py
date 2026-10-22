"""Phase A analyses for the HP-space audit (PI feedback 2026-06-26):
A1 — parallel-coordinates plot of HPs for top-N ensemble members per cell
A2 — round-index histogram of top-N ensemble members (early vs late)
A3 — set overlap between top-K-single-best and top-K-ensemble-weighted models

Run via sbatch on cpuq. Saves PNGs + a JSON summary to outputs/analysis/hp_audit/.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV

E100 = 'outputs/hp_step1_bakeoff_e100'
OUT = 'outputs/analysis/hp_audit'
os.makedirs(OUT, exist_ok=True)
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
BUDGET = 75
TOPN = 10  # top-N by weight / by single val

# Cells to analyze: D=30k genomic (3 seeds), D=300k genomic (3 seeds), plus 1 each of motif & dinuc
CELLS = [
    ('k562_genomic_d30000', 'seed42_0'),
    ('k562_genomic_d30000', 'seed43_1'),
    ('k562_genomic_d30000', 'seed44_2'),
    ('k562_genomic_d300000', 'seed42_0'),
    ('k562_genomic_d300000', 'seed43_1'),
    ('k562_genomic_d300000', 'seed44_2'),
    ('k562_motif_planted_v2_d30000', 'seed42_0'),
    ('k562_dinuc_shuffle_d30000', 'seed42_0'),
]

HP_KEYS = ['lr', 'batch_size', 'conv_dropout', 'dense_dropout', 'n_layers', 'width_base',
           'block_class', 'ks', 'weight_decay', 'use_evoaug', 'use_shift_aug']

def load_cell(cell_dir):
    """Return list of (strategy, round_idx, val_pearson, hp_dict, val_pred, test_pred) for top-BUDGET per K5 strat."""
    lab_path = None
    for s in K5:
        p = os.path.join(cell_dir, s, 'labels.npz')
        if os.path.exists(p): lab_path = p; break
    if not lab_path: return None, None, None
    lab = np.load(lab_path)
    vy, oy = lab['val_labels'], lab['test_oracle']
    rows = []
    for s in K5:
        strat_dir = os.path.join(cell_dir, s)
        if not os.path.isdir(strat_dir): continue
        # Take first BUDGET by round
        metas = []
        for m in sorted(glob.glob(os.path.join(strat_dir, 'r*_meta.json'))):
            try: d = json.load(open(m))
            except: continue
            if 'val_pearson' not in d: continue
            metas.append((int(d.get('round',-1)), float(d['val_pearson']), d, m))
        metas.sort()
        metas = metas[:BUDGET]
        for r, vp, d, m in metas:
            try: z = np.load(m.replace('_meta.json', '.npz'))
            except: continue
            if z['val_pred'].shape != vy.shape or z['test_pred'].shape != oy.shape: continue
            hp = d.get('hp', {})
            rows.append({'strategy': s, 'round': r, 'val_pearson': vp, 'hp': hp,
                         'val_pred': z['val_pred'], 'test_pred': z['test_pred'],
                         'meta_path': m})
    return rows, vy, oy

def fit_ensemble(rows, vy, oy):
    V = np.array([r['val_pred'] for r in rows]).T
    T = np.array([r['test_pred'] for r in rows]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    weights = en.coef_
    pred = en.predict(T)
    score = float(pearsonr(pred, oy)[0])
    return weights, score

def plot_parallel_coords(rows, weights, score, out_path, title):
    # Top-N by weight + top-N by val_pearson
    top_w_idx = np.argsort(-weights)[:TOPN]
    top_v_idx = np.argsort(-np.array([r['val_pearson'] for r in rows]))[:TOPN]
    # Plot HP values for top-N by weight, colored by weight
    fig, ax = plt.subplots(figsize=(14, 6))
    keys = HP_KEYS
    xs = np.arange(len(keys))
    cmap = plt.cm.viridis
    wmax = max(weights[top_w_idx].max(), 1e-9)
    for i in top_w_idx:
        vals = []
        for k in keys:
            v = rows[i]['hp'].get(k)
            if isinstance(v, bool): v = int(v)
            if isinstance(v, str):
                # Encode categorical to small int
                v = hash(v) % 5
            if v is None: v = 0
            try: v = float(v)
            except: v = 0
            # Log-scale lr and weight_decay
            if k in {'lr', 'weight_decay'} and v > 0:
                v = np.log10(v)
            vals.append(v)
        c = cmap(weights[i] / wmax)
        ax.plot(xs, vals, color=c, alpha=0.6, lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_title(f'{title}  (top-{TOPN} ensemble by weight, viridis=weight; ensemble_oracle_r={score:.4f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()

def plot_round_hist(rows, weights, out_path, title):
    rounds = np.array([r['round'] for r in rows])
    top_w = np.argsort(-weights)[:TOPN]
    top_v = np.argsort(-np.array([r['val_pearson'] for r in rows]))[:TOPN]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax[0].hist(rounds[top_w], bins=20, color='steelblue', edgecolor='k')
    ax[0].set_title(f'top-{TOPN} by ENSEMBLE WEIGHT — round indices')
    ax[0].set_xlabel('round'); ax[0].set_ylabel('count')
    ax[1].hist(rounds[top_v], bins=20, color='salmon', edgecolor='k')
    ax[1].set_title(f'top-{TOPN} by SINGLE val_pearson — round indices')
    ax[1].set_xlabel('round')
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

def set_overlap(rows, weights):
    """A3: how much do top-N by weight overlap with top-N by val_pearson?"""
    top_w = set(np.argsort(-weights)[:TOPN].tolist())
    top_v = set(np.argsort(-np.array([r['val_pearson'] for r in rows]))[:TOPN].tolist())
    inter = top_w & top_v
    union = top_w | top_v
    jaccard = len(inter) / len(union) if union else 0
    # Rank correlation across ALL models
    ws = np.array(weights)
    vs = np.array([r['val_pearson'] for r in rows])
    only_w = top_w - top_v  # in ensemble, not in single-top
    only_v = top_v - top_w  # in single-top, not in ensemble
    # HP-distance: per disjoint member, get strategy + round + HP summary
    def summarize(i):
        h = rows[i]['hp']
        return {'strategy': rows[i]['strategy'], 'round': rows[i]['round'],
                'val_pearson': rows[i]['val_pearson'], 'weight': float(weights[i]),
                'ks': h.get('ks'), 'width_base': h.get('width_base'),
                'n_layers': h.get('n_layers'), 'block_class': h.get('block_class')}
    return {
        'jaccard_topN': jaccard,
        'spearman_all': float(spearmanr(ws, vs).statistic),
        'n_in_both': len(inter),
        'n_ensemble_only': len(only_w),
        'n_single_only': len(only_v),
        'ensemble_only_examples': [summarize(i) for i in list(only_w)[:5]],
        'single_only_examples': [summarize(i) for i in list(only_v)[:5]],
    }

summary = {}
for cell, seed in CELLS:
    cd = os.path.join(E100, cell, seed)
    if not os.path.isdir(cd):
        print(f'SKIP missing: {cd}'); continue
    print(f'\n=== {cell}/{seed} ===', flush=True)
    rows, vy, oy = load_cell(cd)
    if not rows:
        print('  no rows'); continue
    weights, score = fit_ensemble(rows, vy, oy)
    n_nonzero = int((weights > 1e-9).sum())
    print(f'  loaded {len(rows)} models  ensemble_oracle_r={score:.4f}  n_nonzero_weights={n_nonzero}')
    tag = f'{cell}__{seed}'
    plot_parallel_coords(rows, weights, score, os.path.join(OUT, f'A1_coords_{tag}.png'), f'{cell}/{seed}')
    plot_round_hist(rows, weights, os.path.join(OUT, f'A2_rounds_{tag}.png'), f'{cell}/{seed}')
    ov = set_overlap(rows, weights)
    print(f'  A3 overlap: jaccard_top{TOPN}={ov["jaccard_topN"]:.2f}  spearman_w_vs_v={ov["spearman_all"]:.3f}  '
          f'(both={ov["n_in_both"]}, ensemble_only={ov["n_ensemble_only"]}, single_only={ov["n_single_only"]})')
    summary[tag] = {'score': score, 'n_models': len(rows), 'n_nonzero': n_nonzero, **ov}

with open(os.path.join(OUT, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f'\nSaved summary + plots to {OUT}/')
