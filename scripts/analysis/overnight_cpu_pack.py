"""Comprehensive CPU analyses for overnight. All run on existing data, no GPU needed.

A. Strategy-specific HP preferences (per-strategy histograms)
B. HP correlation matrix among top-K models
C. Width_jitter per-layer pattern in TOP models (do they grow or shrink?)
D. Ensemble size sweep (oracle_r vs K)
E. Per-D ks distribution comparison (10k vs 30k vs 300k)
F. Reservoir-specific HP preferences
G. Top-1 vs top-K val_pearson gap

Saves to outputs/analysis/overnight_cpu/.
"""
import glob, json, os, csv
import numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.titlesize': 14})

E100 = 'outputs/hp_step1_bakeoff_e100'
OUT = 'outputs/analysis/overnight_cpu'
os.makedirs(OUT, exist_ok=True)
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
BUDGET = 75
TOP_K = 30

CELLS_BY_D = {
    30000: [('k562_genomic_d30000', s) for s in ['seed42_0','seed43_1','seed44_2']]
         + [('k562_motif_planted_v2_d30000', s) for s in ['seed42_0','seed43_1']]
         + [('k562_dinuc_shuffle_d30000', s) for s in ['seed42_0','seed43_1']],
    300000: [('k562_genomic_d300000', s) for s in ['seed42_0','seed43_1','seed44_2']],
    10000: [('k562_genomic_d10000', s) for s in ['seed42_0','seed43_1']]
         + [('k562_motif_planted_v2_d10000', s) for s in ['seed42_0','seed43_1']]
         + [('k562_dinuc_shuffle_d10000', s) for s in ['seed42_0','seed43_1']],
}

def load_all(cell, seed):
    rows = []
    for s in K5:
        for m in sorted(glob.glob(os.path.join(E100, cell, seed, s, 'r*_meta.json')))[:BUDGET]:
            try: d = json.load(open(m))
            except: continue
            if 'val_pearson' not in d: continue
            rows.append({'strategy': s, 'val_pearson': float(d['val_pearson']),
                         'hp': d.get('hp', {}), 'meta_path': m})
    rows.sort(key=lambda r: -r['val_pearson'])
    return rows

# Load top-K per cell, with strategy labels
def load_top_with_strategy(D, k=TOP_K):
    pool = []
    for cell, seed in CELLS_BY_D[D]:
        sd = os.path.join(E100, cell, seed)
        if not os.path.isdir(sd): continue
        pool.extend(load_all(cell, seed)[:k])
    return pool

print('Loading...')
pools = {D: load_top_with_strategy(D) for D in CELLS_BY_D}
for D, p in pools.items(): print(f'  D={D}: {len(p)} top models pooled')

# ── A) Strategy-specific HP preferences ─────────────────────────────
print('\nA) Strategy-specific HP preferences (D=30k)')
HP_KEYS = ['lr', 'batch_size', 'weight_decay', 'conv_dropout', 'dense_dropout',
           'n_layers', 'width_base', 'ks', 'block_class', 'optimizer']
LOG_KEYS = {'lr', 'batch_size', 'weight_decay', 'width_base'}
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
strat_colors = {'evo_single': '#1f77b4', 'evo_batch': '#ff7f0e', 'optuna_gp': '#2ca02c',
                'optuna_tpe': '#d62728', 'llm_explore_nv1': '#9467bd'}
for i, key in enumerate(HP_KEYS):
    ax = axes[i]
    for s in K5:
        vals = [h['hp'].get(key) for h in pools[30000] if h['strategy'] == s and h['hp'].get(key) is not None]
        if not vals: continue
        if isinstance(vals[0], (int, float)):
            arr = np.array([float(v) for v in vals if v is not None])
            arr = arr[np.isfinite(arr)]
            if key in LOG_KEYS and (arr > 0).any():
                arr = np.log10(arr[arr > 0])
            ax.hist(arr, bins=15, alpha=0.5, color=strat_colors[s], label=s, edgecolor='k')
        else:
            counter = Counter(str(v) for v in vals if v is not None)
            keys_sorted = sorted(counter.keys())
            n_keys = len(keys_sorted)
            xs = np.arange(n_keys)
            offset = (list(K5).index(s) - 2) * 0.18
            ys = [counter.get(k, 0) for k in keys_sorted]
            ax.bar(xs + offset, ys, width=0.16, color=strat_colors[s], label=s, alpha=0.85)
            ax.set_xticks(xs); ax.set_xticklabels(keys_sorted, rotation=15, ha='right', fontsize=8)
    ax.set_title(f'{"log10(" if key in LOG_KEYS else ""}{key}{")" if key in LOG_KEYS else ""}', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
for j in range(len(HP_KEYS), len(axes)): axes[j].axis('off')
fig.suptitle('A) Strategy-specific HP preferences in top-30 models per cell (D=30k)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'A_strategy_hp_prefs.png'), dpi=140, bbox_inches='tight')
plt.close()
print('  saved A_strategy_hp_prefs.png')

# ── B) HP correlation matrix among top models ──────────────────────
print('B) HP correlation matrix')
for D in [30000, 300000]:
    pool = pools[D]
    rows = []
    for h in pool:
        row = {}
        for k in HP_KEYS:
            v = h['hp'].get(k)
            if isinstance(v, bool): v = int(v)
            try: v = float(v) if isinstance(v, (int, float)) else hash(str(v)) % 100
            except: v = np.nan
            if k in LOG_KEYS and v > 0: v = np.log10(v)
            row[k] = v
        rows.append(row)
    if not rows: continue
    mat = np.full((len(rows), len(HP_KEYS)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(HP_KEYS):
            mat[i, j] = r.get(k, np.nan)
    # Compute correlations
    corr = np.full((len(HP_KEYS), len(HP_KEYS)), np.nan)
    for i in range(len(HP_KEYS)):
        for j in range(len(HP_KEYS)):
            x, y = mat[:, i], mat[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 3:
                try: corr[i, j] = pearsonr(x[mask], y[mask])[0]
                except: pass
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(HP_KEYS))); ax.set_xticklabels(HP_KEYS, rotation=45, ha='right')
    ax.set_yticks(range(len(HP_KEYS))); ax.set_yticklabels(HP_KEYS)
    for i in range(len(HP_KEYS)):
        for j in range(len(HP_KEYS)):
            if np.isfinite(corr[i, j]):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        color='white' if abs(corr[i, j]) > 0.5 else 'black', fontsize=9)
    ax.set_title(f'HP correlation matrix — top-{TOP_K}-per-cell pooled at D={D//1000}k')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'B_hp_corr_d{D}.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  saved B_hp_corr_d{D}.png')

# ── C) width_jitter per-layer pattern in TOP models ────────────────
print('C) width_jitter per-layer pattern')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, D in zip(axes, [30000, 300000]):
    pool = pools[D]
    jitters = []
    for h in pool:
        j = h['hp'].get('width_jitter')
        if isinstance(j, list) and len(j) >= 4:
            jitters.append(j)
    if not jitters: continue
    maxlen = max(len(j) for j in jitters)
    mat = np.full((len(jitters), maxlen), np.nan)
    for i, j in enumerate(jitters): mat[i, :len(j)] = j
    means = np.nanmean(mat, axis=0)
    medians = np.nanmedian(mat, axis=0)
    p25 = np.nanpercentile(mat, 25, axis=0)
    p75 = np.nanpercentile(mat, 75, axis=0)
    xs = np.arange(maxlen)
    ax.fill_between(xs, p25, p75, alpha=0.25, color='steelblue', label='IQR')
    ax.plot(xs, medians, 'o-', color='steelblue', label='median', lw=2)
    ax.plot(xs, means, 's--', color='darkred', label='mean', lw=1.5)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='1.0 (uniform)')
    ax.set_xlabel('layer index'); ax.set_ylabel('width_jitter')
    ax.set_title(f'width_jitter shape in top-30 models per cell  (D={D//1000}k, n={len(jitters)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'C_width_jitter_shape.png'), dpi=140, bbox_inches='tight')
plt.close()
print('  saved C_width_jitter_shape.png')

# ── D) Ensemble size sweep ─────────────────────────────────────────
print('D) Ensemble size sweep')
ENSEMBLE_KS = [1, 2, 3, 5, 8, 12, 20, 30, 50]
results = []
for D in [30000, 300000]:
    cells = CELLS_BY_D[D]
    for cell, seed in cells[:5]:  # cap at 5 cells to keep this quick
        sd = os.path.join(E100, cell, seed)
        if not os.path.isdir(sd): continue
        # Load all models with val_pred + test_pred + labels
        lab_path = None
        for s in K5:
            p = os.path.join(sd, s, 'labels.npz')
            if os.path.exists(p): lab_path = p; break
        if not lab_path: continue
        lab = np.load(lab_path); vy, oy = lab['val_labels'], lab['test_oracle']
        models = []
        for s in K5:
            for m in sorted(glob.glob(os.path.join(sd, s, 'r*_meta.json')))[:BUDGET]:
                try: d = json.load(open(m))
                except: continue
                vp = d.get('val_pearson')
                if vp is None: continue
                try: z = np.load(m.replace('_meta.json', '.npz'))
                except: continue
                if z['val_pred'].shape != vy.shape or z['test_pred'].shape != oy.shape: continue
                models.append({'val_pearson': float(vp), 'val_pred': z['val_pred'], 'test_pred': z['test_pred']})
        models.sort(key=lambda x: -x['val_pearson'])
        if not models: continue
        for K in ENSEMBLE_KS:
            top = models[:K]
            if len(top) < K: continue
            # ENet ensemble
            V = np.array([m['val_pred'] for m in top]).T
            T = np.array([m['test_pred'] for m in top]).T
            try:
                en = ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0], positive=True, cv=min(5, K), max_iter=20000, n_jobs=1)
                en.fit(V, vy)
                score = float(pearsonr(en.predict(T), oy)[0])
                results.append({'D': D, 'cell': cell, 'seed': seed, 'K': K, 'score': score})
            except Exception as e: pass
# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for D in [30000, 300000]:
    df = [r for r in results if r['D'] == D]
    if not df: continue
    Ks = sorted(set(r['K'] for r in df))
    means = [np.mean([r['score'] for r in df if r['K']==K]) for K in Ks]
    stds = [np.std([r['score'] for r in df if r['K']==K]) for K in Ks]
    ax.errorbar(Ks, means, yerr=stds, marker='o', label=f'D={D//1000}k', capsize=4, lw=2)
ax.set_xlabel('ensemble size K (top-K by val_pearson, then ElasticNetCV)')
ax.set_ylabel('oracle_r on chr-test')
ax.set_title('D) Ensemble quality vs size K')
ax.set_xscale('log')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'D_ensemble_size_sweep.png'), dpi=140, bbox_inches='tight')
plt.close()
print('  saved D_ensemble_size_sweep.png')

# ── E) Per-D ks distribution comparison ────────────────────────────
print('E) Per-D ks distribution')
fig, ax = plt.subplots(figsize=(10, 6))
for D, color in [(10000, '#2ca02c'), (30000, '#1f77b4'), (300000, '#d62728')]:
    pool = pools.get(D, [])
    vals = [h['hp'].get('ks') for h in pool if h['hp'].get('ks') is not None]
    if not vals: continue
    counter = Counter(int(v) for v in vals)
    keys_sorted = sorted(counter.keys())
    total = sum(counter.values())
    ys = [counter[k]/total for k in keys_sorted]
    ax.plot(keys_sorted, ys, 'o-', color=color, label=f'D={D//1000}k (n={total})', lw=2, ms=8)
ax.set_xlabel('ks (kernel size, bp)'); ax.set_ylabel('fraction of top-30 models')
ax.set_title('E) Optimal ks shifts with D')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'E_ks_per_D.png'), dpi=140, bbox_inches='tight')
plt.close()
print('  saved E_ks_per_D.png')

# ── F) Reservoir-specific HP preferences ───────────────────────────
print('F) Reservoir-specific HP preferences (D=30k)')
RES_KEYS = ['genomic', 'motif_planted_v2', 'dinuc_shuffle']
by_res = {r: [] for r in RES_KEYS}
for cell, seed in CELLS_BY_D[30000]:
    for r in RES_KEYS:
        if r in cell:
            sd = os.path.join(E100, cell, seed)
            if os.path.isdir(sd): by_res[r].extend(load_all(cell, seed)[:TOP_K])
fig, axes = plt.subplots(3, 4, figsize=(18, 12)); axes = axes.flatten()
res_colors = {'genomic': '#1f77b4', 'motif_planted_v2': '#ff7f0e', 'dinuc_shuffle': '#2ca02c'}
for i, key in enumerate(HP_KEYS):
    ax = axes[i]
    for r in RES_KEYS:
        vals = [h['hp'].get(key) for h in by_res[r] if h['hp'].get(key) is not None]
        if not vals: continue
        if isinstance(vals[0], (int, float)):
            arr = np.array([float(v) for v in vals], dtype=float)
            arr = arr[np.isfinite(arr)]
            if key in LOG_KEYS and (arr > 0).any(): arr = np.log10(arr[arr > 0])
            ax.hist(arr, bins=12, alpha=0.5, color=res_colors[r], label=r, edgecolor='k')
        else:
            counter = Counter(str(v) for v in vals if v is not None)
            keys_sorted = sorted(counter.keys())
            xs = np.arange(len(keys_sorted))
            offset = (RES_KEYS.index(r) - 1) * 0.25
            ax.bar(xs + offset, [counter.get(k, 0) for k in keys_sorted], width=0.22,
                   color=res_colors[r], label=r, alpha=0.85)
            ax.set_xticks(xs); ax.set_xticklabels(keys_sorted, rotation=15, ha='right', fontsize=8)
    ax.set_title(f'{"log10(" if key in LOG_KEYS else ""}{key}{")" if key in LOG_KEYS else ""}', fontsize=11)
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)
for j in range(len(HP_KEYS), len(axes)): axes[j].axis('off')
fig.suptitle('F) Reservoir-specific HP preferences (D=30k, top-30 per cell)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'F_reservoir_hp_prefs.png'), dpi=140, bbox_inches='tight')
plt.close()
print('  saved F_reservoir_hp_prefs.png')

# ── G) Top-1 vs top-K val_pearson gap ──────────────────────────────
print('G) top-1 vs top-K val_pearson gap')
gaps = []
for D in [30000, 300000]:
    for cell, seed in CELLS_BY_D[D]:
        sd = os.path.join(E100, cell, seed)
        if not os.path.isdir(sd): continue
        rows = load_all(cell, seed)
        if len(rows) < 30: continue
        top1 = rows[0]['val_pearson']
        for K in [5, 10, 20, 30]:
            mean_topK = np.mean([r['val_pearson'] for r in rows[:K]])
            gap = top1 - mean_topK
            gaps.append({'D': D, 'cell': cell, 'seed': seed, 'K': K, 'top1': top1, 'mean_topK': mean_topK, 'gap': gap})
fig, ax = plt.subplots(figsize=(9, 6))
for D, color in [(30000, '#1f77b4'), (300000, '#d62728')]:
    df = [g for g in gaps if g['D'] == D]
    Ks = sorted(set(g['K'] for g in df))
    means = [np.mean([g['gap'] for g in df if g['K']==K]) for K in Ks]
    ax.plot(Ks, means, 'o-', color=color, label=f'D={D//1000}k', lw=2, ms=8)
ax.set_xlabel('K (ensemble size)'); ax.set_ylabel('top1 − mean(top-K) val_pearson')
ax.set_title('G) Gap between best single and top-K mean — how much room for ensembling?')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'G_top1_vs_topK_gap.png'), dpi=140, bbox_inches='tight')
plt.close()
print('  saved G_top1_vs_topK_gap.png')

print('\n=== all CPU analyses done ===')
print(f'outputs in {OUT}/')
