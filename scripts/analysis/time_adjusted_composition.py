"""Time-adjusted ensemble composition: at each GPU-second budget T,
what is the optimal ensemble (oracle_r), what is its size, and which strategies dominate?

Output:
- Per-cell oracle_r vs cumulative GPU-seconds curves
- Optimal K (ensemble size) at each time
- Strategy composition (fraction from each K=5 strat) of greedy K* ensemble at each time
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

OUT = 'outputs/analysis/time_adjusted'
os.makedirs(OUT, exist_ok=True)
E100 = 'outputs/hp_step1_bakeoff_e100'
K5 = ['optuna_gp', 'evo_batch', 'llm_explore_nv1', 'evo_single', 'optuna_tpe']

CELLS = {
    30000: [('k562_genomic_d30000', s) for s in ['seed42_0','seed43_1','seed44_2']]
         + [('k562_motif_planted_v2_d30000', s) for s in ['seed42_0','seed43_1']]
         + [('k562_dinuc_shuffle_d30000', s) for s in ['seed42_0','seed43_1']],
    300000: [('k562_genomic_d300000', s) for s in ['seed42_0','seed43_1','seed44_2']],
}

def load_models(cell, seed):
    sd = os.path.join(E100, cell, seed)
    lab_path = None
    for s in K5:
        p = os.path.join(sd, s, 'labels.npz')
        if os.path.exists(p): lab_path = p; break
    if not lab_path: return None
    lab = np.load(lab_path); vy, oy = lab['val_labels'], lab['test_oracle']
    models_by_strat = {s: [] for s in K5}
    for s in K5:
        strat_dir = os.path.join(sd, s)
        if not os.path.isdir(strat_dir): continue
        for m in sorted(glob.glob(os.path.join(strat_dir, 'r*_meta.json'))):
            try: d = json.load(open(m))
            except: continue
            vp = d.get('val_pearson')
            if vp is None: continue
            t = d.get('train_time_sec')
            if t is None or t < 5: continue
            try: z = np.load(m.replace('_meta.json', '.npz'))
            except: continue
            if z['val_pred'].shape != vy.shape or z['test_pred'].shape != oy.shape: continue
            models_by_strat[s].append({
                'val_pearson': float(vp), 'train_time': float(t),
                'val_pred': z['val_pred'], 'test_pred': z['test_pred'],
            })
    # Sort each strat's models by training order (rNN_*)
    return models_by_strat, vy, oy

def at_budget(models_by_strat, T_per_strat):
    """Return list of models accumulated by each strategy until cumulative time exceeds T."""
    pool = []
    for s, models in models_by_strat.items():
        cum = 0.0
        for m in models:
            if cum + m['train_time'] > T_per_strat: break
            pool.append({'strategy': s, **m})
            cum += m['train_time']
    return pool

def enet_score(pool, vy, oy):
    if not pool: return None, []
    V = np.array([p['val_pred'] for p in pool]).T
    T = np.array([p['test_pred'] for p in pool]).T
    en = ElasticNetCV(l1_ratio=[0.5, 0.9, 0.95, 1.0], positive=True, cv=min(5, len(pool)), max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0]), en.coef_

T_GRID = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]  # GPU-hours per strat

for D, cells in CELLS.items():
    rows = []
    for cell, seed in cells:
        res = load_models(cell, seed)
        if res is None: continue
        mbs, vy, oy = res
        if not any(mbs.values()): continue
        for T_h in T_GRID:
            T_s = T_h * 3600
            pool = at_budget(mbs, T_s)
            if not pool: continue
            score, weights = enet_score(pool, vy, oy)
            if score is None: continue
            # Strategy composition by ENet weight
            comp = {s: 0.0 for s in K5}
            tot_w = sum(weights) + 1e-12
            for p, w in zip(pool, weights):
                comp[p['strategy']] += float(w) / tot_w
            # Optimal K = nonzero weights
            K_opt = int((weights > 1e-9).sum())
            rows.append({'cell': cell, 'seed': seed, 'T_h': T_h, 'score': score,
                         'K_opt': K_opt, 'n_pool': len(pool), **comp})
    # 4-panel figure: (a) oracle_r vs T, (b) K_opt vs T, (c) strategy composition stack, (d) per-cell curves
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    # (a) per-cell oracle_r vs T
    ax = axes[0, 0]
    for cell, seed in cells:
        cs = f'{cell}/{seed}'
        df = [r for r in rows if r['cell']==cell and r['seed']==seed]
        if df: ax.plot([r['T_h'] for r in df], [r['score'] for r in df], 'o-', alpha=0.7, label=cs[:30])
    # Mean
    Ts = sorted({r['T_h'] for r in rows})
    means = [np.mean([r['score'] for r in rows if r['T_h']==t]) for t in Ts]
    ax.plot(Ts, means, 'k-', lw=4, label='MEAN', alpha=0.85)
    ax.set_xscale('log'); ax.set_xlabel('GPU-hours per strategy'); ax.set_ylabel('oracle_r')
    ax.set_title(f'D={D//1000}k — Ensemble quality vs time budget per strategy')
    ax.legend(fontsize=8, loc='lower right', ncol=2); ax.grid(True, alpha=0.3)
    # (b) K_opt vs T
    ax = axes[0, 1]
    for cell, seed in cells:
        df = [r for r in rows if r['cell']==cell and r['seed']==seed]
        if df: ax.plot([r['T_h'] for r in df], [r['K_opt'] for r in df], 'o-', alpha=0.7)
    means_k = [np.mean([r['K_opt'] for r in rows if r['T_h']==t]) for t in Ts]
    ax.plot(Ts, means_k, 'k-', lw=4, label='MEAN', alpha=0.85)
    ax.set_xscale('log'); ax.set_xlabel('GPU-hours per strategy'); ax.set_ylabel('K_opt (# non-zero in ensemble)')
    ax.set_title(f'D={D//1000}k — Optimal ensemble size vs time budget')
    ax.legend(); ax.grid(True, alpha=0.3)
    # (c) strategy composition stack (mean across cells)
    ax = axes[1, 0]
    comp_means = {s: [np.mean([r[s] for r in rows if r['T_h']==t]) for t in Ts] for s in K5}
    ax.stackplot(Ts, *[comp_means[s] for s in K5], labels=K5, alpha=0.85)
    ax.set_xscale('log'); ax.set_xlabel('GPU-hours per strategy'); ax.set_ylabel('ensemble weight fraction')
    ax.set_title(f'D={D//1000}k — Strategy composition of optimal ensemble vs time')
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.0)
    # (d) n_pool vs T (how many models each strat got)
    ax = axes[1, 1]
    n_means = [np.mean([r['n_pool'] for r in rows if r['T_h']==t]) for t in Ts]
    ax.plot(Ts, n_means, 'o-', color='#5080d0', lw=2.5, label='total pool size')
    ax2 = ax.twinx()
    ax2.plot(Ts, [r/5 for r in n_means], 'o--', color='#d05050', lw=2, label='avg models per strat')
    ax.set_xscale('log'); ax.set_xlabel('GPU-hours per strategy'); ax.set_ylabel('total pool size', color='#5080d0')
    ax2.set_ylabel('avg models per strategy', color='#d05050')
    ax.set_title(f'D={D//1000}k — Pool size vs time')
    ax.grid(True, alpha=0.3)
    fig.suptitle(f'Time-adjusted analysis at D={D//1000}k — fairly compare strategies by wall-clock per strategy', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'time_adjusted_d{D}.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print(f'saved time_adjusted_d{D}.png ({len(rows)} rows)')
