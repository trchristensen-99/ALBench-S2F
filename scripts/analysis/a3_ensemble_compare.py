"""A3 ensemble comparison:
(a) Top-5 single (uniform mean of val_pred / test_pred)
(b) Top-5 single (ElasticNetCV fit on those 5 only)
(c) Greedy ENet (ALL models, ElasticNetCV — current production)

Across same cells used in A1-A3. Tests whether the greedy ENet's complementarity
selection is actually beating simple top-K averaging.
"""
import glob, json, os, numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

E100 = 'outputs/hp_step1_bakeoff_e100'
K5 = ['evo_single', 'evo_batch', 'optuna_gp', 'optuna_tpe', 'llm_explore_nv1']
BUDGET = 75

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
            rows.append({'val_pearson': vp, 'val_pred': z['val_pred'], 'test_pred': z['test_pred']})
    return rows, vy, oy

def enet_score(V, T, vy, oy):
    en = ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0], positive=True, cv=5, max_iter=20000, n_jobs=1)
    en.fit(V, vy)
    return float(pearsonr(en.predict(T), oy)[0]), int((en.coef_ > 1e-9).sum())

def uniform_score(T, oy):
    pred = T.mean(axis=1)
    return float(pearsonr(pred, oy)[0])

print(f'{"cell/seed":42s}  {"N":>4s}  {"top1":>8s}  {"top5_mean":>10s}  {"top5_enet":>10s}  {"all_enet":>10s}  {"all_nz":>6s}')
all_scores = []
for cell, seed in CELLS:
    cd = os.path.join(E100, cell, seed)
    if not os.path.isdir(cd): continue
    rows, vy, oy = load_cell(cd)
    if not rows: continue
    # Sort by val_pearson desc
    rows.sort(key=lambda r: -r['val_pearson'])
    top1_test = pearsonr(rows[0]['test_pred'], oy)[0]
    # Top-5 single — uniform mean
    top5 = rows[:5]
    T5 = np.array([r['test_pred'] for r in top5]).T
    V5 = np.array([r['val_pred'] for r in top5]).T
    top5_mean = uniform_score(T5, oy)
    top5_enet, top5_enet_nz = enet_score(V5, T5, vy, oy)
    # All-ENet
    V = np.array([r['val_pred'] for r in rows]).T
    T = np.array([r['test_pred'] for r in rows]).T
    all_enet, all_nz = enet_score(V, T, vy, oy)
    short = cell.replace('k562_','').replace('_planted_v2','PvX').replace('_shuffle','S')
    print(f'{short+"/"+seed:42s}  {len(rows):4d}  {top1_test:8.4f}  {top5_mean:10.4f}  {top5_enet:10.4f}  {all_enet:10.4f}  {all_nz:6d}')
    all_scores.append((top1_test, top5_mean, top5_enet, all_enet))
arr = np.array(all_scores)
print(f'\n{"MEAN across cells":42s}  {"":>4s}  {arr[:,0].mean():8.4f}  {arr[:,1].mean():10.4f}  {arr[:,2].mean():10.4f}  {arr[:,3].mean():10.4f}')
print(f'{"Δ vs top1 (single best)":42s}  {"":>4s}  {"--":>8s}  {arr[:,1].mean()-arr[:,0].mean():+10.4f}  {arr[:,2].mean()-arr[:,0].mean():+10.4f}  {arr[:,3].mean()-arr[:,0].mean():+10.4f}')
print(f'{"Δ vs top5_enet (best 5-only ENet)":42s}  {"":>4s}  {"--":>8s}  {arr[:,1].mean()-arr[:,2].mean():+10.4f}  {"":>10s}  {arr[:,3].mean()-arr[:,2].mean():+10.4f}')
