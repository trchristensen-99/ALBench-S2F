"""Canonical-anchor (ON) vs random-init (OFF): per-strategy individual results + ensemble."""
import glob, json, os
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

ARMS = {'ON (canonical seed)': 'outputs/overnight/anchor_ab_exploit_d30k_genomic',
        'OFF (random init)':   'outputs/overnight/anchor_ab_none_d30k_genomic'}

def _is_anchor(hp):
    ex = hp.get('extra', {}) or {}
    def g(k, d=None): return ex.get(k, hp.get(k, d))
    try:
        return (abs(float(g('width_ratio', 0) or 0) - 2.0) < 1e-6
                and int(g('pool_downsample', 0) or 0) == 4
                and abs(float(hp.get('weight_decay', 0)) - 0.1) < 1e-6)
    except Exception:
        return False

def load(d):
    L = np.load(os.path.join(d, 'labels.npz')); oy = L['test_oracle']; vy = L['val_labels']
    rows = []
    for m in sorted(glob.glob(os.path.join(d, 'r*_meta.json'))):
        try: dd = json.load(open(m)); z = np.load(m.replace('_meta.json', '.npz'))
        except Exception: continue
        if 'test_pred' not in z or 'val_pred' not in z: continue
        if z['test_pred'].shape != oy.shape or z['val_pred'].shape != vy.shape: continue
        rows.append({'strategy': dd.get('strategy'), 'val_pred': z['val_pred'], 'test_pred': z['test_pred'],
                     'oracle_r': float(pearsonr(z['test_pred'], oy)[0]), 'is_anchor': _is_anchor(dd.get('hp', {}))})
    return rows, vy, oy

def top5(rows, vy, oy):
    V = np.array([r['val_pred'] for r in rows]).T
    en = ElasticNetCV(l1_ratio=[.5,.9,.95,1.], positive=True, cv=5, max_iter=20000, n_jobs=1).fit(V, vy)
    w = en.coef_; sel = [i for i in np.argsort(-w)[:5] if w[i] > 1e-9]
    tp = np.mean([rows[i]['test_pred'] for i in sel], axis=0)
    return sel, float(pearsonr(tp, oy)[0])

out = []
data = {}
for name, d in ARMS.items():
    if not os.path.isdir(d): out.append(f'{name}: MISSING ({d})'); continue
    rows, vy, oy = load(d)
    if len(rows) < 5: out.append(f'{name}: only {len(rows)} models'); continue
    sel, ens = top5(rows, vy, oy); data[name] = (rows, sel, ens)

strategies = sorted({r['strategy'] for (rows,_,_) in data.values() for r in rows})
out.append('=== INDIVIDUAL results per HP strategy (test-set oracle_r) ===')
out.append(f'{"strategy":16s} | ' + ' | '.join(f'{n:24s}' for n in data))
out.append(f'{"":16s} | ' + ' | '.join('  best     mean    n   ' for _ in data))
for s in strategies:
    cells = []
    for name, (rows, _, _) in data.items():
        sr = [r['oracle_r'] for r in rows if r['strategy'] == s]
        cells.append(f'{max(sr):.4f}  {np.mean(sr):.4f}  {len(sr):3d}' if sr else '   -        -      0 ')
    out.append(f'{s:16s} | ' + ' | '.join(f'{c:24s}' for c in cells))
out.append('')
out.append('=== ENSEMBLE (top-5 greedy) oracle_r ===')
for name, (rows, sel, ens) in data.items():
    out.append(f'  {name:24s}: {ens:.4f}   (n_models={len(rows)})')
if len(data) == 2:
    (n1, e1), (n2, e2) = [(n, v[2]) for n, v in data.items()]
    out.append(f'  ensemble delta ({n1} - {n2}): {e1-e2:+.4f}')
out.append('')
out.append('=== ANCHOR (canonical config) in the ON arm ===')
on = [v for k, v in data.items() if 'ON' in k]
if on:
    rows, sel, _ = on[0]; ind = np.array([r['oracle_r'] for r in rows])
    ai = [i for i, r in enumerate(rows) if r['is_anchor']]
    if ai:
        a = ai[0]; out.append(f'  anchor oracle_r={rows[a]["oracle_r"]:.4f}  rank={int((ind>rows[a]["oracle_r"]).sum())+1}/{len(rows)}  selected_in_top5={a in sel}')
    else:
        out.append('  (anchor not yet trained / not found)')
rep = '\n'.join(out)
print(rep)
os.makedirs('outputs/analysis', exist_ok=True)
open('outputs/analysis/anchor_effect_report.txt', 'w').write(rep + '\n')
print('\n[written to outputs/analysis/anchor_effect_report.txt]')
