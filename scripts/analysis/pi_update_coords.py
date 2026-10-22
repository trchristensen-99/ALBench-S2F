"""PI-update coordinate plot: HP distributions of ALL models vs the BEST-ENSEMBLE
members, across every searched HP axis. Uses combined-newspace data (D=30k genomic,
which has all 16 axes incl. activation/skip_style/skip_stride).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

plt.rcParams.update({'font.size':13,'axes.titlesize':17,'figure.titlesize':19})
OUT='outputs/analysis/pi_update'; os.makedirs(OUT,exist_ok=True)

# Pool the D=30k genomic combined-newspace cells (both seeds)
SRCS=['outputs/overnight/combo_d30k_genomic_seed42','outputs/overnight/combo_d30k_genomic_seed43']
AXES=[('lr','log'),('batch_size','log'),('weight_decay','log'),('conv_dropout','lin'),
      ('dense_dropout','lin'),('n_layers','lin'),('width_base','log'),('ks','lin'),
      ('block_class','cat'),('optimizer','cat'),('activation','cat'),
      ('outer_skip_style','cat'),('skip_stride','lin')]
CATMAP={'block_class':{'eff':0,'ag':1,'plain':2},'optimizer':{'adam':0,'adamw':1,'muon':2},
        'activation':{'silu':0,'gelu':1,'relu':2},'outer_skip_style':{'concat':0,'add':1,'none':2}}

rows=[]
labels=None
for sd in SRCS:
    # need val_pred/test_pred for ensemble fit -> load labels.npz
    lab=None
    for f in glob.glob(os.path.join(sd,'labels.npz')):
        lab=np.load(f); break
    metas=sorted(glob.glob(os.path.join(sd,'r*_meta.json')))
    for m in metas:
        try: d=json.load(open(m))
        except: continue
        vp=d.get('val_pearson')
        if vp is None: continue
        hp=d.get('hp',{}); ex=hp.get('extra',{}) or {}
        rec={'val_pearson':float(vp),'meta':m}
        for k,_ in AXES:
            v = ex.get(k, hp.get(k))
            rec[k]=v
        # load preds for ensemble
        try:
            z=np.load(m.replace('_meta.json','.npz'))
            rec['val_pred']=z['val_pred']; rec['test_pred']=z['test_pred']
        except: rec['val_pred']=None
        rows.append(rec)
    if lab is not None and labels is None: labels=lab

print(f'loaded {len(rows)} models')
# Fit ENet ensemble to identify best-ensemble members (nonzero weight)
best_idx=set()
if labels is not None:
    good=[r for r in rows if r.get('val_pred') is not None and r['val_pred'].shape==labels['val_labels'].shape and np.all(np.isfinite(r['val_pred'])) and np.all(np.isfinite(r['test_pred']))]
    if good:
        V=np.array([r['val_pred'] for r in good]).T; T=np.array([r['test_pred'] for r in good]).T
        vy=labels['val_labels']; oy=labels['test_oracle']
        en=ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0],positive=True,cv=5,max_iter=20000,n_jobs=1); en.fit(V,vy)
        w=en.coef_
        order=np.argsort(-w)
        for i in order[:15]:
            if w[i]>1e-9: best_idx.add(id(good[i]))
        print(f'ensemble: {int((w>1e-9).sum())} nonzero, top-15 flagged; oracle_r={pearsonr(en.predict(T),oy)[0]:.4f}')

def norm_val(r,key,scale):
    v=r.get(key)
    if v is None: return np.nan
    if scale=='cat': v=CATMAP.get(key,{}).get(str(v),0); return v/2.0
    try: v=float(v)
    except: return np.nan
    return v

# Build normalized matrix + plot: gray=all, colored=best-ensemble
n_ax=len(AXES); xs=np.arange(n_ax)
fig,ax=plt.subplots(figsize=(20,8))
# compute per-axis min/max for normalization (log for log axes)
cols={}
for j,(key,scale) in enumerate(AXES):
    vals=[]
    for r in rows:
        v=norm_val(r,key,scale)
        if scale=='log' and v==v and v>0: v=np.log10(v)
        vals.append(v)
    cols[key]=np.array(vals)
def plot_row(r,idx,color,alpha,lw,z):
    ys=[]
    for j,(key,scale) in enumerate(AXES):
        v=norm_val(r,key,scale)
        arr=cols[key]; fin=arr[np.isfinite(arr)]
        if scale=='log' and v==v and v>0: v=np.log10(v)
        lo,hi=fin.min(),fin.max()
        ys.append((v-lo)/(hi-lo) if hi>lo else 0.5)
    ax.plot(xs,ys,color=color,alpha=alpha,lw=lw,zorder=z)
for r in rows:
    if id(r) not in best_idx: plot_row(r,None,'#bbbbbb',0.10,0.6,1)
for r in rows:
    if id(r) in best_idx: plot_row(r,None,'#d01010',0.85,2.2,3)
for j in xs: ax.axvline(j,color='gray',alpha=0.4,lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels([k for k,_ in AXES],rotation=35,ha='right',fontsize=13)
ax.set_yticks([]); ax.set_ylim(-0.05,1.05); ax.set_xlim(-0.3,n_ax-0.7)
ax.set_title(f'HP distributions: ALL models (gray, n={len(rows)}) vs BEST-ENSEMBLE members (red, n={len(best_idx)})\nD=30k genomic, combined new HP space',fontweight='bold')
import matplotlib.lines as ml
ax.legend([ml.Line2D([],[],color='#bbbbbb',lw=3),ml.Line2D([],[],color='#d01010',lw=3)],
          ['all models','best-ensemble members'],loc='upper right',fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'hp_coords_all_vs_best.png'),dpi=140,bbox_inches='tight')
print('saved hp_coords_all_vs_best.png')
