"""Three coordinate-plot variants of the HP distributions (D=30k genomic, current HP space):
  (A) all models (gray) + best-ensemble members (red)  [existing]
  (B) ONLY the optimal 5-member ensemble (greedy top-5 by ENet weight)
  (C) geometric subsample with performance color-gradient (dark=better):
      top 10 by val_pearson, +10 from ranks 11-100, +10 from 101-1000, +10 from 1000+.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as ml
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

plt.rcParams.update({'font.size':14,'axes.titlesize':18,'figure.titlesize':20})
OUT='outputs/analysis/pi_update'; os.makedirs(OUT,exist_ok=True)
SRCS=['outputs/overnight/combo_d30k_genomic_seed42','outputs/overnight/combo_d30k_genomic_seed43',
      'outputs/overnight/fresh_d30k_genomic_seed42','outputs/overnight/fresh_d30k_genomic_seed43']
AXES=[('lr','log'),('batch_size','log'),('weight_decay','log'),('conv_dropout','lin'),
      ('dense_dropout','lin'),('n_layers','lin'),('width_base','log'),('ks','lin'),
      ('block_class','cat'),('optimizer','cat'),('activation','cat'),
      ('outer_skip_style','cat'),('skip_stride','lin')]
CATMAP={'block_class':{'eff':0,'ag':1,'plain':2},'optimizer':{'adam':0,'adamw':1,'muon':2},
        'activation':{'silu':0,'gelu':1,'relu':2},'outer_skip_style':{'concat':0,'add':1,'none':2}}
LABELS_NICE={'lr':'lr','batch_size':'batch\nsize','weight_decay':'weight\ndecay','conv_dropout':'conv\ndropout',
 'dense_dropout':'dense\ndropout','n_layers':'n\nlayers','width_base':'width\nbase','ks':'ks',
 'block_class':'block\nclass','optimizer':'optim','activation':'activ','outer_skip_style':'skip\nstyle','skip_stride':'skip\nstride'}

rows=[]; labels=None
for sd in SRCS:
    if not os.path.isdir(sd): continue
    lab=None
    if os.path.exists(os.path.join(sd,'labels.npz')): lab=np.load(os.path.join(sd,'labels.npz'))
    for m in sorted(glob.glob(os.path.join(sd,'r*_meta.json'))):
        try: d=json.load(open(m))
        except: continue
        vp=d.get('val_pearson')
        if vp is None: continue
        hp=d.get('hp',{}); ex=hp.get('extra',{}) or {}
        rec={'val_pearson':float(vp),'src':sd}
        for k,_ in AXES: rec[k]=ex.get(k,hp.get(k))
        try:
            z=np.load(m.replace('_meta.json','.npz'))
            if lab is not None and z['val_pred'].shape==lab['val_labels'].shape and np.all(np.isfinite(z['val_pred'])) and np.all(np.isfinite(z['test_pred'])):
                rec['val_pred']=z['val_pred']; rec['test_pred']=z['test_pred']; rec['lab']=lab
        except: pass
        rows.append(rec)
print(f'loaded {len(rows)} models')

def norm_val(r,key,scale):
    v=r.get(key)
    if v is None: return np.nan
    if scale=='cat': return CATMAP.get(key,{}).get(str(v),0)/2.0
    try: v=float(v)
    except: return np.nan
    return v
# per-axis normalization ranges
cols={}
for key,scale in AXES:
    vals=[]
    for r in rows:
        v=norm_val(r,key,scale)
        if scale=='log' and v==v and v>0: v=np.log10(v)
        vals.append(v)
    cols[key]=np.array(vals)
def yfor(r):
    ys=[]
    for key,scale in AXES:
        v=norm_val(r,key,scale)
        if scale=='log' and v==v and v>0: v=np.log10(v)
        fin=cols[key][np.isfinite(cols[key])]; lo,hi=fin.min(),fin.max()
        ys.append((v-lo)/(hi-lo) if hi>lo else 0.5)
    return ys
xs=np.arange(len(AXES))
def setup_ax(ax,title):
    for j in xs: ax.axvline(j,color='gray',alpha=0.4,lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([LABELS_NICE[k] for k,_ in AXES],fontsize=13)
    ax.set_yticks([]); ax.set_ylim(-0.05,1.05); ax.set_xlim(-0.3,len(AXES)-0.7)
    ax.set_title(title,fontweight='bold')

# ENet ensemble -> best-5 (greedy by weight) + all nonzero
best5=[]; nonzero=[]
good=[r for r in rows if 'val_pred' in r]
if good:
    lab=good[0]['lab']; V=np.array([r['val_pred'] for r in good]).T; T=np.array([r['test_pred'] for r in good]).T
    en=ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0],positive=True,cv=5,max_iter=20000,n_jobs=1); en.fit(V,lab['val_labels'])
    w=en.coef_; order=np.argsort(-w)
    best5=[good[i] for i in order[:5] if w[i]>1e-9]
    nonzero=[good[i] for i in order if w[i]>1e-9]
    orc=pearsonr(en.predict(T),lab['test_oracle'])[0]
    print(f'ensemble oracle_r={orc:.4f}, {len(nonzero)} nonzero, best5 weights={[round(float(w[i]),3) for i in order[:5]]}')

# (B) ONLY the best-5 ensemble
figB,axB=plt.subplots(figsize=(18,7))
setup_ax(axB,f'Optimal 5-member ensemble — HP configs (D=30k genomic)')
cmapB=cm.get_cmap('viridis')
for rank,r in enumerate(best5):
    axB.plot(xs,yfor(r),color=cmapB(rank/max(len(best5)-1,1)),lw=3.2,alpha=0.9,marker='o',ms=7,label=f'member {rank+1} (val r={r["val_pearson"]:.3f})')
axB.legend(loc='upper right',fontsize=12)
figB.tight_layout(); figB.savefig(os.path.join(OUT,'hp_coords_best5_only.png'),dpi=140,bbox_inches='tight'); print('saved best5_only')

# (C) geometric subsample with performance gradient
rows_sorted=sorted([r for r in rows], key=lambda r:-r['val_pearson'])
n=len(rows_sorted)
def sample_band(lo,hi,k):
    band=rows_sorted[lo:min(hi,n)]
    if len(band)<=k: return band
    idx=np.linspace(0,len(band)-1,k).astype(int)
    return [band[i] for i in idx]
sub=[]
sub+= rows_sorted[:10]                    # top 10
sub+= sample_band(10,100,10)              # 10 from ranks 11-100
sub+= sample_band(100,1000,10)            # 10 from 101-1000
sub+= sample_band(1000,n,10)              # 10 from 1000+
# color by val_pearson (dark=better)
vps=np.array([r['val_pearson'] for r in sub])
normC=Normalize(vmin=vps.min(),vmax=vps.max())
cmapC=cm.get_cmap('viridis_r')  # reversed: high val -> dark
figC,axC=plt.subplots(figsize=(18,7.5))
setup_ax(axC,f'Geometric subsample by performance ({len(sub)} models: top-10 + log-spaced bands)\ndarker = higher val Pearson')
order_c=np.argsort(vps)  # plot worst first
for i in order_c:
    axC.plot(xs,yfor(sub[i]),color=cmapC(normC(vps[i])),lw=2.2,alpha=0.8)
sm=cm.ScalarMappable(cmap=cmapC,norm=normC); sm.set_array([])
cb=figC.colorbar(sm,ax=axC,pad=0.01,aspect=30); cb.set_label('val Pearson',rotation=270,labelpad=20,fontsize=13)
figC.tight_layout(); figC.savefig(os.path.join(OUT,'hp_coords_geometric_gradient.png'),dpi=140,bbox_inches='tight'); print('saved geometric_gradient')

# (A) refreshed all-vs-best with larger fonts + wrapped labels
figA,axA=plt.subplots(figsize=(18,7.5))
setup_ax(axA,f'All models (gray, n={len(rows)}) vs best-ensemble members (red, n={len(nonzero)}) — D=30k genomic')
for r in rows:
    if r not in nonzero: axA.plot(xs,yfor(r),color='#bbbbbb',alpha=0.08,lw=0.6,zorder=1)
for r in nonzero:
    axA.plot(xs,yfor(r),color='#d01010',alpha=0.8,lw=2.2,zorder=3)
axA.legend([ml.Line2D([],[],color='#bbbbbb',lw=3),ml.Line2D([],[],color='#d01010',lw=3)],['all models','best-ensemble members'],loc='upper right',fontsize=13)
figA.tight_layout(); figA.savefig(os.path.join(OUT,'hp_coords_all_vs_best.png'),dpi=140,bbox_inches='tight'); print('saved all_vs_best (refreshed)')
