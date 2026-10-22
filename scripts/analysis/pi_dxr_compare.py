"""D x R comparison: overlay the top-5 greedy-ensemble HP configs for several
(D, reservoir) cells on SHARED axes, so we can see how the optimal architecture
shifts with dataset size and reservoir. Shared absolute normalization per axis
(global min/max across all cells) so vertical position is comparable across cells.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as ml
import matplotlib.patheffects as pe
from sklearn.linear_model import ElasticNetCV

OUT='outputs/analysis/pi_update'; os.makedirs(OUT,exist_ok=True)
AXES=[('lr','log'),('batch_size','log'),('weight_decay','log'),('conv_dropout','lin'),
      ('dense_dropout','lin'),('n_layers','lin'),('width_base','log'),('ks','lin'),
      ('block_class','cat'),('optimizer','cat'),('activation','cat'),
      ('outer_skip_style','cat'),('skip_stride','lin'),
      ('val_pearson','lin'),('val_mse','lin')]
CATMAP={'block_class':{'eff':0,'ag':1,'plain':2},'optimizer':{'adam':0,'adamw':1,'muon':2},
        'activation':{'silu':0,'gelu':1,'relu':2},'outer_skip_style':{'concat':0,'add':1,'none':2}}
NICE={'lr':'lr','batch_size':'batch\nsize','weight_decay':'weight\ndecay','conv_dropout':'conv\ndropout',
 'dense_dropout':'dense\ndropout','n_layers':'n\nlayers','width_base':'width\nbase','ks':'ks',
 'block_class':'block\nclass','optimizer':'optim','activation':'activ','outer_skip_style':'skip\nstyle','skip_stride':'skip\nstride','val_pearson':'val\nPearson r','val_mse':'val\nMSE'}
# cells: (label, color, dirs)
CELLS=[
 ('genomic, D=30k','#1f77b4',['outputs/overnight/combo_d30k_genomic_seed42','outputs/overnight/combo_d30k_genomic_seed43','outputs/overnight/fresh_d30k_genomic_seed42','outputs/overnight/fresh_d30k_genomic_seed43']),
 ('dinuc-shuffle, D=30k','#d62728',['outputs/overnight/combo_d30k_dinuc_seed42','outputs/overnight/combo_d30k_dinuc_seed43','outputs/overnight/fresh_d30k_dinuc_seed42','outputs/overnight/fresh_d30k_dinuc_seed43']),
 ('motif-planted, D=30k','#2ca02c',['outputs/overnight/combo_d30k_motif_seed42','outputs/overnight/combo_d30k_motif_seed43','outputs/overnight/newspace_d30k_motif_seed42','outputs/overnight/newspace_d30k_motif_seed43']),
]
def raw(r,key,scale):
    v=r.get(key)
    if v is None: return np.nan
    if scale=='cat': return CATMAP.get(key,{}).get(str(v),0)/2.0
    try: v=float(v)
    except: return np.nan
    return np.log10(v) if (scale=='log' and v>0) else v
def load_top5(dirs):
    rows=[]; lab=None
    for sd in dirs:
        if not os.path.isdir(sd): continue
        L=np.load(sd+'/labels.npz') if os.path.exists(sd+'/labels.npz') else None
        for m in sorted(glob.glob(sd+'/r*_meta.json')):
            try:d=json.load(open(m))
            except:continue
            vp=d.get('val_pearson')
            if vp is None or not np.isfinite(float(vp)): continue
            hp=d.get('hp',{}); ex=hp.get('extra',{}) or {}
            rec={'vp':float(vp),'val_pearson':float(vp)}
            _m=d.get('val_mse')
            try: rec['val_mse']=float(_m)
            except: rec['val_mse']=np.nan
            for k,_ in AXES:
                if k in ('val_pearson','val_mse'): continue
                rec[k]=ex.get(k,hp.get(k))
            try:
                z=np.load(m.replace('_meta.json','.npz'))
                if L is not None and z['val_pred'].shape==L['val_labels'].shape and np.all(np.isfinite(z['val_pred'])) and np.all(np.isfinite(z['test_pred'])):
                    rec['vpred']=z['val_pred']; rec['tpred']=z['test_pred']; rec['lab']=L
            except: pass
            rows.append(rec)
    good=[r for r in rows if 'vpred' in r]
    if len(good)<6: return good[:5] if good else []
    L=good[0]['lab']; V=np.array([r['vpred'] for r in good]).T
    en=ElasticNetCV(l1_ratio=[.5,.9,.95,1.],positive=True,cv=5,max_iter=20000,n_jobs=1); en.fit(V,L['val_labels'])
    w=en.coef_; return [good[i] for i in np.argsort(-w)[:5] if w[i]>1e-9]
cell_top5=[(lab,col,load_top5(dirs)) for lab,col,dirs in CELLS]
for lab,col,t5 in cell_top5: print(f'{lab}: {len(t5)} top-5 members')
# global per-axis min/max across all members from all cells (shared normalization)
allm=[m for _,_,t5 in cell_top5 for m in t5]
gmin={}; gmax={}
for key,scale in AXES:
    vals=[raw(m,key,scale) for m in allm]; vals=[v for v in vals if np.isfinite(v)]
    gmin[key]=min(vals) if vals else 0; gmax[key]=max(vals) if vals else 1
def yfor(m):
    ys=[]
    for key,scale in AXES:
        v=raw(m,key,scale); lo,hi=gmin[key],gmax[key]
        ys.append((v-lo)/(hi-lo) if (np.isfinite(v) and hi>lo) else 0.5)
    return ys
xs=np.arange(len(AXES))
fig,ax=plt.subplots(figsize=(18,7.5))
for j in xs: ax.axvline(j,color='gray',alpha=0.4,lw=0.6)
for lab,col,t5 in cell_top5:
    for k,m in enumerate(t5):
        ax.plot(xs,yfor(m),color=col,lw=2.4,alpha=0.75,label=lab if k==0 else None,marker='o',ms=4)
# axis value ticks (shared scale), outlined
for j,(key,scale) in enumerate(AXES):
    lo,hi=gmin[key],gmax[key]
    if key in CATMAP:
        for nm,code in CATMAP[key].items():
            yy=(code/2.0-lo)/(hi-lo) if hi>lo else 0.5
            if -0.05<=yy<=1.05: ax.text(j-0.06,yy,nm,ha='right',va='center',fontsize=8.5,fontweight='bold',path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
    else:
        for f in [0,0.5,1.0]:
            t=lo+f*(hi-lo); real=10**t if scale=='log' else t
            s=f'{real:.1e}' if scale=='log' else (f'{real:.2f}' if real<10 else f'{int(round(real))}')
            ax.text(j-0.06,f,s,ha='right',va='center',fontsize=8.5,fontweight='bold',path_effects=[pe.withStroke(linewidth=2.5,foreground='white')])
ax.set_xticks(xs); _lt=ax.set_xticklabels([NICE[k] for k,_ in AXES],fontsize=13)
ax.axvline(len(AXES)-2-0.5,color='#333333',lw=1.6,ls='--',alpha=0.75,zorder=5)
for _t in _lt[-2:]: _t.set_color('#1a5fb4'); _t.set_fontweight('bold')
ax.set_yticks([]); ax.set_ylim(-0.05,1.08); ax.set_xlim(-0.7,len(AXES)-0.7)
ax.set_title('Top-5 greedy ensemble HP configs across reservoirs (R) at fixed D=30k -- shared axis scales',fontweight='bold',fontsize=16)
ax.legend(loc='upper left',bbox_to_anchor=(1.005,1.0),borderaxespad=0.,fontsize=12,framealpha=0.95)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'4c_DxR_top5_compare.png'),dpi=140,bbox_inches='tight')
print('saved 4c_DxR_top5_compare.png')
