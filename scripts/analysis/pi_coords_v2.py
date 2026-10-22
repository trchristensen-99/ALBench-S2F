"""Per-(D, reservoir) HP coordinate plots, current 13-axis space. Three views each:
  3a all (gray) vs best-ensemble (nonzero ENet weight, red)
  3b all (gray) vs BEST-5 greedy ensemble members (red)
  3c geometric-gradient subsample, MORE worse models, dark=better
Titles include dataset size + reservoir.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as ml
import matplotlib.patheffects as pe
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

plt.rcParams.update({'font.size':14,'axes.titlesize':18,'figure.titlesize':19})
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

# cells: (label, D, reservoir, list of source dirs)
CELLS=[
 ('genomic','30k','genomic',['outputs/overnight/combo_d30k_genomic_seed42','outputs/overnight/combo_d30k_genomic_seed43','outputs/overnight/fresh_d30k_genomic_seed42','outputs/overnight/fresh_d30k_genomic_seed43']),
 ('genomic','10k','genomic',['outputs/overnight/combo_d10k_genomic','outputs/overnight/fresh_d10000_genomic_seed42']),
 ('genomic','100k','genomic',['outputs/overnight/combo_d100k_genomic','outputs/overnight/fresh_d100000_genomic_seed42']),
 ('genomic','300k','genomic',['outputs/overnight/combo_d300k_genomic','outputs/overnight/fresh_d300000_genomic_seed42']),
 ('motif','30k','motif_planted_v2',['outputs/overnight/combo_d30k_motif_seed42','outputs/overnight/combo_d30k_motif_seed43','outputs/overnight/fresh_d30k_motif_seed42']),
 ('dinuc','30k','dinuc_shuffle',['outputs/overnight/combo_d30k_dinuc_seed42','outputs/overnight/combo_d30k_dinuc_seed43','outputs/overnight/fresh_d30k_dinuc_seed42']),
]

def load_cell(dirs):
    rows=[]; labels=None
    for sd in dirs:
        if not os.path.isdir(sd): continue
        lab=np.load(os.path.join(sd,'labels.npz')) if os.path.exists(os.path.join(sd,'labels.npz')) else None
        for m in sorted(glob.glob(os.path.join(sd,'r*_meta.json'))):
            try: d=json.load(open(m))
            except: continue
            vp=d.get('val_pearson')
            if vp is None or not np.isfinite(float(vp)): continue
            hp=d.get('hp',{}); ex=hp.get('extra',{}) or {}
            rec={'val_pearson':float(vp)}
            _mse=d.get('val_mse')
            try: rec['val_mse']=float(_mse)
            except: rec['val_mse']=np.nan
            for k,_ in AXES:
                if k in ('val_pearson','val_mse'): continue
                rec[k]=ex.get(k,hp.get(k))
            try:
                z=np.load(m.replace('_meta.json','.npz'))
                if lab is not None and z['val_pred'].shape==lab['val_labels'].shape and np.all(np.isfinite(z['val_pred'])) and np.all(np.isfinite(z['test_pred'])):
                    rec['val_pred']=z['val_pred']; rec['test_pred']=z['test_pred']; rec['lab']=lab
            except: pass
            rows.append(rec)
    return rows

REVMAP={k:{v:kk for kk,v in d.items()} for k,d in CATMAP.items()}
def draw_axis_ticks(ax, cols, AXES):
    for j,(key,scale) in enumerate(AXES):
        arr=cols[key]; fin=arr[np.isfinite(arr)]
        if len(fin)==0: continue
        lo,hi=fin.min(),fin.max()
        if key in CATMAP:
            for lab,code in CATMAP[key].items():
                yy=(code/2.0-lo)/(hi-lo) if hi>lo else 0.5
                ax.text(j-0.06, yy, lab, ha='right', va='center', fontsize=9, color='black', fontweight='bold', zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
            continue
        if scale=='lin' and key in ('n_layers','ks','skip_stride'):
            import math
            vals=sorted(set(int(round(v)) for v in np.linspace(lo,hi,6)))
            for v in vals:
                yy=(v-lo)/(hi-lo) if hi>lo else 0.5
                ax.text(j-0.06, yy, str(v), ha='right', va='center', fontsize=9, color='black', fontweight='bold', zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
            continue
        # continuous: 5 ticks
        for f in [0,0.25,0.5,0.75,1.0]:
            t=lo+f*(hi-lo); real=10**t if scale=="log" else t
            lab=f"{real:.1e}" if scale=="log" else (f"{real:.2f}" if real<10 else f"{int(round(real))}")
            ax.text(j-0.06, f, lab, ha='right', va='center', fontsize=9, color='black', fontweight='bold', zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

def make_plots(clabel,D,res,rows):
    if len(rows)<20: 
        print(f'skip {res} D={D}: only {len(rows)} models'); return
    cols={}
    for key,scale in AXES:
        vals=[]
        for r in rows:
            v=r.get(key)
            if v is None: vals.append(np.nan); continue
            if scale=='cat': vals.append(CATMAP.get(key,{}).get(str(v),0)/2.0); continue
            try: v=float(v)
            except: vals.append(np.nan); continue
            if scale=='log' and v>0: v=np.log10(v)
            vals.append(v)
        cols[key]=np.array(vals)
    def yfor(r):
        ys=[]
        for key,scale in AXES:
            v=r.get(key)
            if v is None: ys.append(0.5); continue
            if scale=='cat': v=CATMAP.get(key,{}).get(str(v),0)/2.0
            else:
                try: v=float(v)
                except: ys.append(0.5); continue
                if scale=='log' and v>0: v=np.log10(v)
            fin=cols[key][np.isfinite(cols[key])]; lo,hi=fin.min(),fin.max()
            ys.append((v-lo)/(hi-lo) if hi>lo else 0.5)
        return ys
    xs=np.arange(len(AXES))
    def setup(ax,title):
        for j in xs: ax.axvline(j,color='gray',alpha=0.4,lw=0.6)
        ax.set_xticks(xs); _ltxt=ax.set_xticklabels([NICE[k] for k,_ in AXES],fontsize=13)
        ax.axvline(len(AXES)-2-0.5,color='#333333',lw=1.6,ls='--',alpha=0.75,zorder=5)
        for _t in _ltxt[-2:]: _t.set_color('#1a5fb4'); _t.set_fontweight('bold')
        ax.set_yticks([]); ax.set_ylim(-0.05,1.05); ax.set_xlim(-0.7,len(AXES)-0.7)
        ax.set_title(title,fontweight='bold')
        draw_axis_ticks(ax, cols, AXES)
    # ensemble
    good=[r for r in rows if 'val_pred' in r]
    nonzero=[]; best5=[]
    if good:
        lab=good[0]['lab']; V=np.array([r['val_pred'] for r in good]).T; T=np.array([r['test_pred'] for r in good]).T
        en=ElasticNetCV(l1_ratio=[0.5,0.9,0.95,1.0],positive=True,cv=5,max_iter=20000,n_jobs=1); en.fit(V,lab['val_labels'])
        w=en.coef_; order=np.argsort(-w)
        nonzero=[good[i] for i in order if w[i]>1e-9]
        best5=[good[i] for i in order[:5] if w[i]>1e-9]
        orc=pearsonr(en.predict(T),lab['test_oracle'])[0]
        print(f'{res} D={D}: n={len(rows)}, {len(nonzero)} nonzero, oracle_r={orc:.4f}')
    nz_ids=set(id(r) for r in nonzero); b5_ids=set(id(r) for r in best5)
    tag=f'{res}_d{D}'
    ttl=f'{res}, D={D}'
    # 3a all vs best-ensemble (3-tier stratified: all / other ensemble members / top-5 greedy)
    fig,ax=plt.subplots(figsize=(18,7.5)); setup(ax,f'All models (gray, n={len(rows)})  vs  other ensemble members (orange, n={max(len(nonzero)-len(best5),0)})  vs  top-5 greedy ensemble (red)  --  {ttl}')
    for r in rows:
        if id(r) not in nz_ids: ax.plot(xs,yfor(r),color='#cccccc',alpha=0.07,lw=0.6,zorder=1)
    for r in nonzero:
        if id(r) not in b5_ids: ax.plot(xs,yfor(r),color='#f5a623',alpha=0.55,lw=1.4,zorder=2)
    for r in best5: ax.plot(xs,yfor(r),color='#d01010',alpha=0.95,lw=2.6,zorder=4)
    ax.legend([ml.Line2D([],[],color='#cccccc',lw=3),ml.Line2D([],[],color='#f5a623',lw=3),ml.Line2D([],[],color='#d01010',lw=3)],['all models','other ensemble members','top-5 greedy ensemble'],loc='upper left',bbox_to_anchor=(1.005,1.0),borderaxespad=0.,fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f'4a_{tag}.png'),dpi=140,bbox_inches='tight'); plt.close(fig)
    # 3b all vs best-5
    fig,ax=plt.subplots(figsize=(18,7.5)); setup(ax,f'All models (gray, n={len(rows)}) vs optimal 5-member ensemble (red)  --  {ttl}')
    for r in rows:
        if id(r) not in b5_ids: ax.plot(xs,yfor(r),color='#bbbbbb',alpha=0.08,lw=0.6,zorder=1)
    for r in [x for x in rows if id(x) in b5_ids]: ax.plot(xs,yfor(r),color='#d01010',alpha=0.9,lw=2.6,zorder=3)
    ax.legend([ml.Line2D([],[],color='#bbbbbb',lw=3),ml.Line2D([],[],color='#d01010',lw=3)],['all models','top-5 greedy ensemble'],loc='upper left',bbox_to_anchor=(1.005,1.0),borderaxespad=0.,fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f'4b_{tag}.png'),dpi=140,bbox_inches='tight'); plt.close(fig)
    # 3c geometric gradient with MORE worse models
    rs=sorted(rows,key=lambda r:-r['val_pearson']); n=len(rs)
    def band(lo,hi,k):
        b=rs[lo:min(hi,n)]
        if len(b)<=k: return b
        return [b[i] for i in np.linspace(0,len(b)-1,k).astype(int)]
    sub=rs[:15]+band(15,100,20)+band(100,400,20)+band(400,n,25)  # more worse models
    vps=np.array([r['val_pearson'] for r in sub]); nc=Normalize(vmin=np.nanpercentile(vps,5),vmax=np.nanpercentile(vps,95)); cmapC=cm.get_cmap('viridis_r')
    fig,ax=plt.subplots(figsize=(18,7.5)); setup(ax,f'Performance-gradient subsample (n={len(sub)}: top-15 + log-spaced bands, dark=better)  --  {ttl}')
    for i in np.argsort(vps): ax.plot(xs,yfor(sub[i]),color=cmapC(nc(vps[i])),lw=2.0,alpha=0.75)
    sm=cm.ScalarMappable(cmap=cmapC,norm=nc); sm.set_array([]); cb=fig.colorbar(sm,ax=ax,pad=0.01,aspect=30); cb.set_label('val Pearson',rotation=270,labelpad=20,fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f'4c_{tag}.png'),dpi=140,bbox_inches='tight'); plt.close(fig)

for clabel,D,res,dirs in CELLS:
    rows=load_cell(dirs)
    make_plots(clabel,D,res,rows)
print('done')
