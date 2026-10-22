"""Analyze the NEW architectural axes (activation, outer_skip_style, skip_stride)
from the combined-newspace + archnew ablation data.

For each new axis, show: does it converge in top models? Does it contribute
ensemble diversity? Which values win?
"""
import glob, json, os
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'outputs/analysis/new_axes'
os.makedirs(OUT, exist_ok=True)

# Sources with the new axes present (extra dict)
SOURCES = {
    'combo_d10k': 'outputs/overnight/combo_d10k_genomic',
    'combo_d30k_gen': 'outputs/overnight/combo_d30k_genomic_seed42',
    'combo_d30k_motif': 'outputs/overnight/combo_d30k_motif_seed42',
    'combo_d30k_dinuc': 'outputs/overnight/combo_d30k_dinuc_seed42',
    'combo_d100k': 'outputs/overnight/combo_d100k_genomic',
    'archnew_d10k': 'outputs/overnight/archnew_d10000',
    'archnew_d30k': 'outputs/overnight/archnew_d30000',
    'archnew_d100k': 'outputs/overnight/archnew_d100000',
}
NEW_AXES = ['activation', 'outer_skip_style', 'skip_stride']

def load(path):
    rows = []
    for m in sorted(glob.glob(os.path.join(path, 'r*_meta.json'))):
        try: d = json.load(open(m))
        except: continue
        vp = d.get('val_pearson')
        if vp is None: continue
        extra = d.get('hp', {}).get('extra', {}) or {}
        rows.append({'val_pearson': float(vp), 'extra': extra})
    return rows

# Aggregate: for each new axis, top-30 by val_pearson, value distribution
print('=== New-axis value distribution among TOP-30 models per source ===')
agg = {ax: Counter() for ax in NEW_AXES}
agg_all = {ax: Counter() for ax in NEW_AXES}
n_total = 0
for name, path in SOURCES.items():
    rows = load(path)
    if not rows: continue
    n_total += len(rows)
    rows.sort(key=lambda r: -r['val_pearson'])
    top = rows[:30]
    for ax in NEW_AXES:
        for r in top:
            v = r['extra'].get(ax)
            if v is not None: agg[ax][str(v)] += 1
        for r in rows:
            v = r['extra'].get(ax)
            if v is not None: agg_all[ax][str(v)] += 1

print(f'total models scanned: {n_total}')
for ax in NEW_AXES:
    print(f'\n{ax}:')
    print(f'  TOP-30 pooled: {dict(agg[ax].most_common())}')
    print(f'  ALL pooled:    {dict(agg_all[ax].most_common())}')
    # Enrichment = top-30 fraction / all fraction (>1 means winning)
    tot_top = sum(agg[ax].values()) or 1
    tot_all = sum(agg_all[ax].values()) or 1
    print(f'  Enrichment (top vs all):')
    for v in sorted(set(agg[ax]) | set(agg_all[ax])):
        ft = agg[ax][v]/tot_top; fa = agg_all[ax][v]/tot_all
        enr = ft/fa if fa > 0 else 0
        flag = ' <-- enriched' if enr > 1.15 else (' <-- depleted' if enr < 0.85 else '')
        print(f'    {v:10s} top={ft:.3f} all={fa:.3f} enrichment={enr:.2f}{flag}')

# Bar plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_i, axis in zip(axes, NEW_AXES):
    vals = sorted(set(agg[axis]) | set(agg_all[axis]))
    tot_top = sum(agg[axis].values()) or 1
    tot_all = sum(agg_all[axis].values()) or 1
    top_fr = [agg[axis][v]/tot_top for v in vals]
    all_fr = [agg_all[axis][v]/tot_all for v in vals]
    x = np.arange(len(vals))
    ax_i.bar(x-0.2, all_fr, 0.4, label='all models', color='#aaa', edgecolor='k')
    ax_i.bar(x+0.2, top_fr, 0.4, label='top-30', color='#5080d0', edgecolor='k')
    ax_i.set_xticks(x); ax_i.set_xticklabels(vals, rotation=20)
    ax_i.set_title(axis); ax_i.set_ylabel('fraction'); ax_i.legend(); ax_i.grid(True, alpha=0.3)
fig.suptitle('New architectural axes: value distribution in top-30 vs all models', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'new_axes_enrichment.png'), dpi=140, bbox_inches='tight')
print('\nsaved new_axes_enrichment.png')
