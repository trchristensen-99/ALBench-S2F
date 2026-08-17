"""Val-set options slide for PI feedback — simple, direct table. -> pi_meeting_figs/"""
import os
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
fig.text(0.06, 0.945, "DECISION (for PI feedback)", fontsize=12, weight="bold", color="#2563eb")
fig.text(0.06, 0.895, "Which validation set?", fontsize=22, weight="bold", color="0.1")

# evidence banner
fig.text(0.06, 0.83,
         "Two of our results constrain this:\n"
         "  • Motif A/B: a val set from a non-target distribution is inflated & doesn't track the target (distributional, not leakage).\n"
         "  • Selection-regret: genomic val reaches best-possible on genomic/SNV/structural, but FAILS on OOD (can't pick best OOD model).",
         fontsize=11.5, color="0.2", va="top")

rows = [
    ["Option", "What it is", "Pros", "Cons"],
    ["1. Target-matched\n+ genomic default", "Select each target on a\nheld-out val from ITS OWN\ndistribution; genomic for the\nsingle deployable model",
     "Honest per-target;\nfixes OOD gap; best for\nthe informativeness map", "A val split per target;\nmore bookkeeping"],
    ["2. Common\ngenomic val", "One genomic chr-val\nfor everything", "Simplest, uniform,\none model", "Fails OOD; inflated for\nnon-genomic targets"],
    ["3. Mixed / diverse\nval", "One blended val across\ndistributions", "Robust; one model;\nless overfit to any one", "Optimal for none;\nblend weights arbitrary"],
    ["4. Leave-one-\ndistribution-out", "Select on all targets\nexcept the one tested", "Strongest anti-overfit;\ntests unseen target", "Pessimistic by design;\ncomplex"],
]
ax = fig.add_axes([0.04, 0.08, 0.92, 0.60]); ax.axis("off")
tbl = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 3.1)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.8")
    if r == 0:
        cell.set_facecolor("#1e3a8a"); cell.set_text_props(color="white", weight="bold")
    elif r == 1:
        cell.set_facecolor("#dbeafe")  # highlight recommended
col_w = [0.17, 0.30, 0.27, 0.26]
for (r, c), cell in tbl.get_celld().items():
    cell.set_width(col_w[c])

fig.text(0.06, 0.045,
         "Leaning: Option 1 (val distribution = target distribution) + a lock-box test to guard against method overfitting — "
         "the only choice consistent with BOTH results. Open for PI input.",
         fontsize=11, style="italic", color="0.25")
fig.savefig(f"{OUT}/val_set_options.png", dpi=150, bbox_inches="tight")
print("WROTE", f"{OUT}/val_set_options.png")
