"""Full PI deck slides: decision register + 5 summary/decision slides + reference charts
(reservoir & acquisition rebuilt with the FULL strategy taxonomy). -> pi_meeting_figs/"""
import os
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")
os.makedirs(OUT, exist_ok=True)


def slide(fname, tag, title, note, rows, colw, hi=None, footer=None, fs=9.2,
          note_y=0.83, rect=(0.03, 0.06, 0.94, 0.62), note_fs=11):
    fig = plt.figure(figsize=(13, 8.5)); fig.patch.set_facecolor("white")
    fig.text(0.04, 0.955, tag, fontsize=12, weight="bold", color="#2563eb")
    fig.text(0.04, 0.905, title, fontsize=19, weight="bold", color="0.1")
    if note:
        fig.text(0.04, note_y, note, fontsize=note_fs, color="0.2", va="top")
    ax = fig.add_axes(list(rect)); ax.axis("off")
    t = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.85)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("0.82"); cell.set_width(colw[c])
        if r == 0:
            cell.set_facecolor("#1e3a8a"); cell.set_text_props(color="white", weight="bold")
        elif hi and r in hi:
            cell.set_facecolor("#dbeafe")
    if footer:
        fig.text(0.04, 0.035, footer, fontsize=10, style="italic", color="0.3", wrap=True)
    fig.savefig(f"{OUT}/{fname}", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("WROTE", fname)


# ---- decision register (the map that lets the PI pick where to dive) ----
slide("decision_register.png", "OVERVIEW", "Decision register — where your input is needed",
      "Two axes are genuinely OPEN (val, test); the rest are largely validated. Pick where to dive; "
      "full detail in the reference slides at the end.",
      [["Axis", "Current", "Status", "Question for you"],
       ["Val set", "per-cell chr-val / holdout", "OPEN ⭐", "target-matched vs common vs mixed?"],
       ["Test / eval", "battery (in-dist/SNV/OOD/struct)", "OPEN ⭐", "which regimes + how to weight?"],
       ["HP-opt", "15+7 axes; knee~4; ~50 rounds", "validated", "freeze which axes? novel axes on?"],
       ["Reservoir", "5 tested of ~8+ families", "studied", "which to prioritize / measure?"],
       ["Acquisition", "3 tested of ~6 families", "LOSO-validated", "which to prioritize testing?"]],
      colw=[0.13, 0.29, 0.16, 0.42], hi=[1, 2], fs=10.5,
      footer="Feedback priority: 1 Val → 2 Test-weighting → 3 HP-freeze → 4 Reservoir-for-wetlab → 5 Acquisition.")

# ---- HP summary (decision) ----
slide("hp_summary.png", "DECISION", "HP-optimization ranges — what to freeze?",
      "Validated: ensemble knee ~4–5; search plateaus ~50 rounds; ~98% early-stop by epoch ~30; "
      "backbone stable across reservoirs, fine regularization idiosyncratic. Full space → reference slide.",
      [["Option", "Freeze", "Keep searching"],
       ["A. Fix backbone, span reg (rec.)", "block_class family + one wide width range",
        "lr, dropout, weight_decay, schedule"],
       ["B. Freeze more", "+ optimizer, kernel size", "lr + regularization only"],
       ["C. Keep all searchable", "nothing", "everything (most compute)"]],
      colw=[0.26, 0.36, 0.36], hi=[1], fs=10,
      footer="Plus: enable NOVEL axes (activation/loss/skip/rev-comp) by default? AutoResearch may exploit them.")

# ---- reservoir summary (decision) ----
slide("reservoir_summary.png", "DECISION", "Reservoir strategies — which to prioritize?",
      "Finding: sequence SOURCE sets the LEVEL (offset), not the scaling rate. We have ~8+ families "
      "(+ variants); 5 tested at all D. Full taxonomy → reference slide.",
      [["Family", "Tested?", "Priority rationale"],
       ["Genomic", "✓ all D", "the real-sequence deployment target"],
       ["Random (+variants: GC, dinuc)", "✓ all D", "controls / negatives"],
       ["TF shuffling / planting", "✓ all D", "strong signal (val inflates)"],
       ["EvoAug augmentation", "✓ all D", "highest per-model score"],
       ["Uncertainty-guided / Phylogenetic", "✓ D=30k", "active + evolutionary signal"],
       ["PRM, Generative, In-silico-evolution", "planned", "novel candidates for the wet-lab"]],
      colw=[0.34, 0.14, 0.52], fs=9.6,
      footer="Question: which families to run full HP-opt on, and which are most informative to MEASURE experimentally?")

# ---- acquisition summary (decision) ----
slide("acquisition_summary.png", "DECISION", "Acquisition strategies — which to prioritize?",
      "Finding: the menu already GENERALIZES across acquisition (LOSO gaps ~0) → likely second-order. "
      "We have ~6 families (many methods each); 3 tested. Full taxonomy → reference slide.",
      [["Family", "Methods", "Tested?"],
       ["Random", "uniform subset", "✓ (= full pool)"],
       ["Uncertainty", "attribution / uncertainty scoring", "✓"],
       ["Diversity", "LCMD, k-mer spectrum, motif-diversity, embedding", "✓ (kmer/emb)"],
       ["Uncertainty + Diversity (batch)", "BatchBALD, BADGE, BAIT", "planned"],
       ["Priors", "motif, activity", "planned"],
       ["Expected error", "EER", "planned"]],
      colw=[0.28, 0.50, 0.22], fs=9.6,
      footer="Question: which to prioritize in the R×A grid, given the menu looks second-order for design?")

# ---- reservoir REFERENCE (full taxonomy) ----
slide("reservoir_reference.png", "REFERENCE", "Reservoir strategies — full menu",
      "Generate candidate sequences. ✓all D = in the slope experiment (5); ✓D=30k = bake-off only.",
      [["Family", "Examples / variants", "Tested?"],
       ["Genomic", "real K562-MPRA chr-train (Gosai 2023)", "✓ all D"],
       ["Random (+variants)", "uniform, GC-matched, dinuc-shuffle", "✓ all D (gc, dinuc)"],
       ["PRM (+variants)", "prm_10pct, prm_1pct, …", "planned"],
       ["TF shuffling / planting", "motif-planted v2, motif-shuffled, grammar", "✓ all D (v2)"],
       ["EvoAug augmentation", "evoaug_heavy, evoaug_structural", "✓ all D (heavy)"],
       ["Uncertainty-guided", "attribution / uncertainty mutagenesis", "✓ D=30k"],
       ["Phylogenetic variation", "zoonomia-based", "✓ D=30k"],
       ["Generative", "student/oracle-guided generation", "planned"],
       ["In-silico evolution", "evolutionary sequence optimization", "planned"]],
      colw=[0.22, 0.50, 0.28], fs=9.4,
      footer="Finding: source = LEVEL (offset), not rate. Open: right set + which to measure experimentally.")

# ---- acquisition REFERENCE (full taxonomy) ----
slide("acquisition_reference.png", "REFERENCE", "Acquisition strategies — full menu",
      "Select which D sequences to label/train on, on top of a reservoir.",
      [["Family", "Methods", "Tested?"],
       ["Random", "uniform subset (= full pool)", "✓"],
       ["Uncertainty", "attribution / uncertainty scoring; mutagenesis", "✓"],
       ["Diversity", "LCMD, k-mer spectrum, motif diversity, embedding (k-center)", "✓ (kmer/emb)"],
       ["Uncertainty + Diversity", "BatchBALD, BADGE, BAIT (batch AL)", "planned"],
       ["Priors", "motif priors, activity priors", "planned"],
       ["Expected error", "expected error reduction (EER)", "planned"]],
      colw=[0.24, 0.54, 0.22], fs=9.6, rect=(0.03, 0.14, 0.94, 0.54),
      footer="Finding: menu generalizes across acquisition (LOSO uncertainty -0.010, diversity -0.003) → second-order.")

print("DONE deck slides")
