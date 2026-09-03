"""PI-meeting slides: test/eval options (decision) + 3 reference charts (HP axes, reservoirs,
acquisition). Parameters reflect the current code (scaling_hp_search.py search space; albench
reservoir/acquisition samplers). -> pi_meeting_figs/"""

import os
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")
os.makedirs(OUT, exist_ok=True)


def table_slide(
    fname,
    tag,
    title,
    header_note,
    rows,
    colw,
    hi_row=None,
    footer=None,
    fontsize=9.2,
    note_y=0.83,
    tbl_rect=(0.03, 0.06, 0.94, 0.62),
):
    fig = plt.figure(figsize=(13, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.955, tag, fontsize=12, weight="bold", color="#2563eb")
    fig.text(0.04, 0.905, title, fontsize=20, weight="bold", color="0.1")
    if header_note:
        fig.text(0.04, note_y, header_note, fontsize=11, color="0.2", va="top")
    ax = fig.add_axes(list(tbl_rect))
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.82")
        cell.set_width(colw[c])
        if r == 0:
            cell.set_facecolor("#1e3a8a")
            cell.set_text_props(color="white", weight="bold")
        elif hi_row and r == hi_row:
            cell.set_facecolor("#dbeafe")
    if footer:
        fig.text(0.04, 0.035, footer, fontsize=10, style="italic", color="0.3", wrap=True)
    fig.savefig(f"{OUT}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", fname)


# ---- 1. TEST / EVAL options (decision slide) ----
table_slide(
    "test_set_options.png",
    "DECISION (for PI feedback)",
    "Which test / eval set?",
    "Finding: two scaling regimes — in-dist + SNV scale ~-0.28, but OOD + structural scale FASTER "
    "(~-0.35). A single number hides this profile.",
    [
        ["Option", "What it is", "Pros", "Cons"],
        [
            "1. Single genomic test",
            "One in-dist held-out set",
            "Simple, one number",
            "Hides variant/OOD/structural behavior",
        ],
        [
            "2. Battery, equal-weight\n   per regime",
            "genomic + SNV + OOD +\nstructural, weighted equally",
            "Full profile; unbiased across\nregimes; drives informativeness map",
            "More bookkeeping",
        ],
        [
            "3. Battery + LOCK-BOX",
            "battery + a sequestered slice\nscored only at the end",
            "Guards against method-overfit;\npublishable rigor",
            "Costs some held-out data",
        ],
        [
            "4. Deployment-weighted",
            "weight regimes by how much\nwe care at deployment",
            "Matches real priorities",
            "Weights are a subjective choice",
        ],
    ],
    colw=[0.17, 0.28, 0.28, 0.27],
    hi_row=2,
    footer="Leaning: battery + LOCK-BOX, equal-weighted across regimes for selection. Reference slide "
    "lists every test set (n, provenance). Open for PI input.",
)

# ---- 2. HP axes reference ----
table_slide(
    "hp_axes_reference.png",
    "REFERENCE",
    "HP-optimization search space (current)",
    "15 core axes searched by every strategy + 7 optional 'novel' axes (nv1). Env-overridable.",
    [
        ["Axis", "Range / options", "Axis", "Range / options"],
        ["lr", "log-uniform 1e-5 – 1e-2", "kernel size (ks)", "{3,5,7,9,11} (per-layer opt.)"],
        ["batch_size", "D-aware menu", "pct_start (warmup)", "{0.1,0.2,0.3,0.4}"],
        ["conv_dropout", "0 – 0.3", "optimizer", "{adam, adamw, muon}"],
        ["dense_dropout", "0 – 0.5", "weight_decay", "log-uniform 1e-6 – 1e-2"],
        ["n_layers", "2 – 12", "pool_downsample", "{0,1,2,3,4}"],
        ["width_base", "{16,32,64,128,256}", "shift-aug / shift_max", "on/off; {5,10,15,20}"],
        ["width_jitter", "per-layer 0.5 – 2×", "lr_schedule", "plateau/onecycle/cosine/…"],
        ["block_class", "{eff, ag, plain}", "NOVEL (nv1)", "activation{silu,relu,gelu,quickgelu,"],
        ["", "", "", "mish,elu}; loss{mse,huber,smoothl1};"],
        ["", "", "", "se_reduction 2–16; rev-comp; evoaug"],
    ],
    colw=[0.15, 0.35, 0.18, 0.32],
    fontsize=8.8,
    footer="Validated: ensemble knee ~4-5; search plateau ~50 rounds; ~98% early-stop by epoch ~30. "
    "Open: which axes to freeze (fix backbone, span reg?) and whether to enable novel axes by default.",
)

# ---- 3. Reservoir reference ----
table_slide(
    "reservoir_reference.png",
    "REFERENCE",
    "Reservoir strategies (generate candidate sequences)",
    "5 tested at D=10k/30k/100k (below). Framework has many more (activity-stratified, curriculum, "
    "in-silico-evolution, phylogenetic, mixed-pool, motif-clustering, …).",
    [
        ["Reservoir", "How it generates", "Role / note"],
        [
            "genomic",
            "real K562-MPRA chr-train sequences (Gosai 2023)",
            "the in-distribution reference",
        ],
        [
            "motif_planted_v2",
            "plants regulatory motifs into backgrounds",
            "strong signal; val inflates +0.14",
        ],
        [
            "dinuc_shuffle",
            "dinucleotide-preserving shuffle of real seqs",
            "composition-matched negative-ish",
        ],
        [
            "evoaug_heavy",
            "EvoAug evolutionary augmentation (heavy)",
            "highest per-model score in tests",
        ],
        ["gc_matched", "GC-content-matched random sequences", "GC-controlled baseline"],
    ],
    colw=[0.18, 0.50, 0.32],
    fontsize=9.5,
    footer="Finding: sequence SOURCE sets the LEVEL (offset), not the scaling rate. Open: right set of 5? "
    "which to MEASURE experimentally with collaborators?",
)

# ---- 4. Acquisition reference ----
table_slide(
    "acquisition_reference.png",
    "REFERENCE",
    "Acquisition strategies (select a subset from a pool)",
    "Applied on top of a reservoir to choose which D sequences to label/train on.",
    [
        ["Acquisition", "How it selects", "Note"],
        ["full-pool", "no selection — use the reservoir's D as-is", "baseline"],
        [
            "uncertainty_guided",
            "picks high-uncertainty / high-attribution sequences\n(mutagenesis-scored)",
            "targets informative regions",
        ],
        [
            "diversity_guided",
            "k-center farthest-first on a k-mer(4) frequency\nembedding",
            "maximizes compositional spread",
        ],
    ],
    colw=[0.20, 0.52, 0.28],
    fontsize=9.5,
    tbl_rect=(0.03, 0.20, 0.94, 0.45),
    footer="Finding: the menu already GENERALIZES across acquisition (LOSO gaps: uncertainty -0.010, "
    "diversity -0.003) → likely second-order. Open: which to prioritize testing in the R×A grid?",
)

print("DONE — 4 slides")
