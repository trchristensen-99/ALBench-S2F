"""Assemble the PI-meeting figure pack: one figure per page, big simple title +
one-line takeaway. Clean, direct style (PI preference). -> pi_meeting_figs/figure_pack.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

SLOPE = os.path.expanduser("~/Downloads/slope_experiment_figs")
PIF = os.path.expanduser("~/Downloads/pi_meeting_figs")

# ordered: (path, section-tag, big title, one-line takeaway)
PAGES = [
    (f"{PIF}/schematic_pipeline.png", "SETUP",
     "The pipeline", "Oracle labels → generate → select → search → ensemble → evaluate; three independent axes."),
    (f"{SLOPE}/fig_scaling.png", "RESULT 1",
     "Error follows a power law in data",
     "Test-error drops as a straight line in log–log; the slope is shared across sequence sources."),
    (f"{SLOPE}/slope_variance.png", "RESULT 1b",
     "Source sets the LEVEL, target sets the RATE",
     "Reservoirs differ only in offset (same slope); OOD/structural targets scale faster (−0.35 vs −0.28)."),
    (f"{PIF}/schematic_informativeness.png", "GOAL",
     "Informativeness = which sequences to measure",
     "Per eval-target, the lowest curve wins → tells collaborators which sequences buy the most accuracy."),
    (f"{SLOPE}/selection_regret.png", "RESULT 2",
     "The ensemble reaches ~the best possible model",
     "Val-selected ensemble matches/beats the best single model everywhere except OOD (+0.02–0.04)."),
    (f"{SLOPE}/loso.png", "RESULT 3",
     "One reweightable menu works across strategies",
     "Leave-one-reservoir-out: the shared menu ≈ each reservoir's own best (6/7; gap = coverage, not bias)."),
    (f"{SLOPE}/fig_ensemble_knee.png", "METHOD",
     "Ensembles plateau at ~4–5 models",
     "Diminishing returns past ~4–5 models — small, cheap ensembles capture the gain."),
    (f"{SLOPE}/fig_rounds_plateau.png", "METHOD",
     "HP search plateaus at ~50 rounds",
     "Beyond ~50 proposals the best-so-far flattens — sets the search budget."),
    (f"{SLOPE}/fig_overfit_by_reservoir.png", "VAL-SET",
     "Val–test gaps are distribution shift, not overfitting",
     "Gap ≈0 in-distribution, large on OOD/motif → the val set must match the target you care about."),
]

pdf_path = f"{PIF}/figure_pack.pdf"
with PdfPages(pdf_path) as pdf:
    for path, tag, title, caption in PAGES:
        fig = plt.figure(figsize=(11, 8.5))  # landscape slide
        fig.patch.set_facecolor("white")
        # header band
        fig.text(0.06, 0.945, tag, fontsize=12, weight="bold", color="#2563eb")
        fig.text(0.06, 0.895, title, fontsize=22, weight="bold", color="0.1")
        # image
        ax = fig.add_axes([0.06, 0.14, 0.88, 0.70])
        ax.axis("off")
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, f"[missing: {os.path.basename(path)}]", ha="center", va="center",
                    fontsize=14, color="red")
        # caption
        fig.text(0.06, 0.065, caption, fontsize=13.5, color="0.25", style="italic", wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # val-set options: already a fully-composed slide -> add full-bleed (no extra title band)
    vpath = f"{PIF}/val_set_options.png"
    if os.path.exists(vpath):
        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(mpimg.imread(vpath))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print("WROTE", pdf_path, f"({len(PAGES) + 1} pages)")
