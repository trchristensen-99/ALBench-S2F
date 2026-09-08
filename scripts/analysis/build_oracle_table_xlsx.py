"""Editable oracle-accuracy table (.xlsx): individual model out-of-fold vs the 10-fold ensemble.

Two quantities per evaluation set, each reported as both r and MSE:
  individual, out-of-fold   each sequence scored only by the fold that held it out - one model,
                            honestly evaluated.
  ensemble of 10            the 10-fold mean prediction, i.e. the deployed oracle. Note every
                            sequence was in 9 of the 10 folds' training data, so this is not a
                            held-out number.

r and MSE answer different questions and can disagree: r is scale-free and says whether the
ranking is right, MSE says whether the values are right. That distinction matters here because the
predictions are used as training targets, where absolute calibration counts.

All cells are real values, so font, column width and number format stay editable, and the block
pastes into PowerPoint or Slides as a native table.
"""

import argparse
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

HDR = PatternFill("solid", fgColor="1E293B")
BAND = PatternFill("solid", fgColor="F4F6F9")
thin = Side(style="thin", color="C8CFD8")

# (label, n/fold, label sd, ind_r, ind_r_sd, ind_mse, ens_r, ens_mse)
ROWS = [
    ("WT / genomic ref", 39143, 1.190, 0.9153, 0.0079, 0.2299, 0.9181, 0.2188),
    ("SNV alt allele", 39169, 1.184, 0.9156, 0.0074, 0.2274, 0.9192, 0.2117),
    ("Designed high-activity", 2296, 1.589, 0.8760, 0.0050, 0.5988, 0.8920, 0.5298),
    ("Negative controls (ctrl_neg)", 503, 0.491, 0.8465, None, 0.0720, None, None),
    ("SNV effect (alt - ref)", 35691, 0.470, 0.4015, 0.0228, 0.1860, 0.4038, 0.1931),
]
COLS = ["evaluation set", "n / fold", "label SD", "individual\nmodel: r",
        "individual\nmodel: SD", "individual\nmodel: MSE", "8-model\nensemble: r",
        "8-model\nensemble: MSE"]
WIDTH = [26, 10, 10, 12, 11, 12, 11, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=os.path.expanduser("~/Downloads/notion_updates/oracle_label_quality.xlsx")
    )
    ap.add_argument("--decimals", type=int, default=3)
    a = ap.parse_args()

    wb = Workbook()
    ws = wb.active
    ws.title = "Oracle label quality"
    fmt = "0." + "0" * a.decimals

    ws["A1"] = "Oracle label quality"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = "Held-out test-fold performance vs measured K562 activity"
    ws["A2"].font = Font(size=10, italic=True, color="666666")

    HROW = 4
    for j, c in enumerate(COLS, start=1):
        cell = ws.cell(HROW, j, c)
        cell.fill = HDR
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.alignment = Alignment(
            horizontal="left" if j == 1 else "right", vertical="center", wrap_text=True
        )
        cell.border = Border(bottom=thin)
    ws.row_dimensions[HROW].height = 30

    for i, (label, n, lsd, i_r, i_sd, i_m, e_r, e_m) in enumerate(ROWS):
        r = HROW + 1 + i
        vals = [label, n, lsd, i_r, i_sd, i_m, e_r, e_m]
        for j, v in enumerate(vals, start=1):
            cell = ws.cell(r, j, v)
            cell.font = Font(size=10, bold=(j in (4, 6)))
            if i % 2 == 1:
                cell.fill = BAND
            if j == 1:
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "#,##0" if j == 2 else fmt
        ws.cell(r, 1).border = Border(top=thin) if i == len(ROWS) - 1 else Border()

    # ensembling gain, live so edits propagate
    d = HROW + len(ROWS) + 2
    ws.cell(d, 1, "Ensembling gain (8 models - 1 model)").font = Font(bold=True, size=10)
    for i in range(len(ROWS)):
        src = HROW + 1 + i
        ws.cell(d + 1 + i, 1, ROWS[i][0]).font = Font(size=9.5)
        for col, f in ((7, f"=G{src}-D{src}"), (8, f"=H{src}-F{src}")):
            c = ws.cell(d + 1 + i, col, f)
            c.number_format = "+0." + "0" * a.decimals + ";-0." + "0" * a.decimals
            c.font = Font(size=9.5)
            c.alignment = Alignment(horizontal="right")

    note = (
        "Every number is measured on a HELD-OUT TEST FOLD - never the fold used for early "
        "stopping. Folds are 2-3 chromosomes of ref/alt sequences plus a random tenth of the "
        "designed sequences; for any fold, 8 of the 10 models trained on it, 1 validated on it, "
        "1 tested on it.\n"
        "individual model = mean over the 10 models, each on its own test fold; SD is the spread "
        "across those folds.\n"
        "8-model ensemble = 8 models trained on one fold's split with different seeds, averaged. "
        "This is the honest analogue of ensembling: a super-ensemble across folds is what the "
        "deployed oracle is, but it cannot be measured cleanly, since for any sequence 9 of the 10 "
        "models saw it in training.\n"
        "label SD is included because MSE is not comparable across sets with different dynamic "
        "ranges - SNV effect has a much narrower range (0.47) than the activity sets (1.2-2.0), "
        "so its low MSE reflects small targets rather than better prediction.\n"
        "Training: full encoder unfrozen, reverse-complement and native-context shift "
        "augmentation. val - test = +0.0003 across folds, so the val fold was NOT optimistic here."
    )
    nr = d + 3 + len(ROWS)
    ws.cell(nr, 1, note).font = Font(size=8.5, color="555555")
    ws.cell(nr, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=nr, start_column=1, end_row=nr + 6, end_column=8)

    for j, w in enumerate(WIDTH, start=1):
        ws.column_dimensions[chr(64 + j)].width = w

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    wb.save(a.out)
    print(f"wrote {a.out}")
    print(f"  paste A{HROW}:H{HROW + len(ROWS)} into Slides or PowerPoint as a native table")


if __name__ == "__main__":
    main()
