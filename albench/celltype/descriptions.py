"""What we tell a model about a cell type, and how that becomes a vector.

A conditioning scheme that takes a free per-cell embedding can only ever reach cell types it
was trained on. A scheme that reads a *description* can in principle reach any cell type
somebody can describe, which is the only route to the primary-cell and arbitrary-cell-type
use we care about. So the descriptions are data, and they are written here rather than
generated, because what is in them determines what the model can possibly transfer.

Each entry carries structured fields and a free-text ``description`` built from them. The
structured fields exist so a scheme can be given a clean one-hot or profile view of the same
facts; the text exists so a frozen sentence encoder can be given the facts as prose.

One honest limitation, stated here because it is easy to forget downstream: every line below
is an immortalised cancer or transformed line. ``immortalised`` and ``malignant`` are
therefore CONSTANT across the training set, so a model cannot learn what they mean and
cannot extrapolate along those axes. They are recorded anyway, because the moment one
primary-cell dataset enters training the axis starts to vary and the descriptions already
say which side each cell sits on. Until then, any claim about primary-cell transfer is
untested by construction -- see ``docs`` and the session notes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class CellLine:
    """One cell type, as both structured fields and prose."""

    name: str
    tissue: str
    lineage: str
    disease: str
    immortalised: bool
    malignant: bool
    sex: str
    karyotype: str
    key_factors: tuple[str, ...]
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def description(self) -> str:
        state = (
            "an immortalised malignant cell line"
            if self.malignant
            else (
                "an immortalised non-malignant cell line"
                if self.immortalised
                else "a primary, non-immortalised cell"
            )
        )
        return (
            f"{self.name} is {state} derived from {self.tissue}. "
            f"Its lineage is {self.lineage}. Disease of origin: {self.disease}. "
            f"Donor sex: {self.sex}. Karyotype: {self.karyotype}. "
            f"Transcription factors and regulators characteristic of this cell type include "
            f"{', '.join(self.key_factors)}. {self.notes}"
        ).strip()


def _c(**kw) -> CellLine:
    return CellLine(**kw)


CELL_LINES: dict[str, CellLine] = {
    "K562": _c(
        name="K562",
        tissue="bone marrow of a patient with chronic myelogenous leukaemia",
        lineage="erythroleukaemic, an erythroid and megakaryocytic progenitor-like state",
        disease="chronic myelogenous leukaemia, BCR-ABL1 positive",
        immortalised=True,
        malignant=True,
        sex="female",
        karyotype="near-triploid, highly rearranged",
        key_factors=("GATA1", "TAL1", "KLF1", "NFE2", "LMO2", "GATA2"),
        notes="It expresses haemoglobin and is the standard erythroid reporter-assay host.",
    ),
    "HepG2": _c(
        name="HepG2",
        tissue="a liver hepatocellular carcinoma in a 15-year-old male",
        lineage="hepatocyte-like, endoderm derived",
        disease="hepatoblastoma or hepatocellular carcinoma",
        immortalised=True,
        malignant=True,
        sex="male",
        karyotype="hyperdiploid, relatively stable",
        key_factors=("HNF4A", "HNF1A", "FOXA1", "FOXA2", "CEBPA", "NR1H4"),
        notes="It secretes albumin and other plasma proteins and is the standard hepatic host.",
    ),
    "SK-N-SH": _c(
        name="SK-N-SH",
        tissue="a bone marrow metastasis of a neuroblastoma in a 4-year-old",
        lineage="neuronal and neural-crest derived, sympathetic adrenergic",
        disease="neuroblastoma",
        immortalised=True,
        malignant=True,
        sex="female",
        karyotype="near-diploid with 1p and 17q alterations",
        key_factors=("PHOX2B", "HAND2", "GATA3", "ASCL1", "TBX2", "MYCN"),
        notes="It contains both neuroblastic and substrate-adherent subpopulations.",
        aliases=("SKNSH", "SK_N_SH"),
    ),
    "A549": _c(
        name="A549",
        tissue="a lung adenocarcinoma in a 58-year-old male",
        lineage="alveolar type II pneumocyte-like, epithelial",
        disease="lung adenocarcinoma, KRAS G12S and STK11 mutant",
        immortalised=True,
        malignant=True,
        sex="male",
        karyotype="hypotriploid",
        key_factors=("NKX2-1", "FOXA2", "NFE2L2", "CEBPB", "TEAD1"),
        notes="It is glucocorticoid responsive and widely used for NR3C1 studies.",
    ),
    "GM12878": _c(
        name="GM12878",
        tissue="peripheral blood of a healthy female donor",
        lineage="B-lymphoblastoid, Epstein-Barr-virus transformed",
        disease="none; the donor was healthy",
        immortalised=True,
        malignant=False,
        sex="female",
        karyotype="normal diploid",
        key_factors=("PAX5", "EBF1", "SPI1", "POU2F2", "IRF4", "NFKB1"),
        notes=(
            "It is transformed by EBV rather than by a tumour, so it is the closest line here "
            "to a normal primary cell and the only karyotypically normal one."
        ),
    ),
    "HCT116": _c(
        name="HCT116",
        tissue="a colorectal carcinoma in an adult male",
        lineage="colonic epithelial, endoderm derived",
        disease="colorectal carcinoma, mismatch-repair deficient, KRAS G13D mutant",
        immortalised=True,
        malignant=True,
        sex="male",
        karyotype="near-diploid, microsatellite unstable",
        key_factors=("CDX2", "TCF7L2", "HNF4A", "KLF5", "TP53"),
        notes="It is near-diploid and TP53 wild type, unusually for a carcinoma line.",
    ),
    "Jurkat": _c(
        name="Jurkat",
        tissue="peripheral blood of a 14-year-old male with T-cell leukaemia",
        lineage="T-lymphocyte, CD4 positive mature T-cell-like",
        disease="acute T-cell lymphoblastic leukaemia",
        immortalised=True,
        malignant=True,
        sex="male",
        karyotype="near-diploid with PTEN loss",
        key_factors=("TCF7", "LEF1", "GATA3", "RUNX1", "ETS1", "NOTCH1"),
        notes="It signals through the T-cell receptor and is the standard T-cell host.",
    ),
    # Not in training yet: the deposition holds reads only. Declared so that the moment the
    # fastqs are processed the description axis has a non-immortalised value in it.
    "HMEC": _c(
        name="HMEC",
        tissue="reduction mammoplasty tissue from a healthy donor",
        lineage="mammary luminal and basal epithelial",
        disease="none; the donor was healthy",
        immortalised=False,
        malignant=False,
        sex="female",
        karyotype="normal diploid",
        key_factors=("TP63", "GRHL2", "KLF5", "ESR1", "GATA3"),
        notes="A finite-lifespan primary culture, not a cell line.",
        aliases=("mammary epithelial cell",),
    ),
    "HEK293": _c(
        name="HEK293",
        tissue="embryonic kidney of a female fetus",
        lineage="adrenal and neuronal-like despite the kidney origin",
        disease="none; transformed with sheared adenovirus 5 DNA",
        immortalised=True,
        malignant=False,
        sex="female",
        karyotype="hypotriploid",
        key_factors=("PAX2", "LHX1", "SOX11", "POU3F2"),
        notes="Transformed by adenoviral E1A and E1B rather than by a tumour.",
    ),
}

_ALIAS = {a.lower(): k for k, v in CELL_LINES.items() for a in (*v.aliases, k)}


def canonical(name: str) -> str:
    """Map any spelling we have seen in a data file onto a key of ``CELL_LINES``."""
    key = _ALIAS.get(str(name).strip().lower())
    if key is None:
        raise KeyError(f"Unknown cell type {name!r}. Known: {sorted(CELL_LINES)}")
    return key


def describe(name: str) -> str:
    return CELL_LINES[canonical(name)].description


def structured_matrix(cells: list[str]) -> np.ndarray:
    """A hand-built numeric view of the same facts, for a conditioning scheme that wants
    features rather than prose. Columns are interpretable on purpose: this is the ablation
    that says how much of a text embedding's value is in a handful of categorical facts."""
    lineages = sorted({CELL_LINES[canonical(c)].lineage.split(",")[0] for c in cells})
    tissues = sorted({CELL_LINES[canonical(c)].tissue.split()[-1] for c in cells})
    rows = []
    for c in cells:
        cl = CELL_LINES[canonical(c)]
        row = [float(cl.immortalised), float(cl.malignant), float(cl.sex == "female")]
        row += [float(cl.lineage.split(",")[0] == x) for x in lineages]
        row += [float(cl.tissue.split()[-1] == x) for x in tissues]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _cache_path(cells: list[str], model: str) -> str:
    from albench.core import paths

    tag = hashlib.sha1(("|".join(cells) + "::" + model).encode()).hexdigest()[:16]
    d = os.path.join(str(paths.data_root()), "celltype_text")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"text_emb_{tag}.npz")


def text_embeddings(
    cells: list[str], *, model: str = TEXT_ENCODER, cache: bool = True
) -> np.ndarray:
    """Frozen sentence embeddings of each cell's description, mean-pooled and L2-normalised.

    Deliberately frozen and deliberately generic: the point of the text route is that a cell
    type we have never measured can be reached by writing a paragraph about it, which only
    holds if the encoder was not fitted to our cells. Cached because it is the same handful
    of strings every run, and the cache key covers the cell list and the model name so a
    changed description cannot be served stale -- the description text itself is hashed in.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = [describe(c) for c in cells]
    key = hashlib.sha1("||".join(texts).encode()).hexdigest()[:12]
    path = _cache_path([*cells, key], model)
    if cache and os.path.exists(path):
        z = np.load(path)
        return z["emb"]

    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).eval()
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        out = net(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    arr = emb.numpy().astype(np.float32)
    if cache:
        np.savez(path, emb=arr, cells=np.array(cells, dtype=object))
    return arr
