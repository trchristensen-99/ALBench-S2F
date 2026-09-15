"""Where data assets live, resolved without hardcoding anyone's filesystem.

Every external asset (reference genome, motif PFMs, phylogenetic rates, oracle
weights) is declared once in ``ASSETS`` with three things: the environment variable
that overrides it, a default location relative to the data root, and -- crucially --
HOW TO OBTAIN IT. That last field is what turns a missing-file crash into an
instruction a new collaborator can act on without asking anyone.

Resolution order for any asset, first hit wins:

  1. an explicit path passed by the caller
  2. the asset's environment variable          (e.g. ALBENCH_HG38)
  3. ``paths.local.yaml`` at the repo root     (git-ignored, per-machine)
  4. ``<ALBENCH_DATA>/<default_relpath>``      (ALBENCH_DATA defaults to ./data)
  5. any of the asset's ``fallbacks``          (site-specific known locations)

Nothing in this module may contain a path that only exists on one machine, except
inside ``fallbacks`` -- which is explicitly the place for "it happens to be here on
the CSHL cluster" and is always optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root derived from this file's location, never hardcoded.
REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Root for downloadable/derived data. Override with ALBENCH_DATA."""
    return Path(os.environ.get("ALBENCH_DATA", str(REPO_ROOT / "data"))).expanduser()


@dataclass(frozen=True)
class Asset:
    """One external file or directory the code may need."""

    key: str
    env_var: str
    default_relpath: str
    description: str
    how_to_get: str
    required_by: tuple[str, ...] = ()
    fallbacks: tuple[str, ...] = ()
    repo_relpaths: tuple[str, ...] = ()
    """Locations relative to the repo root. Several assets are produced by our own
    pipelines into ``outputs/`` rather than downloaded into ``data/``, so this is
    where a built-in-place artefact is found without any per-machine config."""
    is_dir: bool = False


ASSETS: dict[str, Asset] = {}


def _add(a: Asset) -> None:
    ASSETS[a.key] = a


_add(
    Asset(
        key="hg38",
        env_var="ALBENCH_HG38",
        default_relpath="reference/hg38.fa",
        description="hg38 reference FASTA (a .fai index beside it makes access fast)",
        how_to_get=(
            "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz\n"
            "        gunzip hg38.fa.gz && samtools faidx hg38.fa\n"
            "        (or: albench setup-data --asset hg38)"
        ),
        required_by=("encode_accessibility",),
        fallbacks=(
            "/grid/koo/home/dalin/ref/hg38.fa",
            "/grid/vakoc/home/toobian/Koo_Collab/hg38_genome.fa",
            "/grid/ngs/data/Elzar_Oxford/McCombie_Lab/hg38.fa",
        ),
    )
)

_add(
    Asset(
        key="jaspar_meme",
        env_var="ALBENCH_JASPAR_MEME",
        default_relpath="motifs/JASPAR2022_CORE_pfms.meme",
        description="JASPAR2022 CORE PFMs in MEME format",
        how_to_get=(
            "Download the CORE non-redundant vertebrate PFMs (MEME format) from\n"
            "        https://jaspar.elixir.no/downloads/  (or: albench setup-data "
            "--asset jaspar_meme)"
        ),
        required_by=("motif_planted_v2",),
        fallbacks=("/grid/koo/home/shared/cl_procap/annotations/JASPAR2022_CORE_pfms.meme",),
    )
)

_add(
    Asset(
        key="cisbp_meme",
        env_var="ALBENCH_CISBP_MEME",
        default_relpath="motifs/CISBP_human.meme",
        description=(
            "CIS-BP human PFMs in MEME format. Preferred over JASPAR for the motif "
            "vocabulary arms: far more comprehensive coverage of human TFs."
        ),
        how_to_get=(
            "Download the Homo sapiens bulk archive from http://cisbp.ccbr.utoronto.ca/\n"
            "        bulk_archive.php, then convert the PWM files to MEME format.\n"
            "        (or: albench setup-data --asset cisbp_meme)"
        ),
        required_by=("motif_planted_v2",),
    )
)

_add(
    Asset(
        key="zoonomia_rates",
        env_var="ALBENCH_ZOONOMIA_RATES",
        default_relpath="zoonomia/per_position_rates.npz",
        description=(
            "Per-position substitution rates across 241 mammals for cCRE regions, "
            "with matching sequences, phyloP and cCRE class"
        ),
        how_to_get=(
            "Built by scripts/build_zoonomia_rates.py from the Zoonomia 241-way\n"
            "        alignment. Ask Trevor for a copy of the built npz (~190 MB) if you\n"
            "        do not need to rebuild it."
        ),
        required_by=("phylogenetic_zoonomia",),
        repo_relpaths=("data/zoonomia/per_position_rates.npz",),
    )
)

_add(
    Asset(
        key="encode_peaks",
        env_var="ALBENCH_ENCODE_PEAKS",
        default_relpath="encode_accessibility",
        description="K562/HepG2 accessibility peaks and their shared/specific partition",
        how_to_get=(
            "Built by scripts/build_encode_partition.py from ENCODE DNase narrowPeak\n"
            "        accessions for K562 and HepG2."
        ),
        required_by=("encode_accessibility",),
        repo_relpaths=("data/encode_accessibility",),
        is_dir=True,
    )
)

_add(
    Asset(
        key="bg_cache",
        env_var="ALBENCH_BG_CACHE",
        default_relpath="chr_split_cache/chr_train_ref_only.npz",
        description=(
            "Genomic background pool used for planting, shuffling and mutating. "
            "Train-chromosome sequences only."
        ),
        how_to_get="Built by scripts/build_chr_split_cache.py from the K562 MPRA dataset.",
        required_by=("motif_planted_v2", "phylogenetic_zoonomia", "motif_shuffled"),
        repo_relpaths=("outputs/chr_split_cache/chr_train_ref_only.npz",),
    )
)

_add(
    Asset(
        key="tf_sets",
        env_var="ALBENCH_TF_SETS",
        default_relpath="motifs/tf_sets.yaml",
        description=(
            "Named TF sets used by the motif vocabulary arms: 'ct_enriched' (TFs "
            "active in K562/HepG2) and 'shared_core' (TFs whose preference is shared "
            "across cell types). A YAML mapping set-name -> list of gene symbols."
        ),
        how_to_get=(
            "Built by scripts/build_tf_sets.py from ENCODE TF ChIP-seq peak presence\n"
            "        in K562/HepG2 (preferred) or from public RNA-seq expression.\n"
            "        Only needed for vocab_subset=ct_enriched / shared_core; the\n"
            "        syntax_core arm needs no external data."
        ),
        required_by=("motif_ct_enriched", "motif_shared_core"),
    )
)

_add(
    Asset(
        key="oracle",
        env_var="ALBENCH_ORACLE_DIR",
        default_relpath="oracle/ag_mpra_v2",
        description=(
            "AlphaGenome-MPRA oracle ensemble (10 chromosome-fold models) used to "
            "pseudo-label generated sequences"
        ),
        how_to_get=(
            "Two options:\n"
            "        (a) OUR ENSEMBLE (recommended, matches published numbers): ask for\n"
            "            the outputs/oracle_v2 bundle and point ALBENCH_ORACLE_DIR at it.\n"
            "        (b) TRAIN YOUR OWN from public AlphaGenome weights:\n"
            "            https://github.com/google-deepmind/alphagenome then\n"
            "            experiments/train_oracle_s2_v2.py --fold_id 0..9\n"
            "        A single fold is enough to develop against; the full 10 are only\n"
            "        needed to reproduce the reported label quality."
        ),
        required_by=("labelling",),
        repo_relpaths=("outputs/oracle_v2",),
        is_dir=True,
    )
)


class MissingAsset(FileNotFoundError):
    """Raised with instructions rather than just a path."""


def resolve(key: str, explicit: str | Path | None = None, *, required: bool = True) -> Path | None:
    """Resolve an asset to a concrete path. See module docstring for the order.

    Raises:
        MissingAsset: if ``required`` and nothing resolved, with instructions.
    """
    if key not in ASSETS:
        raise KeyError(f"Unknown asset {key!r}. Known: {sorted(ASSETS)}")
    asset = ASSETS[key]

    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise MissingAsset(f"{key}: explicitly given path does not exist: {p}")

    env = os.environ.get(asset.env_var)
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        raise MissingAsset(f"{key}: {asset.env_var}={env} does not exist")

    for p in (_local_overrides().get(key), data_root() / asset.default_relpath):
        if p and Path(p).expanduser().exists():
            return Path(p).expanduser()

    for rel in asset.repo_relpaths:
        p = REPO_ROOT / rel
        if p.exists():
            return p

    for fb in asset.fallbacks:
        p = Path(fb)
        if p.exists() and os.access(p, os.R_OK):
            return p

    if not required:
        return None
    raise MissingAsset(
        f"\nMissing required asset: {key}\n"
        f"  what it is   : {asset.description}\n"
        f"  needed by    : {', '.join(asset.required_by) or 'various'}\n"
        f"  looked in    : ${asset.env_var}, paths.local.yaml, "
        f"{data_root() / asset.default_relpath}\n"
        f"  how to get it: {asset.how_to_get}\n"
        f"\nRun `albench doctor` to see the status of every asset at once.\n"
    )


def _local_overrides() -> dict[str, str]:
    """Per-machine overrides from paths.local.yaml at the repo root (git-ignored)."""
    f = REPO_ROOT / "paths.local.yaml"
    if not f.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(f.read_text()) or {}
    except Exception:
        return {}


def status() -> list[tuple[str, bool, str]]:
    """(key, present, resolved-path-or-reason) for every asset. Powers `albench doctor`."""
    out = []
    for key in sorted(ASSETS):
        try:
            p = resolve(key, required=False)
        except MissingAsset as e:
            out.append((key, False, str(e).strip().splitlines()[0]))
            continue
        out.append((key, p is not None, str(p) if p else "not found"))
    return out
