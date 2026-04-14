#!/usr/bin/env python
"""In-context perturbation experiment v2: use AG predict_variant API.

Tests whether default AlphaGenome can distinguish functional enhancer/promoter
from random DNA when placed in full genomic context.

For each locus:
  1. Get reference prediction (real enhancer in genomic context)
  2. Create "variant" with random DNA replacing the enhancer
  3. Use predict_variant to get the difference

Usage:
    uv run --no-sync python scripts/analysis/in_context_perturbation_v2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# Well-known K562 regulatory loci (hg38)
LOCI = [
    ("GATA1_promoter", "chrX", 48786522, 48786722, "GATA1"),
    ("MYC_enhancer", "chr8", 128746000, 128746200, "MYC"),
    ("BCL2_promoter", "chr18", 63123346, 63123546, "BCL2"),
    ("HBG1_promoter", "chr11", 5249833, 5250033, "HBG1"),
    ("HMBS_promoter", "chr11", 119084553, 119084753, "HMBS"),
]


def get_genomic_sequence(chrom, start, end):
    """Fetch from local hg38 FASTA or UCSC API."""
    try:
        import pysam

        for genome_path in [
            "/grid/wsbs/home_norepl/christen/genomes/hg38.fa",
            "/grid/koo/data/genomes/hg38.fa",
        ]:
            if Path(genome_path).exists():
                fa = pysam.FastaFile(genome_path)
                seq = fa.fetch(chrom, start, end)
                fa.close()
                return seq.upper()
    except (ImportError, Exception):
        pass

    try:
        import pyfaidx

        for genome_path in [
            "/grid/wsbs/home_norepl/christen/genomes/hg38.fa",
            "/grid/koo/data/genomes/hg38.fa",
        ]:
            if Path(genome_path).exists():
                fa = pyfaidx.Fasta(genome_path)
                seq = str(fa[chrom][start:end])
                return seq.upper()
    except (ImportError, Exception):
        pass

    # Fallback: UCSC API
    try:
        import urllib.request

        url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start};end={end}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("dna", "").upper()
    except Exception:
        pass
    return None


def one_hot_encode(seq):
    """Encode DNA sequence as (L, 4) one-hot array."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping:
            ohe[i, mapping[c]] = 1.0
    return ohe


def main():
    import jax
    import jax.numpy as jnp
    from alphagenome_ft import create_model_with_heads

    rng = np.random.default_rng(42)
    out_dir = REPO / "outputs" / "in_context_perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )

    print("Loading default AlphaGenome...")
    model = create_model_with_heads(
        "all_folds",
        heads=[],
        checkpoint_path=weights_path,
        use_encoder_output=False,
    )
    print(f"  Loaded. Using predict_variant API.")

    results = {}

    for name, chrom, enh_start, enh_end, gene_name in LOCI:
        print(f"\n=== {name} ({gene_name}) ===")

        # Get 16384bp window centered on enhancer
        center = (enh_start + enh_end) // 2
        window_start = center - 8192
        window_end = center + 8192

        ref_seq = get_genomic_sequence(chrom, window_start, window_end)
        if ref_seq is None or len(ref_seq) < 16384:
            print(f"  SKIP: could not fetch genomic sequence")
            continue

        enh_offset = enh_start - window_start
        enh_len = enh_end - enh_start
        print(f"  Window: {chrom}:{window_start}-{window_end}")
        print(f"  Enhancer offset: {enh_offset}-{enh_offset + enh_len}")

        # Generate replacements
        n_random = 20
        replacements = {
            "random": ["".join(rng.choice(list("ACGT"), enh_len)) for _ in range(n_random)],
            "cpg_depleted_random": [],
            "shuffled_enhancer": [],
        }
        # CpG-depleted random
        for seq in replacements["random"]:
            s = list(seq)
            for i in range(len(s) - 1):
                if s[i] == "C" and s[i + 1] == "G":
                    s[i] = "T"
            replacements["cpg_depleted_random"].append("".join(s))
        # Shuffled enhancer
        real_enh = ref_seq[enh_offset : enh_offset + enh_len]
        for _ in range(n_random):
            s = list(real_enh)
            rng.shuffle(s)
            replacements["shuffled_enhancer"].append("".join(s))

        # For each replacement type, create alt sequences and predict
        ref_ohe = one_hot_encode(ref_seq)
        locus_results = {}

        for rep_type, alt_inserts in replacements.items():
            diffs = []
            for alt_insert in alt_inserts[:10]:
                alt_seq = ref_seq[:enh_offset] + alt_insert + ref_seq[enh_offset + enh_len :]
                alt_ohe = one_hot_encode(alt_seq)

                # Use predict_variant which computes ref and alt predictions
                try:
                    ref_preds, alt_preds = model.predict_variant(
                        jnp.array(ref_ohe)[None, ...],
                        jnp.array(alt_ohe)[None, ...],
                        jnp.zeros(1, dtype=jnp.int32),
                    )

                    # Sum absolute differences across all tracks at center bins
                    for track_name in list(ref_preds.keys())[:5]:
                        ref_val = np.array(ref_preds[track_name]).squeeze()
                        alt_val = np.array(alt_preds[track_name]).squeeze()
                        if ref_val.ndim >= 1:
                            center = len(ref_val) // 2
                            # Take center 10 bins
                            r = float(ref_val[max(0, center - 5) : center + 5].mean())
                            a = float(alt_val[max(0, center - 5) : center + 5].mean())
                            diffs.append(
                                {
                                    "track": track_name,
                                    "ref": r,
                                    "alt": a,
                                    "diff": a - r,
                                    "pct_change": (a - r) / (abs(r) + 1e-8) * 100,
                                }
                            )
                except Exception as e:
                    print(f"  predict_variant failed: {e}")
                    # Fall back to separate predictions
                    break

            if diffs:
                mean_diff = np.mean([d["diff"] for d in diffs])
                mean_pct = np.mean([d["pct_change"] for d in diffs])
                print(
                    f"  {rep_type}: mean_diff={mean_diff:+.4f} mean_pct_change={mean_pct:+.1f}% (N={len(diffs)})"
                )
                locus_results[rep_type] = {
                    "mean_diff": float(mean_diff),
                    "mean_pct_change": float(mean_pct),
                    "n_comparisons": len(diffs),
                }
            else:
                print(f"  {rep_type}: no valid comparisons")

        results[name] = {"gene": gene_name, "chrom": chrom, **locus_results}

    # Save
    with open(out_dir / "perturbation_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'perturbation_results_v2.json'}")


if __name__ == "__main__":
    main()
