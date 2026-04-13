#!/usr/bin/env python
"""In-context MPRA perturbation experiment.

Tests whether AlphaGenome (default, NOT fine-tuned) can distinguish
functional enhancer/promoter sequences from random DNA when placed
in full genomic context.

Experiment:
1. Select ~5 well-known K562 enhancer/promoter loci near genes
2. For each locus:
   a. Get the full ~16kb genomic context (AG input window)
   b. Predict gene expression with the REAL enhancer/promoter
   c. Replace the enhancer/promoter with random 200bp DNA
   d. Predict gene expression again
   e. If expression drops: AG correctly identifies enhancer loss
   f. If expression stays: AG has bias / can't distinguish
3. Also test: replace enhancer with CpG-depleted random DNA
   (controls for the CpG confound in full genomic context)

This tests whether the CpG bias exists in the FULL model (encoder + all
heads) or only in the fine-tuned S2 head.

Usage (on HPC with GPU):
    uv run --no-sync python scripts/analysis/in_context_perturbation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# Well-known K562 regulatory loci (hg38 coordinates)
# Each: (name, chr, enhancer_start, enhancer_end, gene_name, gene_tss)
LOCI = [
    # GATA1 promoter — key K562 TF, strong enhancer
    ("GATA1_promoter", "chrX", 48786522, 48786722, "GATA1", 48786622),
    # MYC super-enhancer — one of the strongest K562 enhancers
    ("MYC_enhancer", "chr8", 128746000, 128746200, "MYC", 128748315),
    # BCL2 — regulated in K562
    ("BCL2_promoter", "chr18", 63123346, 63123546, "BCL2", 63123446),
    # HBG1 (fetal hemoglobin) — active in K562
    ("HBG1_promoter", "chr11", 5249833, 5250033, "HBG1", 5249933),
    # HMBS (porphobilinogen deaminase) — housekeeping, active in K562
    ("HMBS_promoter", "chr11", 119084553, 119084753, "HMBS", 119084653),
]


def get_genomic_sequence(chrom, start, end):
    """Fetch genomic sequence from local genome file or pysam."""
    # Try to use pysam/pyfaidx if available
    try:
        import pysam

        genome_path = "/grid/wsbs/home_norepl/christen/genomes/hg38.fa"
        if Path(genome_path).exists():
            fa = pysam.FastaFile(genome_path)
            seq = fa.fetch(chrom, start, end)
            fa.close()
            return seq.upper()
    except ImportError:
        pass

    # Fallback: try to fetch from UCSC
    try:
        import urllib.request

        url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start};end={end}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("dna", "").upper()
    except Exception:
        pass

    return None


def main():
    rng = np.random.default_rng(42)
    out_dir = REPO / "outputs" / "in_context_perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load default AlphaGenome (NOT fine-tuned)
    print("Loading default AlphaGenome...")
    import jax
    import jax.numpy as jnp
    from alphagenome_ft import create_model_with_heads

    weights_path = (
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[],  # No custom heads — use default AG outputs
        checkpoint_path=weights_path,
        use_encoder_output=False,
    )
    print(f"  Model loaded. Params: {sum(p.size for p in jax.tree.leaves(model._params)):,}")

    results = {}

    for name, chrom, enh_start, enh_end, gene_name, gene_tss in LOCI:
        print(f"\n=== {name} ({gene_name}) ===")

        # Get 16384bp window centered on the enhancer
        center = (enh_start + enh_end) // 2
        window_start = center - 8192
        window_end = center + 8192

        genomic_seq = get_genomic_sequence(chrom, window_start, window_end)
        if genomic_seq is None or len(genomic_seq) < 16384:
            print(f"  SKIP: could not fetch genomic sequence")
            continue

        print(f"  Genomic window: {chrom}:{window_start}-{window_end} ({len(genomic_seq)}bp)")

        # Position of enhancer within the window
        enh_offset_start = enh_start - window_start
        enh_offset_end = enh_end - window_start
        enh_len = enh_offset_end - enh_offset_start

        print(f"  Enhancer at offset {enh_offset_start}-{enh_offset_end} ({enh_len}bp)")

        # One-hot encode
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

        def encode(seq):
            ohe = np.zeros((len(seq), 4), dtype=np.float32)
            for i, c in enumerate(seq):
                if c in mapping:
                    ohe[i, mapping[c]] = 1.0
            return ohe

        # Predict with REAL enhancer
        real_ohe = encode(genomic_seq)

        # Generate random replacements
        n_random = 50
        random_seqs = ["".join(rng.choice(list("ACGT"), enh_len)) for _ in range(n_random)]
        # CpG-depleted random
        cpg_depleted_seqs = []
        for seq in random_seqs:
            s = list(seq)
            for i in range(len(s) - 1):
                if s[i] == "C" and s[i + 1] == "G":
                    s[i] = "T"
            cpg_depleted_seqs.append("".join(s))

        # Shuffled versions of the real enhancer
        real_enh_seq = genomic_seq[enh_offset_start:enh_offset_end]
        shuffled_seqs = []
        for _ in range(n_random):
            s = list(real_enh_seq)
            rng.shuffle(s)
            shuffled_seqs.append("".join(s))

        # Predict with each replacement
        @jax.jit
        def predict(params, state, seq_ohe):
            """Predict with AG model."""
            return model._predict(
                params,
                state,
                seq_ohe[None, ...],  # (1, 16384, 4)
                jnp.zeros(1, dtype=jnp.int32),
                negative_strand_mask=jnp.zeros(1, dtype=bool),
                strand_reindexing=None,
                requested_outputs=None,  # all outputs
            )

        # Get baseline prediction
        real_pred = predict(model._params, model._state, jnp.array(real_ohe))
        # Extract a representative output track (e.g., first track, center bin)
        # AG outputs multiple tracks; we want to see if expression changes
        real_tracks = {k: np.array(v).squeeze() for k, v in real_pred.items()}

        print(f"  Real enhancer prediction done. Tracks: {list(real_tracks.keys())[:5]}")

        # For each replacement type, predict and compare
        for rep_name, rep_seqs in [
            ("random", random_seqs),
            ("cpg_depleted_random", cpg_depleted_seqs),
            ("shuffled_enhancer", shuffled_seqs),
        ]:
            diffs = []
            for rep_seq in rep_seqs[:10]:  # first 10 for speed
                modified_seq = (
                    genomic_seq[:enh_offset_start] + rep_seq + genomic_seq[enh_offset_end:]
                )
                mod_ohe = encode(modified_seq)
                mod_pred = predict(model._params, model._state, jnp.array(mod_ohe))

                # Compare center bin predictions
                for track_name in list(real_tracks.keys())[:3]:
                    real_val = real_tracks[track_name]
                    mod_val = np.array(mod_pred[track_name]).squeeze()
                    # Center bins
                    center_bin = len(real_val) // 2
                    r = (
                        float(real_val[center_bin])
                        if real_val.ndim == 1
                        else float(real_val[center_bin].mean())
                    )
                    m = (
                        float(mod_val[center_bin])
                        if mod_val.ndim == 1
                        else float(mod_val[center_bin].mean())
                    )
                    diffs.append({"track": track_name, "real": r, "modified": m, "diff": m - r})

            mean_diff = np.mean([d["diff"] for d in diffs])
            print(f"  {rep_name}: mean diff = {mean_diff:+.4f} (N={len(diffs)})")

        results[name] = {"chrom": chrom, "gene": gene_name}

    # Save
    with open(out_dir / "perturbation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
