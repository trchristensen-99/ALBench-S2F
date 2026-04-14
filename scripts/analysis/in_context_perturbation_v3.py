#!/usr/bin/env python
"""In-context perturbation v3: use AG's genome-aware predict_variant API.

The AG model has a proper predict_variant(interval, variant) method that
handles genomic context, FASTA extraction, and ref/alt comparison internally.

We create synthetic "variants" that replace enhancer regions with random DNA.

Usage:
    uv run --no-sync python scripts/analysis/in_context_perturbation_v3.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# Well-known K562 loci (hg38)
LOCI = [
    ("GATA1_promoter", "chrX", 48786522, 48786722, "GATA1"),
    ("MYC_enhancer", "chr8", 128746000, 128746200, "MYC"),
    ("BCL2_promoter", "chr18", 63123346, 63123546, "BCL2"),
    ("HBG1_promoter", "chr11", 5249833, 5250033, "HBG1"),
    ("HMBS_promoter", "chr11", 119084553, 119084753, "HMBS"),
]


def main():
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    out_dir = REPO / "outputs" / "in_context_perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )

    # Use the low-level _predict API directly with one-hot encoding
    # since predict_variant needs genome FASTA access
    from alphagenome_ft import create_model_with_heads

    print("Loading default AlphaGenome...")
    model = create_model_with_heads(
        "all_folds",
        heads=[],
        checkpoint_path=weights_path,
        use_encoder_output=False,
    )

    # Get the base model's raw predict function
    base_model = model._base_model

    def one_hot(seq):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        ohe = np.zeros((len(seq), 4), dtype=np.float32)
        for i, c in enumerate(seq.upper()):
            if c in mapping:
                ohe[i, mapping[c]] = 1.0
        return ohe

    def get_genomic_seq(chrom, start, end):
        """Fetch from UCSC API."""
        try:
            import urllib.request

            url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start};end={end}"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("dna", "").upper()
        except Exception as e:
            print(f"  UCSC fetch failed: {e}")
            return None

    # Use the _predict method which takes raw one-hot sequences
    @jax.jit
    def predict_raw(params, state, seq_ohe):
        """Raw prediction on one-hot encoded sequence."""
        return base_model._apply(
            params,
            state,
            seq_ohe[None, ...],
            jnp.zeros(1, dtype=jnp.int32),
            is_training=False,
        )

    results = {}

    for name, chrom, enh_start, enh_end, gene_name in LOCI:
        print(f"\n=== {name} ({gene_name}) ===")

        center = (enh_start + enh_end) // 2
        window_start = center - 8192
        window_end = center + 8192

        ref_seq = get_genomic_seq(chrom, window_start, window_end)
        if not ref_seq or len(ref_seq) < 16384:
            print(f"  SKIP: couldn't fetch sequence (got {len(ref_seq) if ref_seq else 0}bp)")
            continue

        enh_offset = enh_start - window_start
        enh_len = enh_end - enh_start

        # Predict reference
        ref_ohe = jnp.array(one_hot(ref_seq))
        try:
            ref_out = predict_raw(model._params, model._state, ref_ohe)
            # ref_out is a dict of track predictions
            ref_tracks = {
                k: np.array(v).squeeze() for k, v in ref_out.items() if hasattr(v, "shape")
            }
            print(f"  Ref prediction: {len(ref_tracks)} tracks")
            if not ref_tracks:
                print(f"  WARNING: no tracks returned. Keys: {list(ref_out.keys())[:5]}")
                # Try getting the raw output
                print(f"  Output type: {type(ref_out)}")
                if hasattr(ref_out, "items"):
                    for k, v in list(ref_out.items())[:3]:
                        print(f"    {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
                continue
        except Exception as e:
            print(f"  Ref prediction failed: {e}")
            continue

        # Generate replacements and predict
        n_random = 10
        locus_results = {}

        for rep_type in ["random", "cpg_depleted_random", "shuffled"]:
            diffs = []
            for i in range(n_random):
                if rep_type == "random":
                    insert = "".join(rng.choice(list("ACGT"), enh_len))
                elif rep_type == "cpg_depleted_random":
                    insert = list("".join(rng.choice(list("ACGT"), enh_len)))
                    for j in range(len(insert) - 1):
                        if insert[j] == "C" and insert[j + 1] == "G":
                            insert[j] = "T"
                    insert = "".join(insert)
                else:
                    insert = list(ref_seq[enh_offset : enh_offset + enh_len])
                    rng.shuffle(insert)
                    insert = "".join(insert)

                alt_seq = ref_seq[:enh_offset] + insert + ref_seq[enh_offset + enh_len :]
                alt_ohe = jnp.array(one_hot(alt_seq))

                try:
                    alt_out = predict_raw(model._params, model._state, alt_ohe)
                    alt_tracks = {
                        k: np.array(v).squeeze() for k, v in alt_out.items() if hasattr(v, "shape")
                    }

                    for track_name in list(ref_tracks.keys())[:5]:
                        if track_name not in alt_tracks:
                            continue
                        rv = ref_tracks[track_name]
                        av = alt_tracks[track_name]
                        if rv.ndim >= 1 and len(rv) > 10:
                            c = len(rv) // 2
                            r_val = float(rv[max(0, c - 5) : c + 5].mean())
                            a_val = float(av[max(0, c - 5) : c + 5].mean())
                            diffs.append(
                                {
                                    "track": track_name,
                                    "ref": r_val,
                                    "alt": a_val,
                                    "diff": a_val - r_val,
                                    "pct_change": (a_val - r_val) / (abs(r_val) + 1e-8) * 100,
                                }
                            )
                except Exception as e:
                    print(f"  Alt prediction failed: {e}")
                    break

            if diffs:
                mean_diff = np.mean([d["diff"] for d in diffs])
                mean_pct = np.mean([d["pct_change"] for d in diffs])
                n_tracks = len(set(d["track"] for d in diffs))
                print(
                    f"  {rep_type}: mean_diff={mean_diff:+.4f} pct_change={mean_pct:+.1f}% ({n_tracks} tracks, {len(diffs)} comparisons)"
                )
                locus_results[rep_type] = {
                    "mean_diff": float(mean_diff),
                    "mean_pct_change": float(mean_pct),
                    "n_comparisons": len(diffs),
                    "sample_diffs": diffs[:5],
                }
            else:
                print(f"  {rep_type}: no valid comparisons")

        results[name] = {"gene": gene_name, "chrom": chrom, **locus_results}

    with open(out_dir / "perturbation_results_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'perturbation_results_v3.json'}")


if __name__ == "__main__":
    main()
