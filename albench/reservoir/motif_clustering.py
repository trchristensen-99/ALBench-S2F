"""Regulatory architecture clustering reservoir sampler.

Clusters sequences by their TF motif composition (regulatory architecture)
and samples uniformly across clusters to ensure diverse representation of
different regulatory programs. This tests the hypothesis that covering a
wide range of regulatory architectures improves student model generalization.

Each sequence is characterized by a motif feature vector capturing:
- Presence/count of key TF binding motifs (consensus substring matching)
- Total motif count and motif density
- GC content

Sequences are clustered via KMeans on these features, and samples are drawn
uniformly across clusters so that rare regulatory architectures are
represented proportionally.

Optional post-selection mutagenesis generates additional diversity around
selected sequences.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

_RC_MAP = str.maketrans("ACGT", "TGCA")

# Curated motif library for K562/MPRA regulatory sequence analysis.
# Motifs use IUPAC codes where needed; matching resolves ambiguity.
REGULATORY_MOTIF_LIBRARY = {
    # Core promoter elements
    "TATA": "TATAAA",
    "INR": "YYANWYY",  # Initiator (Y=C/T, W=A/T, N=any)
    "BRE": "SSRCGCC",  # TFIIB recognition element (S=G/C, R=A/G)
    "DPE": "RGWYV",  # Downstream promoter element (V=A/C/G)
    # Key TFs in K562/MPRA
    "SP1": "GGGCGG",
    "AP1": "TGAGTCA",
    "GATA": "AGATAA",
    "ETS": "GGAA",
    "NFKB": "GGGACTTTCC",
    "CRE": "TGACGTCA",
    "EBOX": "CACGTG",
    "CTCF": "CCGCGNGGNGGCAG",
    "YY1": "CGCCATNTT",  # N=any
    "CCAAT": "CCAAT",
    # Additional regulatory motifs
    "MYC": "CACGTG",  # E-box (overlaps EBOX; kept for naming clarity)
    "CEBP": "TTGCGCAA",
    "OCT4": "ATGCAAAT",
    "SRF": "CCWTATAWGG",  # W=A/T
}

# IUPAC ambiguity codes → allowed nucleotides
_IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(_RC_MAP)[::-1]


def _iupac_matches(seq: str, motif: str) -> int:
    """Count non-overlapping occurrences of an IUPAC motif in a sequence.

    Scans both forward and reverse complement strands.
    """
    seq = seq.upper()
    count = 0
    for strand_seq in [seq, _reverse_complement(seq)]:
        mlen = len(motif)
        i = 0
        while i <= len(strand_seq) - mlen:
            match = True
            for j in range(mlen):
                allowed = _IUPAC.get(motif[j], motif[j])
                if strand_seq[i + j] not in allowed:
                    match = False
                    break
            if match:
                count += 1
                i += mlen  # non-overlapping
            else:
                i += 1
    return count


def _gc_content(seq: str) -> float:
    """Compute GC fraction of a sequence."""
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return gc / max(len(seq), 1)


def build_motif_features(
    sequences: list[str] | np.ndarray,
    motif_library: dict[str, str] | None = None,
    mode: str = "count",
) -> tuple[np.ndarray, list[str]]:
    """Build motif feature matrix for a list of sequences.

    Args:
        sequences: DNA sequences to scan.
        motif_library: Motif name -> IUPAC consensus. Defaults to
            ``REGULATORY_MOTIF_LIBRARY``.
        mode: ``"count"`` (occurrence counts) or ``"binary"`` (0/1 presence).

    Returns:
        Tuple of (feature_matrix [N, D], feature_names).
        Features include per-motif counts/binary, total_motifs,
        motif_density, and gc_content.
    """
    if motif_library is None:
        motif_library = REGULATORY_MOTIF_LIBRARY

    motif_names = list(motif_library.keys())
    motif_seqs = list(motif_library.values())
    n_seq = len(sequences)
    n_motifs = len(motif_names)

    # Per-motif counts
    counts = np.zeros((n_seq, n_motifs), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq_str = str(seq)
        for j, motif in enumerate(motif_seqs):
            counts[i, j] = _iupac_matches(seq_str, motif)

    if mode == "binary":
        motif_features = (counts > 0).astype(np.float32)
    else:
        motif_features = counts

    # Aggregate features
    total_motifs = counts.sum(axis=1, keepdims=True)  # total hits across all motifs
    seq_lengths = np.array([len(str(s)) for s in sequences], dtype=np.float32)
    motif_density = total_motifs / np.maximum(seq_lengths, 1.0)[:, None]
    gc = np.array([_gc_content(str(s)) for s in sequences], dtype=np.float32)[:, None]

    features = np.concatenate([motif_features, total_motifs, motif_density, gc], axis=1)
    feature_names = motif_names + ["total_motifs", "motif_density", "gc_content"]

    return features, feature_names


class MotifClusteringSampler(ReservoirSampler):
    """Sample sequences with uniform coverage across regulatory architecture clusters.

    Pipeline:
    1. Scan all pool sequences for TF binding motif occurrences
    2. Build a feature vector per sequence (motif counts + GC + density)
    3. Cluster with KMeans (or MiniBatchKMeans for large pools)
    4. Draw equal numbers from each cluster for maximum regulatory diversity
    5. Optionally apply light point mutagenesis to selected sequences

    This ensures that rare regulatory architectures (e.g., sequences with
    NFkB + AP1 co-occurrence) are represented even when they are a small
    fraction of the pool.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_clusters: int = 30,
        feature_mode: str = "count",
        normalize_features: bool = True,
        mutagenesis_rate: float = 0.0,
        mutagenesis_fraction: float = 0.0,
        mini_batch_threshold: int = 50_000,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed for reproducibility.
            n_clusters: Number of KMeans clusters (default 30).
            feature_mode: ``"count"`` or ``"binary"`` for motif features.
            normalize_features: Whether to z-score normalize features before
                clustering (recommended).
            mutagenesis_rate: Per-base mutation rate for optional post-selection
                mutagenesis. Set to 0.0 to disable.
            mutagenesis_fraction: Fraction of selected sequences to mutagenize
                (0.0 = none, 1.0 = all). Mutagenized copies are *appended*
                to the selection, so final count may exceed ``n_sequences``.
            mini_batch_threshold: Use MiniBatchKMeans when pool size exceeds
                this threshold (faster for large pools).
        """
        self._rng = np.random.default_rng(seed)
        self.n_clusters = n_clusters
        self.feature_mode = feature_mode
        self.normalize_features = normalize_features
        self.mutagenesis_rate = mutagenesis_rate
        self.mutagenesis_fraction = mutagenesis_fraction
        self.mini_batch_threshold = mini_batch_threshold
        self._seed = seed

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Backward-compatible: random subset (ignores clustering)."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def _cluster_sequences(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Cluster feature matrix and return cluster labels.

        Uses MiniBatchKMeans for large pools, standard KMeans otherwise.
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        if self.normalize_features:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        n_samples = features.shape[0]
        # Cap clusters at pool size
        k = min(self.n_clusters, n_samples)

        if n_samples > self.mini_batch_threshold:
            logger.info(
                f"Using MiniBatchKMeans (n={n_samples:,} > threshold={self.mini_batch_threshold:,})"
            )
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=self._seed,
                batch_size=min(10_000, n_samples),
                n_init=3,
            )
        else:
            km = KMeans(
                n_clusters=k,
                random_state=self._seed,
                n_init=10,
            )

        labels = km.fit_predict(features)
        return labels

    def _mutagenize(
        self,
        sequences: list[str],
    ) -> tuple[list[str], list[int], list[int]]:
        """Apply light point mutagenesis to a subset of sequences.

        Returns:
            Tuple of (mutated_sequences, parent_indices, n_mutations_per_seq).
        """
        n_to_mutate = max(1, int(len(sequences) * self.mutagenesis_fraction))
        parent_indices = self._rng.choice(len(sequences), size=n_to_mutate, replace=True).tolist()

        nuc_bytes = np.frombuffer(b"ACGT", dtype=np.uint8)
        nuc_map = {c: i for i, c in enumerate("ACGT")}
        mutated: list[str] = []
        n_mutations_list: list[int] = []

        for pi in parent_indices:
            seq = sequences[pi].upper()
            arr = np.array([nuc_map.get(c, 0) for c in seq], dtype=np.uint8)
            seq_len = len(arr)
            n_mut = max(1, int(round(self.mutagenesis_rate * seq_len)))
            n_mut = min(n_mut, seq_len)

            positions = self._rng.choice(seq_len, size=n_mut, replace=False)
            shifts = self._rng.integers(1, 4, size=n_mut, dtype=np.uint8)
            arr[positions] = (arr[positions] + shifts) % 4

            mut_seq = nuc_bytes[arr].tobytes().decode("ascii")
            mutated.append(mut_seq)
            n_mutations_list.append(n_mut)

        return mutated, parent_indices, n_mutations_list

    def generate(
        self,
        n_sequences: int,
        pool_sequences: list[str] | np.ndarray,
        motif_library: dict[str, str] | None = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Sample sequences with uniform coverage across motif-based clusters.

        Args:
            n_sequences: Number of sequences to select from the pool.
            pool_sequences: Full pool of candidate sequences.
            motif_library: Optional custom motif library (name -> IUPAC consensus).

        Returns:
            Tuple of (selected_sequences, metadata_df).
            If mutagenesis is enabled, mutagenized copies are appended after
            the cluster-selected sequences.
        """
        n_pool = len(pool_sequences)
        if n_pool == 0:
            raise ValueError("Pool is empty.")

        # Step 1: Build motif features
        logger.info(f"Scanning {n_pool:,} sequences for motif features...")
        features, feature_names = build_motif_features(
            pool_sequences,
            motif_library=motif_library,
            mode=self.feature_mode,
        )

        # Step 2: Cluster
        n_effective_clusters = min(self.n_clusters, n_pool)
        logger.info(f"Clustering into {n_effective_clusters} clusters...")
        labels = self._cluster_sequences(features)
        unique_labels = np.unique(labels)
        n_actual_clusters = len(unique_labels)

        # Step 3: Uniform sampling across clusters
        per_cluster = n_sequences // n_actual_clusters
        remainder = n_sequences % n_actual_clusters

        selected_indices: list[int] = []
        selected_clusters: list[int] = []

        # Shuffle cluster order so remainder is distributed randomly
        cluster_order = unique_labels.copy()
        self._rng.shuffle(cluster_order)

        for rank, cid in enumerate(cluster_order):
            members = np.where(labels == cid)[0]
            n_draw = per_cluster + (1 if rank < remainder else 0)
            if n_draw == 0:
                continue
            replace = len(members) < n_draw
            drawn = self._rng.choice(members, size=n_draw, replace=replace)
            selected_indices.extend(drawn.tolist())
            selected_clusters.extend([int(cid)] * n_draw)

        indices = np.array(selected_indices)
        sequences = [str(pool_sequences[i]) for i in indices]

        # Build per-sequence motif composition strings for metadata
        motif_names_list = list((motif_library or REGULATORY_MOTIF_LIBRARY).keys())
        motif_compositions: list[str] = []
        n_motifs_per_seq: list[int] = []
        for i in indices:
            row = features[i]
            n_motif_features = len(motif_names_list)
            present = [motif_names_list[j] for j in range(n_motif_features) if row[j] > 0]
            motif_compositions.append(",".join(present) if present else "none")
            n_motifs_per_seq.append(int(row[n_motif_features]))  # total_motifs column

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(len(sequences), dtype=np.int64),
                "method": "motif_clustering",
                "source": "pool",
                "pool_idx": indices,
                "cluster_id": np.array(selected_clusters, dtype=np.int32),
                "motif_composition": motif_compositions,
                "n_motifs": np.array(n_motifs_per_seq, dtype=np.int32),
                "gc_content": features[indices, -1],
            }
        )

        # Log cluster size distribution
        cluster_sizes = np.bincount(labels, minlength=n_actual_clusters)
        logger.info(
            f"Motif clustering: {len(sequences):,} sequences from "
            f"{n_actual_clusters} clusters "
            f"(cluster sizes: min={cluster_sizes.min()}, "
            f"median={int(np.median(cluster_sizes))}, "
            f"max={cluster_sizes.max()})"
        )

        # Step 4: Optional mutagenesis
        if self.mutagenesis_rate > 0 and self.mutagenesis_fraction > 0:
            mut_seqs, mut_parents, mut_counts = self._mutagenize(sequences)
            n_mut = len(mut_seqs)

            mut_meta = pd.DataFrame(
                {
                    "seq_idx": np.arange(len(sequences), len(sequences) + n_mut, dtype=np.int64),
                    "method": "motif_clustering_mutant",
                    "source": "mutagenized",
                    "pool_idx": indices[mut_parents],
                    "cluster_id": np.array(
                        [selected_clusters[p] for p in mut_parents], dtype=np.int32
                    ),
                    "motif_composition": [motif_compositions[p] for p in mut_parents],
                    "n_motifs": np.array(
                        [n_motifs_per_seq[p] for p in mut_parents], dtype=np.int32
                    ),
                    "gc_content": np.array([_gc_content(s) for s in mut_seqs], dtype=np.float32),
                    "n_mutations": np.array(mut_counts, dtype=np.int32),
                }
            )

            sequences.extend(mut_seqs)
            meta = pd.concat([meta, mut_meta], ignore_index=True)
            logger.info(
                f"  + {n_mut:,} mutagenized variants "
                f"(rate={self.mutagenesis_rate:.3f}, "
                f"fraction={self.mutagenesis_fraction:.2f})"
            )

        return sequences, meta
