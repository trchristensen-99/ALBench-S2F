"""Real-replay anchors from Borzoi's ACTUAL training data (on-cluster, read in place).

`/grid/hpc/data/borzoi/data/hg38/tfrecords/` holds Borzoi's training shards (3473 .tfr, ~2 TB,
world-readable). We read them in place — no copy — and extract (sequence, real track targets) pairs to
use as continual-learning replay anchors, the counterpart to `--cl distill` (which instead matches a
frozen teacher's outputs).

Geometry (probed, see constants below): each record is a 524,288 bp sequence with a float16
(6144, 7611) target — 6144 bins x 32 bp = 196,608 bp, i.e. targets cover only the CENTERED span, with
163,840 bp cropped each side. Our MPRA path runs short (512 bp) windows, so we take sub-windows inside
the target-covered span plus the target bins that cover them.

CAVEAT worth stating in any writeup: real targets are the measured signal for a locus that the full
model reads WITH ~524 kb of context. A short-window forward cannot reproduce that, so real-replay loss
has an irreducible floor which distillation-on-the-same-window does not (there the teacher is handed
the same limited context). Raise --window_bp to trade faithfulness against cost.
"""

import argparse
import glob
import os

import numpy as np

TFR_DIR = "/grid/hpc/data/borzoi/data/hg38/tfrecords"
# Schema recovered by probing a shard (sidecar metadata is permission-blocked):
#   sequence: 524288 uint8 BASE INDICES (0..3), i.e. 1 byte/bp -> SEQ_LEN = 524288
#   target:   float16 (6144, 7611) -> 6144 bins x 7611 tracks
# 6144 bins * 32 bp = 196608 bp, so targets cover only the CENTERED 196608 bp of the sequence;
# (524288 - 196608)/2 = 163840 bp is cropped on each side. Hence seq_pos <-> bin mapping below.
SEQ_LEN = 524288
N_BINS = 6144
N_TRACKS = 7611
BIN_BP = 32
TARGET_SPAN = N_BINS * BIN_BP  # 196608
CROP_BP = (SEQ_LEN - TARGET_SPAN) // 2  # 163840
_IDX2BASE = np.array(list("ACGT"))


def _decode_seq(buf):
    """uint8 base-index buffer (0..3, one byte per bp) -> ACGT string."""
    a = np.frombuffer(buf, dtype=np.uint8)
    return "".join(_IDX2BASE[np.clip(a, 0, 3)])


def seq_pos_to_bin(pos):
    """Sequence coordinate -> target bin index (targets are cropped/centered)."""
    return (pos - CROP_BP) // BIN_BP


def load_replay_anchors(
    n, window_bp=512, windows_per_record=8, shards=4, seed=42, tfr_dir=TFR_DIR, tracks=None
):
    """Return (seqs, targets): `window_bp` crops with the REAL track values for the bins covering
    them -> targets (n, n_tracks, window_bp // BIN_BP).

    Each 524 kb record yields many usable windows, so we sample `windows_per_record` per record
    (uniformly within the target-covered span) rather than only the center — far fewer records read
    for a given n. `tracks` optionally subsets the 7611 tracks (memory: full set is ~47 MB/record).
    """
    import tensorflow as tf

    files = sorted(glob.glob(os.path.join(tfr_dir, "*.tfr")))
    if not files:
        raise FileNotFoundError(f"no tfrecords under {tfr_dir}")
    rng = np.random.default_rng(seed)
    pick = [files[i] for i in rng.choice(len(files), size=min(shards, len(files)), replace=False)]

    feat = {
        "sequence": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string),
    }
    nbins = max(1, window_bp // BIN_BP)
    seqs, tgts = [], []
    ds = tf.data.TFRecordDataset(pick, compression_type="ZLIB")
    for raw in ds:
        ex = tf.io.parse_single_example(raw, feat)
        seq = _decode_seq(ex["sequence"].numpy())
        tgt = np.frombuffer(ex["target"].numpy(), dtype=np.float16).astype(np.float32)
        tgt = tgt.reshape(-1, N_TRACKS)  # (6144, 7611)
        if tracks is not None:
            tgt = tgt[:, tracks]

        # sample windows whose bins lie inside the target-covered span
        for b0 in rng.choice(max(1, tgt.shape[0] - nbins), size=windows_per_record, replace=False):
            s0 = CROP_BP + int(b0) * BIN_BP
            sub = seq[s0 : s0 + window_bp]
            sub_t = tgt[int(b0) : int(b0) + nbins]
            if len(sub) != window_bp or sub_t.shape[0] != nbins:
                continue
            seqs.append(sub)
            tgts.append(sub_t.T)  # (n_tracks, nbins) to match borzoi_tracks output
            if len(seqs) >= n:
                break
        if len(seqs) >= n:
            break
    if not seqs:
        raise RuntimeError("no usable records decoded — check schema/geometry assumptions")
    return seqs, np.stack(tgts).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--window_bp", type=int, default=512)
    ap.add_argument("--windows_per_record", type=int, default=8)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tfr_dir", default=TFR_DIR)
    ap.add_argument(
        "--out", required=True, help="npz with sequences + track_targets (replay_real anchor cache)"
    )
    args = ap.parse_args()
    seqs, tgts = load_replay_anchors(
        args.n,
        args.window_bp,
        windows_per_record=args.windows_per_record,
        shards=args.shards,
        seed=args.seed,
        tfr_dir=args.tfr_dir,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, sequences=np.array(seqs), track_targets=tgts)
    print(
        f"[replay] wrote {args.out}: n={len(seqs)} targets={tgts.shape} ({tgts.nbytes / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
