"""Probe the on-cluster Borzoi training tfrecords to recover their schema.

The sidecar metadata (targets.txt / statistics.json) is permission-blocked (ebsoft:itstaff), so we
infer sequence length, target shape, and dtype directly from one record. Needed to wire real replay
(--cl replay_real) against /grid/hpc/data/borzoi/data/hg38/tfrecords/.
"""

import glob
import sys

import numpy as np
import tensorflow as tf

TFR_DIR = "/grid/hpc/data/borzoi/data/hg38/tfrecords"


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else f"{TFR_DIR}/*.tfr"
    files = sorted(glob.glob(pattern))
    print(f"found {len(files)} tfrecord files")
    if not files:
        return
    print(f"probing: {files[0]}")

    ds = tf.data.TFRecordDataset([files[0]], compression_type="ZLIB")
    raw = next(iter(ds.take(1)))
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    print("\n=== feature keys ===")
    for k, v in ex.features.feature.items():
        kind = v.WhichOneof("kind")
        n = len(getattr(v, kind).value)
        print(f"  {k}: {kind}, n_values={n}")

    # decode the usual basenji/borzoi layout: raw bytes -> uint8 seq, float16 targets
    for k, v in ex.features.feature.items():
        if v.WhichOneof("kind") != "bytes_list":
            continue
        b = v.bytes_list.value[0]
        print(f"\n=== '{k}' raw bytes: {len(b)} ===")
        for dt in (np.uint8, np.float16, np.float32):
            if len(b) % np.dtype(dt).itemsize:
                continue
            a = np.frombuffer(b, dtype=dt)
            print(f"  as {np.dtype(dt).name}: n={a.size}, min={a.min():.4g}, max={a.max():.4g}")
            if dt is np.uint8 and a.size % 4 == 0:
                print(f"     -> if (L,4) one-hot: L={a.size // 4}")
            if dt is np.float16 and a.size % 7611 == 0:
                print(f"     -> if (T,7611): T={a.size // 7611}")


if __name__ == "__main__":
    main()
