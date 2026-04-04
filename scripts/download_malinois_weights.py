#!/usr/bin/env python
"""Download pretrained Malinois model and Basset parent weights from GCS.

Downloads:
  1. Malinois trained model (3-output, K562+HepG2+SknSh): ~49 MB
     -> data/pretrained/malinois_trained/torch_checkpoint.pt
  2. Basset parent weights (conv layers only): ~19 MB
     -> data/pretrained/basset_pretrained.pkl

These are the official weights from:
  Gosai et al. (2024) "Machine-guided design of cell-type-targeting
  cis-regulatory elements" Nature.
  https://github.com/sjgosai/boda2
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

MALINOIS_URL = (
    "https://storage.googleapis.com/tewhey-public-data/CODA_resources/"
    "malinois_artifacts__20211113_021200__287348.tar.gz"
)
BASSET_URL = (
    "https://storage.googleapis.com/tewhey-public-data/CODA_resources/"
    "my-model.epoch_5-step_19885.pkl"
)

PRETRAINED_DIR = Path("data/pretrained")


def download_file(url: str, dest: Path, desc: str) -> None:
    if dest.exists():
        print(f"  {desc}: already exists at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc} ({url})...")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  Saved to {dest} ({size_mb:.1f} MB)")


def main():
    print("Downloading Malinois pretrained weights...")

    # 1. Basset parent weights
    basset_path = PRETRAINED_DIR / "basset_pretrained.pkl"
    download_file(BASSET_URL, basset_path, "Basset parent weights")

    # 2. Malinois trained model (tar.gz -> extract torch_checkpoint.pt)
    malinois_dir = PRETRAINED_DIR / "malinois_trained"
    checkpoint_path = malinois_dir / "torch_checkpoint.pt"
    if checkpoint_path.exists():
        print(f"  Malinois checkpoint: already exists at {checkpoint_path}")
    else:
        tar_path = PRETRAINED_DIR / "malinois_artifacts.tar.gz"
        download_file(MALINOIS_URL, tar_path, "Malinois model artifacts")

        # Extract
        print("  Extracting tar.gz...")
        malinois_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(tar_path), "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith("torch_checkpoint.pt"):
                    # Extract to malinois_dir
                    member.name = "torch_checkpoint.pt"
                    tf.extract(member, str(malinois_dir))
                    print(f"  Extracted: {checkpoint_path}")
                    break
            else:
                # Extract everything if torch_checkpoint.pt not found by name
                tf.extractall(str(malinois_dir))
                print(f"  Extracted all to {malinois_dir}")

        # Clean up tar
        tar_path.unlink()
        print(f"  Removed {tar_path}")

    # Verify
    if checkpoint_path.exists():
        import torch

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        print(f"\n  Malinois checkpoint keys: {list(ckpt.keys())}")
        if "model_module" in ckpt:
            print(f"  Model module: {ckpt['model_module']}")
        if "model_hparams" in ckpt:
            hp = vars(ckpt["model_hparams"])
            print(f"  Model hparams: {hp}")
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            n_params = sum(p.numel() for p in sd.values())
            print(f"  Parameters: {n_params:,}")
    else:
        print(f"  WARNING: {checkpoint_path} not found after extraction")

    print("\nDone!")


if __name__ == "__main__":
    main()
