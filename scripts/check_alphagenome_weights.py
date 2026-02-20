#!/usr/bin/env python3
"""Validate AlphaGenome checkpoint directory structure and basic integrity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REQUIRED_FILES = ("_METADATA", "_CHECKPOINT_METADATA", "manifest.ocdbt")
REQUIRED_DIRS = ("d", "ocdbt.process_0")


def _format_bytes(num_bytes: int) -> str:
    """Return human-readable byte size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def validate_weights_dir(path: Path, min_files: int) -> int:
    """Validate checkpoint directory and return process exit code."""
    if not path.exists():
        print(f"ERROR: path does not exist: {path}")
        return 2
    if not path.is_dir():
        print(f"ERROR: path is not a directory: {path}")
        return 2

    errors: list[str] = []
    for file_name in REQUIRED_FILES:
        if not (path / file_name).is_file():
            errors.append(f"missing required file: {file_name}")
    for dir_name in REQUIRED_DIRS:
        if not (path / dir_name).is_dir():
            errors.append(f"missing required directory: {dir_name}")

    file_count = 0
    total_size = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_count += 1
            file_path = Path(root) / filename
            try:
                total_size += file_path.stat().st_size
            except OSError:
                errors.append(f"failed to stat: {file_path}")

    if file_count < min_files:
        errors.append(f"file count below threshold: {file_count} < {min_files}")

    print(f"Path: {path}")
    print(f"Files: {file_count}")
    print(f"Size: {_format_bytes(total_size)}")

    if errors:
        print("STATUS: INVALID")
        for err in errors:
            print(f"- {err}")
        return 1

    print("STATUS: VALID")
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate AlphaGenome checkpoint directory.")
    parser.add_argument(
        "--path",
        default="~/alphagenome_weights/alphagenome-jax-all_folds-v1",
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=5,
        help="Minimum total file count required for a valid checkpoint.",
    )
    args = parser.parse_args()
    target = Path(args.path).expanduser().resolve()
    raise SystemExit(validate_weights_dir(target, min_files=args.min_files))


if __name__ == "__main__":
    main()
