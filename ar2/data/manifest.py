"""Dataset manifests (spec 2.3-2.4). One manifest points to one npz file; `role` decides what the data may be used for.

Only three roles exist (train, val_loop, val_select). Anything else, including anything that looks like a test set, is refused: no path to test data may
appear in any config.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ALLOWED_ROLES = ("train", "val_loop", "val_select")
# a path component that names test data: test, tests, testset(s), test_sets*, *_test_sets* (ordinary names that merely start with "test_" are fine)
_TEST_COMPONENT = re.compile(
    r"^(tests?|testsets?|test[_-]sets?([_-].*)?|.*[_-]test[_-]sets?([_-].*)?)(\.[a-z0-9]+)*$", re.I
)
REQUIRED = ("path", "role", "tasks", "label_keys")


class ManifestError(ValueError):
    """A manifest that does not follow the contract."""


class RoleError(ManifestError):
    """A manifest whose role is not one of train, val_loop, val_select."""


def make_manifest(
    path: str,
    role: str,
    *,
    pool_id: str = "",
    reservoir: str = "",
    generation_seed: int | None = None,
    D: int | None = None,
    label_source: str = "oracle_v3",
    tasks: list[str] | None = None,
    seq_key: str = "sequences",
    label_keys: dict[str, str] | None = None,
    chrom_key: str | None = None,
    index_file: str | None = None,
    sha256: str | None = None,
    n: int | None = None,
) -> dict:
    """A manifest dict in the contract's shape (use :func:`save_manifest` to write it)."""
    tasks = tasks or ["K562"]
    return {
        "path": str(path),
        "sha256": sha256,
        "n": n,
        "role": role,
        "pool_id": pool_id,
        "reservoir": reservoir,
        "generation_seed": generation_seed,
        "D": D,
        "label_source": label_source,
        "tasks": tasks,
        "seq_key": seq_key,
        "label_keys": label_keys or {tasks[0]: "oracle_labels"},
        "chrom_key": chrom_key,
        "index_file": index_file,
    }


def validate_manifest(m: dict) -> dict:
    """Check keys, role and that no path looks like test data; returns the manifest."""
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise ManifestError(f"manifest is missing {missing}")
    if m["role"] not in ALLOWED_ROLES:
        raise RoleError(f"manifest role {m['role']!r} is not one of {ALLOWED_ROLES}")
    for key in ("path", "index_file"):
        p = m.get(key)
        if p and any(_TEST_COMPONENT.match(part) for part in Path(str(p)).parts):
            raise ManifestError(
                f"{key} {p!r} looks like a test-set path; test data is never loaded"
            )
    for t in m["tasks"]:
        if t not in m["label_keys"]:
            raise ManifestError(f"task {t!r} has no entry in label_keys")
    return m


def load_manifest(path: str | Path) -> dict:
    """Read and validate a manifest JSON file."""
    with open(path) as fh:
        m = json.load(fh)
    return validate_manifest(m)


def save_manifest(path: str | Path, manifest: dict) -> None:
    """Write a validated manifest (atomic)."""
    validate_manifest(manifest)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(path)
