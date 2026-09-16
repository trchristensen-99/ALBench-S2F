"""The deprecated module paths must stay aliases, not copies.

During the docs/STRUCTURE.md migration both `albench.loop` and `albench.core.loop`
resolve. If the shim ever degrades to `from albench.core.loop import *`, the two names
become separate module objects holding separate state: patching a private attribute on
one would not be seen by code reading the other, and monkeypatched tests would pass
while the patched code never ran. That failure is invisible in normal use, so it is
pinned here.
"""

from __future__ import annotations

import importlib

import pytest

MOVED = ["loop", "pools", "registry", "paths", "run"]


@pytest.mark.parametrize("name", MOVED)
def test_shim_is_the_same_module_object(name: str) -> None:
    old = importlib.import_module(f"albench.{name}")
    new = importlib.import_module(f"albench.core.{name}")
    assert old is new, (
        f"albench.{name} and albench.core.{name} are different module objects; "
        f"the shim has degraded from an alias to a re-export and their state has split"
    )


@pytest.mark.parametrize("name", MOVED)
def test_private_names_survive_the_shim(name: str) -> None:
    """A star-import shim drops underscore-prefixed names; an alias keeps them."""
    new = importlib.import_module(f"albench.core.{name}")
    old = importlib.import_module(f"albench.{name}")
    privates = [n for n in vars(new) if n.startswith("_") and not n.startswith("__")]
    missing = [n for n in privates if not hasattr(old, n)]
    assert not missing, f"albench.{name} is missing private names {missing}"


def test_protocols_are_importable() -> None:
    from albench.core.protocols import EvalSuite, Fitted, Oracle, Reservoir, Student

    for proto in (Oracle, Student, Fitted, Reservoir, EvalSuite):
        assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_protocol")


def test_repo_root_is_the_repo_not_the_package() -> None:
    """REPO_ROOT must survive this module being moved between directories.

    It was `Path(__file__).resolve().parents[1]`, correct while paths.py sat at
    albench/paths.py and wrong the moment it moved to albench/core/paths.py, where
    parents[1] is albench/. Every repo-relative asset then resolved to None, and
    because the CLI asked with required=False the symptom appeared much later as a
    missing genomic pool. Anchored on pyproject.toml now; this pins it.
    """
    from albench.core.paths import REPO_ROOT

    assert (REPO_ROOT / "pyproject.toml").is_file(), f"{REPO_ROOT} is not the repo root"
    assert (REPO_ROOT / "albench").is_dir()
    assert REPO_ROOT.name != "albench", "REPO_ROOT points at the package, not the repo"


def test_repo_relative_assets_resolve_against_the_repo() -> None:
    """A declared repo_relpath must be looked up under the repo root."""
    from albench.core.paths import ASSETS, REPO_ROOT

    bg = ASSETS["bg_cache"]
    assert bg.repo_relpaths, "bg_cache lost its repo_relpaths"
    for rel in bg.repo_relpaths:
        candidate = REPO_ROOT / rel
        assert "albench/albench" not in str(candidate), f"double-nested path: {candidate}"
