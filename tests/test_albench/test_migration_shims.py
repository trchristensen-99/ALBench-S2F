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
