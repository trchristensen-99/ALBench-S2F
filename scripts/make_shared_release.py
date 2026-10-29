#!/usr/bin/env python
"""Build the clean code snapshot that is handed to collaborators (and refresh it whenever the live code changes).

Why: the repository holds a year of work, including the analysis and bookkeeping scripts of the study itself. A collaborator who opens it cannot tell
which of those matter for their task, and has more than once run an outdated one. The release contains only what the shared workflow needs:

* ``albench/`` whole (registry factories and lazy imports make tracing inside the package unreliable), its tests, the two docs a newcomer needs;
* the entry-point scripts for labelling pools with the oracle;
* every first-party module those import, found by a static import closure over ALL import statements (including function-level ones), so nothing
  the kept code reaches for is missing.

Only files tracked by git at HEAD are copied (no outputs, no scratch files, no attic). The snapshot is written to a fresh directory next to the target
and swapped in, so a reader never sees a half-built tree; ``RELEASE.md`` records the commit it was built from.

    python scripts/make_shared_release.py --dest /grid/koo/home/shared/albench/ALBench-S2F [--verify]
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEP_TREES = ("albench",)
KEEP_FILES = (
    "AGENTS.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".pre-commit-config.yaml",
    "docs/MENU.md",
    "docs/STRUCTURE.md",
    "docs/ENVIRONMENT.md",
    "scripts/setup_env.sh",
    "requirements/menu.txt",
    "requirements/oracle.txt",
    "requirements/lock_full.txt",
    "scripts/make_shared_release.py",
)
# tests of the live package; the AutoResearch loop tests, the legacy experiment-utility tests and tests that need absent data are left out
KEEP_TESTS = (
    "tests/test_menu.py",
    "tests/test_flank_context.py",
    "tests/test_acquisition.py",
    "tests/test_registry.py",
    "tests/test_pools.py",
    "tests/test_run.py",
    "tests/test_composition.py",
    "tests/test_holdout_defaults.py",
    "tests/test_albench",
)
# scripts a collaborator uses: build a candidate pool from a registry strategy, label sequences with the oracle
ENTRY_SCRIPTS = (
    "scripts/build_pool.py",
    "scripts/label_pool.py",
    "scripts/generate_labeled_pools.py",
)
IMPORT_ROOTS = (
    "",
    "experiments",
    "scripts",
)  # where a top-level module name may live (experiments/ and scripts/ are put on sys.path by our launchers)


def tracked() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return set(out.splitlines())


def under(files: set[str], prefix: str) -> set[str]:
    return {f for f in files if f == prefix or f.startswith(prefix.rstrip("/") + "/")}


def resolve_module(name: str, files: set[str]) -> list[str]:
    """Tracked files that provide dotted module ``name`` (a module, or a package and its __init__ chain)."""
    parts = name.split(".")
    found: list[str] = []
    for base in IMPORT_ROOTS:
        pre = f"{base}/" if base else ""
        for k in range(1, len(parts) + 1):
            stem = pre + "/".join(parts[:k])
            for cand in (f"{stem}.py", f"{stem}/__init__.py"):
                if cand in files:
                    found.append(cand)
    return found


# The AutoResearch search driver (albench/arch_search) reaches into the study's HP-search code from inside main() and helper functions. Those paths run only
# when a search is launched, which a collaborator does not do, so for these files only module-level imports are followed.
SEARCH_ONLY = ("albench/arch_search/",)


def first_party_imports(path: Path, module_level_only: bool = False) -> list[tuple[str, list[str]]]:
    """(module, imported names) for every import statement in a file; function-level ones included unless ``module_level_only``."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    out: list[tuple[str, list[str]]] = []
    for node in tree.body if module_level_only else ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, [a.name for a in node.names]))
    return out


def closure(seeds: set[str], files: set[str]) -> tuple[set[str], list[str]]:
    keep, todo, missing = set(), [s for s in seeds if s.endswith(".py")], []
    top_names = {p.split("/")[0] for p in files} | {
        Path(p).stem
        for p in files
        if p.startswith(("experiments/", "scripts/")) and p.endswith(".py")
    }
    while todo:
        f = todo.pop()
        if f in keep:
            continue
        keep.add(f)
        for mod, names in first_party_imports(
            ROOT / f, module_level_only=f.startswith(SEARCH_ONLY) and not f.endswith("validate.py")
        ):
            cands = resolve_module(mod, files)
            for n in names:  # `from pkg import submodule`
                cands += resolve_module(f"{mod}.{n}", files)
            if (
                not cands
                and mod.split(".")[0] in top_names
                and mod.split(".")[0] not in sys.stdlib_module_names
            ):
                missing.append(f"{f}: {mod}")
            todo += [c for c in cands if c.endswith(".py") and c not in keep]
            keep.update(c for c in cands if c.endswith("__init__.py"))
    return keep, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="import the package and run the kept fast tests inside the snapshot",
    )
    args = ap.parse_args()
    files = tracked()
    seeds: set[str] = set()
    for t in KEEP_TREES:
        seeds |= under(files, t)
    for f in KEEP_FILES:
        if f in files:
            seeds.add(f)
    for t in KEEP_TESTS:
        seeds |= under(files, t)
    for s in ENTRY_SCRIPTS:
        if s in files:
            seeds.add(s)
    keep, missing = closure({f for f in seeds if f.endswith(".py")}, files)
    keep |= seeds
    # data files the kept code reads from the repo (package data, e.g. the menu)
    keep |= {f for f in files if f.startswith("albench/") and not f.endswith(".py")}
    pulled = sorted(f for f in keep if f not in seeds)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dest = Path(args.dest)
    tmp = dest.with_name(dest.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    for f in sorted(keep):
        dst = tmp / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
    notes = [
        "# Release snapshot",
        "",
        f"Built {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}`{' (working tree had uncommitted changes to tracked files; only HEAD was copied)' if dirty else ''} by `scripts/make_shared_release.py`.",
        "",
        "This is the part of the repository a collaborator needs: the `albench` package, its tests, `docs/MENU.md` (which model configurations to train and how),",
        "`docs/STRUCTURE.md`, `AGENTS.md`, the oracle-labelling scripts and the few first-party modules they import. It deliberately leaves out the study's own",
        "analysis, registration and bookkeeping scripts and everything under `outputs/`; those are in the full repository and are not needed to use the menu,",
        "the acquisition code or the oracle. Do not edit this tree in place: change the full repository and rebuild.",
        "",
        "First-party modules pulled in by import closure (beyond the kept trees):",
        "",
        *[f"* `{f}`" for f in pulled],
        "",
        "Imports of first-party modules that could not be resolved to a tracked file (none is expected):",
        "",
        *([f"* {m}" for m in missing] or ["* none"]),
    ]
    (tmp / "RELEASE.md").write_text("\n".join(notes) + "\n")
    if dest.exists():
        old = dest.with_name(dest.name + ".previous")
        if old.exists():
            shutil.rmtree(old)
        dest.rename(old)
    tmp.rename(dest)
    shutil.rmtree(dest.with_name(dest.name + ".previous"), ignore_errors=True)
    print(
        f"wrote {len(keep)} files to {dest} (commit {commit}); {len(pulled)} pulled in by import closure; unresolved first-party imports: {len(missing)}"
    )
    for m in missing:
        print("  UNRESOLVED", m)
    if args.verify:
        env = dict(
            os.environ,
            PYTHONPATH=f"{dest}:{dest}/experiments:{dest}/scripts",
            PYTHONDONTWRITEBYTECODE="1",
        )
        code = (
            "import importlib, pkgutil, albench\n"
            "bad = []\n"
            "for m in pkgutil.walk_packages(albench.__path__, 'albench.'):\n"
            "    try:\n        importlib.import_module(m.name)\n    except Exception as e:\n        bad.append((m.name, type(e).__name__, str(e)[:80]))\n"
            "print('modules failing to import:', bad or 'none')\n"
        )
        subprocess.run([sys.executable, "-c", code], cwd=dest, env=env, check=False)
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-x", "-p", "no:cacheprovider"],
            cwd=dest,
            env=env,
            check=False,
        )
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
