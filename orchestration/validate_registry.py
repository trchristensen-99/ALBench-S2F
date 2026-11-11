#!/usr/bin/env python3
"""Check a dispatch registry against orchestration/registry_schema.md before the dispatcher reads it.

The dispatcher re-reads the registry every cycle and trusts it: a malformed cell raises inside ``cycle()``,
which logs "cycle error" and submits nothing for any cell, and a well-formed but mis-named cell is routed by
literal name prefixes (``MIG_NO``, ``RETRAIN_PFX``, ``LLM_SMALL``) without complaint. So besides the schema
this reports the two silent failures that have actually happened: an ``AL_*`` knob that no code reads
(the cell runs, the knob does nothing), and a cell family that the prefix tuples route by accident
(a 2-day wall instead of 3.5 h, or a MIG slice for a model that does not fit in 10.8 GB).

Stdlib only, so it runs under the same system python as the dispatcher, on the login node:

    python3 orchestration/validate_registry.py scratchpad/dispatch_registry.json
    python3 orchestration/validate_registry.py REG --root .   # also: scripts and `needs` paths exist

Exit status 1 on schema errors; routing and knob findings are warnings (exit 0) because some are intended.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KEYS_REQUIRED = {"name", "gpus", "ext", "out_dir", "env", "cmd", "done"}
KEYS_OPTIONAL = {"needs"}
GPUS = {"h100", "v100"}
PLACEHOLDERS = {"{BUDGET}"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


def schema_errors(cells: object) -> list[str]:
    """Structural problems that would make the dispatcher raise or mis-handle a cell."""
    if not isinstance(cells, list):
        return ["top level must be a JSON list of cells"]
    errs, seen = [], collections.Counter()
    for i, c in enumerate(cells):
        where = f"cell {i}"
        if not isinstance(c, dict):
            errs.append(f"{where}: not an object")
            continue
        where = f"cell {i} ({c.get('name', '?')})"
        missing, extra = KEYS_REQUIRED - c.keys(), c.keys() - KEYS_REQUIRED - KEYS_OPTIONAL
        if missing:
            errs.append(f"{where}: missing {sorted(missing)}")
        if extra:
            errs.append(f"{where}: unknown keys {sorted(extra)} (the dispatcher ignores them)")
        n = c.get("name")
        if not (isinstance(n, str) and NAME_RE.match(n)):
            errs.append(f"{where}: name must match {NAME_RE.pattern} (it becomes a Slurm job name)")
        seen[n] += 1
        g = c.get("gpus")
        if not (isinstance(g, list) and g and set(g) <= GPUS):
            errs.append(f"{where}: gpus must be a non-empty list drawn from {sorted(GPUS)}")
        if not (isinstance(c.get("ext"), list) and all(isinstance(x, str) for x in c["ext"])):
            errs.append(f"{where}: ext must be a list of job names")
        od = c.get("out_dir")
        if not (isinstance(od, str) and od and not od.startswith("/")):
            errs.append(f"{where}: out_dir must be a non-empty path relative to the repo root")
        env = c.get("env")
        if not (
            isinstance(env, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
        ):
            errs.append(f"{where}: env must map str -> str (values are pasted into `export K=V`)")
        nd = c.get("needs")
        if nd is not None and not (isinstance(nd, list) and all(isinstance(x, str) for x in nd)):
            errs.append(f"{where}: needs must be null or a list of repo-relative paths")
        cmd = c.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            errs.append(f"{where}: cmd must be a non-empty string")
        else:
            bad = set(re.findall(r"\{[A-Z_]+\}", cmd)) - PLACEHOLDERS
            if bad:
                errs.append(
                    f"{where}: cmd has placeholders {sorted(bad)}; only {sorted(PLACEHOLDERS)} is substituted"
                )
        d = c.get("done")
        if not isinstance(d, dict) or d.get("type") not in ("file", "hab_keys"):
            errs.append(
                f"{where}: done.type must be 'file' or 'hab_keys' (anything else is never done)"
            )
        elif d["type"] == "file" and not isinstance(d.get("path"), str):
            errs.append(f"{where}: done.type=file needs done.path")
        elif d["type"] == "hab_keys" and not {"file", "cfg", "seeds", "arms"} <= d.keys():
            errs.append(f"{where}: done.type=hab_keys needs file, cfg, seeds, arms")
    errs += [
        f"duplicate name {n!r} x{k} (state.json is keyed by name)" for n, k in seen.items() if k > 1
    ]
    return errs


def load_dispatcher():
    """The dispatcher module itself, so routing is judged by its real functions and cannot drift from them.

    Importing it only defines constants and functions (no Slurm calls, no file writes)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dispatcher", HERE / "dispatcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family(name: str) -> str:
    """The arm family a cell belongs to: its first underscore-separated token ('archinvar_dr_30k' -> 'archinvar_')."""
    return name.split("_", 1)[0] + "_" if "_" in name else name


def routing_warnings(cells: list[dict], disp) -> list[str]:
    """Families whose walltime or GPU class follows from a name-prefix default rather than a rule. One line per family."""
    fam = collections.defaultdict(lambda: collections.Counter())
    default_wall = disp.WALL["slow_nice"]
    for c in cells:
        f, retrain = family(c["name"]), "menu_retrain.py" in c["cmd"]
        gpu = (
            "v100" if "v100" in c["gpus"] else c["gpus"][0]
        )  # the slowest class it may use on slow_nice
        wall, _ = disp.wall_and_budget(c, "slow_nice", gpu)
        if retrain and wall == default_wall:
            fam[f][f"retrain cell gets the default slow_nice wall {default_wall}"] += 1
        m = re.search(r"--base_cap_min (\S+)", c["cmd"])
        cap = float(m.group(1) if m else 25) * (5 if gpu == "v100" else 1)
        if retrain and cap + disp.MENU_WALL_MARGIN_MIN > 48 * 60:
            fam[f][
                f"per-model cap {cap:.0f} min on {gpu} does not fit slow_nice's 2-day maximum"
            ] += 1
        if disp.mig_ok(c):
            fam[f][
                "eligible for a 10.8 GB MIG slice (not in MIG_NO): check the largest model fits"
            ] += 1
    return [
        f"{f:<18} {k:>4} cells  {msg}" for f, msgs in sorted(fam.items()) for msg, k in msgs.items()
    ]


def unread_knobs(cells: list[dict], root: Path = REPO) -> list[str]:
    """AL_* variables set in some cell env that no .py file under the repo (or its scratchpad) reads."""
    names = collections.Counter(k for c in cells for k in c.get("env", {}) if k.startswith("AL_"))
    src = "\n".join(
        p.read_text(errors="ignore")
        for d in ("albench", "experiments", "scripts", "orchestration", "scratchpad")
        for p in (root / d).rglob("*.py")
        if p.is_file()
    )
    return [
        f"{k} set in {n} cells, read by no .py file"
        for k, n in sorted(names.items())
        if f'"{k}"' not in src and f"'{k}'" not in src
    ]


def path_warnings(cells: list[dict], root: Path) -> list[str]:
    """Scripts that cells invoke but that do not exist under root (a cell would fail in seconds, 3 times, then pause)."""
    missing = collections.Counter()
    for c in cells:
        for s in re.findall(
            r"(?:^|\s)((?:scratchpad|scripts|experiments|orchestration)/\S+\.py)", c["cmd"]
        ):
            if not (root / s).exists():
                missing[s] += 1
    return [
        f"{s} invoked by {n} cells does not exist under {root}" for s, n in missing.most_common()
    ]


def is_done(c: dict, root: Path) -> bool:
    """The dispatcher's own completion test (dispatcher.is_done), against an explicit root."""
    d = c["done"]
    if d["type"] == "file":
        return (root / d["path"]).exists()
    have, p = set(), root / d["file"]
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            have.add((r["cfg"] if "cfg" in r else f"F{r.get('arch')}", r["seed"], r["arm"]))
    return {(d["cfg"], s, a) for s in d["seeds"] for a in d["arms"]} <= have


def collapse(lines: list[str]) -> list[str]:
    """One line per kind of finding, listing the families, for a whole-registry run."""
    by = collections.defaultdict(list)
    for ln in lines:
        fam_, rest = ln.split(None, 1)
        k, msg = rest.split(" cells  ", 1)
        by[msg].append(f"{fam_}({k.strip()})")
    return [f"{len(v)} families: {msg}: {' '.join(v)}" for msg, v in by.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("registry", type=Path)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repo root (on the cluster: .): routing checks then cover only cells not yet done, "
        "and invoked scripts must exist",
    )
    args = ap.parse_args()
    cells = json.loads(args.registry.read_text())
    errs = schema_errors(cells)
    for e in errs:
        print("ERROR  ", e)
    if errs:
        print(f"{len(errs)} schema errors; routing checks skipped")
        return 1
    if args.root:
        todo = [c for c in cells if not is_done(c, args.root)]
        route = routing_warnings(todo, load_dispatcher())
        scope = f"{len(todo)} cells not done"
    else:
        route = collapse(routing_warnings(cells, load_dispatcher()))
        scope = "all cells, done or not (pass --root to check only pending ones)"
    warns = route + unread_knobs(cells, args.root or REPO)
    if args.root:
        warns += path_warnings(todo, args.root)
    for w in warns:
        print("WARN   ", w)
    print(f"{len(cells)} cells: schema OK; routing checked on {scope}; {len(warns)} warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
