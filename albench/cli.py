"""Command line entry point: ``albench <command>`` (or ``python -m albench.cli``).

Four commands, each answering one question a newcomer actually has:

  doctor    which data assets do I have, and how do I get the missing ones?
  list      what strategies exist and what can I tune on each?
  generate  make me N sequences from one strategy with these parameters
  sweep     expand a parameter grid into one job per combination

``generate`` and ``sweep`` take parameters as ``--set key=value``, so no new file
or code change is needed to try a parameter that nobody has tried before. That is
the whole point of the registry: the parameter screen is a config, not a codebase
edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _fmt_default(v):
    return "None" if v is None else repr(v) if isinstance(v, str) else str(v)


def cmd_doctor(args) -> int:
    from albench.paths import ASSETS, data_root, status

    print(f"data root: {data_root()}   (override with ALBENCH_DATA)\n")
    rows = status()
    width = max(len(k) for k, _, _ in rows)
    missing = []
    for key, present, info in rows:
        mark = "OK  " if present else "MISS"
        print(f"  [{mark}] {key:<{width}}  {info}")
        if not present:
            missing.append(key)
    if not missing:
        print("\nAll assets present.")
        return 0
    print(f"\n{len(missing)} missing. What each one is and how to get it:\n")
    for key in missing:
        a = ASSETS[key]
        print(f"  {key}")
        print(f"    {a.description}")
        print(f"    env override: ${a.env_var}")
        print(f"    obtain      : {a.how_to_get}\n")
    print(
        "Not all assets are needed for all strategies -- `albench list` shows which\n"
        "strategy needs which asset. You can start work with a subset."
    )
    return 1


def cmd_list(args) -> int:
    from albench.registry import ACQ_REGISTRY, REGISTRY

    which = {
        "reservoir": [REGISTRY],
        "acquisition": [ACQ_REGISTRY],
        "all": [REGISTRY, ACQ_REGISTRY],
    }[args.kind]
    groups: dict[str, list] = {}
    for reg in which:
        kind = "RESERVOIR" if reg is REGISTRY else "ACQUISITION"
        for name, spec in sorted(reg.items()):
            groups.setdefault(f"{kind} / {spec.group or 'other'}", []).append((name, spec))

    for group, items in groups.items():
        print(f"\n{'=' * 78}\n{group.upper()}\n{'=' * 78}")
        for name, spec in items:
            print(f"\n  {name}")
            for line in _wrap(spec.doc, 72):
                print(f"      {line}")
            if spec.assets:
                print(f"      needs assets: {', '.join(spec.assets)}")
            if not spec.params:
                print("      (no tunable parameters)")
                continue
            print("      parameters:")
            w = max(len(k) for k in spec.params)
            for k, p in spec.params.items():
                choices = f"  [{'|'.join(map(str, p.choices))}]" if p.choices else ""
                print(f"        {k:<{w}} = {_fmt_default(p.default):<22} {p.help}{choices}")
    print()
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width) or [""]


def _parse_sets(pairs: list[str]) -> dict:
    """``--set k=v`` and ``--set k=v1,v2`` (a list, which sweeps)."""
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        vals = [_coerce(x) for x in v.split(",")] if "," in v else _coerce(v)
        out[k.strip()] = vals
    return out


def _coerce(v: str):
    v = v.strip()
    low = v.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _load_pool(task: str, limit: int | None):
    """Genomic pool for strategies that derive from real sequence."""
    from albench.paths import resolve

    # required=True on purpose. This is only called for strategies that DECLARE
    # needs_pool, so a missing pool is fatal for them -- and returning (None, None)
    # deferred the failure until deep inside the sampler, where it surfaced as
    # "strategy needs a genomic pool but none was supplied" with no hint that the
    # real problem was an unresolvable asset. Ask loudly, and resolve() explains how
    # to obtain it.
    p = resolve("bg_cache", required=True)
    z = np.load(p, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"][: limit or None]]
    labels = None
    for k in ("oracle_labels", "oracle_mean", "labels"):
        if k in z.files:
            labels = np.asarray(z[k], dtype=np.float32)[: limit or None]
            break
    return seqs, labels


def _dedupe_to_target(spec, seqs, target, ctx, seed, params, allow_duplicates):
    """Return `target` DISTINCT sequences, or as many as the strategy can supply.

    WHY THIS IS CENTRAL RATHER THAN PER-SAMPLER. A pool containing the same sequence
    twice is a silent confound: the duplicate is weighted twice in training and, once
    labelled, twice in any average, so a strategy whose source has fewer distinct
    windows than the requested draw looks like it supplied more information than it
    did. Measured before this existed: zoonomia returned 3,234 duplicate copies in a
    300k draw (1.08%, one sequence appearing six times) because its ortholog source
    holds fewer than 300k distinct windows.

    Top-up rounds re-draw with a DERIVED seed so the extra sequences are not the same
    draw again, and stop as soon as a round adds nothing -- that is the signal the
    source is exhausted, and the honest response is to return fewer sequences and say
    so, not to loop forever or pad with repeats.
    """
    seen, uniq = set(), []
    for x in seqs:
        t = str(x)
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    dropped = len(seqs) - len(uniq)
    if allow_duplicates:
        if dropped:
            print(f"  NOTE: {dropped:,} duplicate sequences kept (--allow-duplicates)")
        return list(seqs)
    if dropped == 0:
        return uniq

    print(f"  {dropped:,} duplicates removed; topping up to {target:,} distinct")
    for attempt in range(1, 6):
        need = target - len(uniq)
        if need <= 0:
            break
        extra, _ = spec.generate(
            min(need * 3, max(need, 10_000)), ctx, seed=seed + 1000 * attempt, **params
        )
        before = len(uniq)
        for x in extra:
            t = str(x)
            if t not in seen and len(uniq) < target:
                seen.add(t)
                uniq.append(t)
        gained = len(uniq) - before
        print(f"    top-up {attempt}: +{gained:,} distinct (now {len(uniq):,}/{target:,})")
        if gained == 0:
            print("    source exhausted -- no further distinct sequences available")
            break
    if len(uniq) < target:
        print(
            f"  CAPACITY LIMIT: {len(uniq):,} distinct of {target:,} requested. "
            f"This strategy cannot supply more; treat its largest curve point as "
            f"capacity-limited rather than comparable."
        )
    return uniq


def cmd_generate(args) -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from albench.registry import Context, expand_sweep, get

    spec = get(args.strategy)
    params = _parse_sets(args.set)
    combos = expand_sweep(params)
    if len(combos) > 1:
        raise SystemExit(
            f"--set expanded to {len(combos)} combinations. Use `albench sweep` for grids."
        )
    pool, labels = _load_pool(args.task, args.pool_limit) if spec.needs_pool else (None, None)
    ctx = Context(task=args.task, pool_sequences=pool, pool_labels=labels)

    seqs, _meta = spec.generate(args.n, ctx, seed=args.seed, **combos[0])
    seqs = _dedupe_to_target(spec, seqs, args.n, ctx, args.seed, combos[0], args.allow_duplicates)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sequences=np.array(seqs, dtype=object),
        strategy=args.strategy,
        # Params are stamped INTO the artefact: a cache built six weeks ago can be
        # told apart from one built today without consulting anyone's notes.
        params=json.dumps(combos[0], default=str),
        seed=args.seed,
    )
    print(f"\nwrote {len(seqs):,} sequences -> {out}")
    print(f"  strategy={args.strategy}  seed={args.seed}  params={combos[0]}")
    return 0


def cmd_sweep(args) -> int:
    from albench.registry import expand_sweep, get

    spec = get(args.strategy)  # validates the name before emitting any commands
    params = _parse_sets(args.set)
    unknown = set(params) - set(spec.params)
    if unknown:
        raise SystemExit(
            f"{args.strategy}: unknown parameter(s) {sorted(unknown)}.\n"
            f"Tunable: {sorted(spec.params)}  (see `albench list`)"
        )
    combos = expand_sweep(params)
    print(f"{args.strategy}: {len(combos)} combination(s)")
    lines = []
    for i, c in enumerate(combos):
        sets = " ".join(f"--set {k}={v}" for k, v in c.items())
        tag = "_".join(f"{k}{v}" for k, v in c.items()).replace(".", "p")[:80]
        out = f"{args.out_dir}/{args.strategy}__{tag}__seed{args.seed}.npz"
        lines.append(
            f"albench generate --strategy {args.strategy} --n {args.n} "
            f"--seed {args.seed} {sets} --out {out}"
        )
        print(f"  [{i + 1:>3}] {c}")
    if args.write:
        Path(args.write).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {len(lines)} commands -> {args.write}")
    return 0


def cmd_screen(args) -> int:
    import yaml

    from albench.run import ScreenConfig, expand_screen, screen_summary

    cfg = ScreenConfig.from_dict(yaml.safe_load(Path(args.config).read_text()))
    cells = expand_screen(cfg, args.stage)
    print(screen_summary(cells, epochs=args.epochs))
    if args.limit:
        cells = cells[: args.limit]
        print(f"\n(limited to the first {len(cells)} cells)")

    lines = []
    for c in cells:
        sets = " ".join(f"--set {k}={v}" for k, v in sorted(c.params.items()))
        out = f"{args.out_dir}/{c.tag}.npz"
        lines.append(
            f"albench generate --strategy {c.reservoir} --n {c.d} --seed {c.seed} "
            f"{sets} --out {out}".replace("  ", " ")
        )
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {len(lines)} reservoir-generation commands -> {args.write}")
        print("Each produces one labelled-pool input; training is driven separately.")
    else:
        for ln in lines[:5]:
            print("  " + ln)
        if len(lines) > 5:
            print(f"  ... and {len(lines) - 5} more (use --write to save them all)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="albench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check data assets and print how to get missing ones")
    ls = sub.add_parser("list", help="list strategies and their tunable parameters")
    ls.add_argument("--kind", choices=("all", "reservoir", "acquisition"), default="all")

    g = sub.add_parser("generate", help="generate sequences from one strategy")
    g.add_argument("--strategy", required=True)
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--task", default="k562", choices=["k562", "yeast"])
    g.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a parameter")
    g.add_argument("--out", required=True)
    g.add_argument("--pool-limit", type=int, default=None, help="cap pool size (for quick tests)")
    g.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="keep duplicate sequences instead of deduplicating and topping up. Off by "
        "default: duplicates are weighted twice in training and in any label average.",
    )

    s = sub.add_parser("sweep", help="expand a parameter grid into one command per combination")
    s.add_argument("--strategy", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--set", action="append", metavar="KEY=V1,V2", help="comma-separated = sweep")
    s.add_argument("--out-dir", default="outputs/reservoir_cache")
    s.add_argument("--write", default=None, help="write the commands to this file")

    sc = sub.add_parser("screen", help="expand a parameter-screen config into jobs")
    sc.add_argument("--config", required=True)
    sc.add_argument("--stage", choices=("screen", "factorial"), default="screen")
    sc.add_argument("--out-dir", default="outputs/screen")
    sc.add_argument("--write", default=None)
    sc.add_argument("--limit", type=int, default=None, help="emit only the first N cells")
    sc.add_argument("--epochs", type=int, default=60, help="for the cost estimate")

    args = ap.parse_args(argv)
    return {
        "doctor": cmd_doctor,
        "list": cmd_list,
        "generate": cmd_generate,
        "sweep": cmd_sweep,
        "screen": cmd_screen,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
