"""Teach the shared loader about the 1M pools and the mixtures directory.

Publishing `motif_shared_core_d1000000.npz` beside `_d300000.npz` broke `load_pool`:
it picked the largest file by `sorted()`, which is LEXICAL, and "d1000000" sorts before
"d300000". A caller asking for the 1M pool would have been handed the 300k file and told
it holds 300,000 sequences. Sizes are now parsed and compared as integers.

Two smaller gaps closed at the same time: the mixture pools live in a subdirectory the
glob never looked in, and `list_pools()` read only the Nov-8 manifest, so the new files
were invisible to `python albench_pools.py` and absent from the not-published error
message.

Idempotent: re-running leaves an already-patched file alone.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

P = Path("/grid/koo/home/shared/albench/reservoir_pools/albench_pools.py")

OLD_LIST = '''def list_pools() -> dict:
    """Every published reservoir, with n, labels, GC and provenance."""
    return json.loads((POOLS / "MANIFEST.json").read_text())["reservoir_pools"]'''

NEW_LIST = '''def _pool_files(strategy: str) -> list[tuple[int, Path]]:
    """Every published file for a strategy, newest-convention first, sorted by SIZE.

    Sorting by name would be wrong: "..._d1000000.npz" sorts before "..._d300000.npz"
    lexically, so the largest pool is not the last name. Mixtures live in their own
    subdirectory and are searched here too, so `load_pool("mix_ms_mc_c")` just works.
    """
    out = []
    for root in (POOLS, POOLS / "mixtures"):
        for p in root.glob(f"{strategy}_d*.npz"):
            tail = p.name.rsplit("_d", 1)[1][: -len(".npz")]
            if tail.isdigit():
                out.append((int(tail), p))
    return sorted(out)


def list_pools() -> dict:
    """Every published reservoir, with n, labels, GC and provenance."""
    out = dict(json.loads((POOLS / "MANIFEST.json").read_text())["reservoir_pools"])
    extra = POOLS / "MANIFEST_nov21.json"
    if extra.exists():
        for e in json.loads(extra.read_text())["files"]:
            name = Path(e["file"]).name[: -len(".npz")]
            out.setdefault(name, {
                "strategy": e.get("strategy") or name,
                "n": e.get("n"),
                "group": "added_nov21",
                "prefix_sizes": e.get("all_sizes") or [],
                "doc": e.get("nesting_note", ""),
                "file": e["file"],
            })
    return out'''

OLD_HITS = """    hits = sorted(POOLS.glob(f"{strategy}_d*.npz"))
    if not hits:
        raise FileNotFoundError(f"{strategy}: not published; have {sorted(list_pools())}")
    seqs, y = _open(hits[-1], n)"""

NEW_HITS = """    hits = _pool_files(strategy)
    if not hits:
        raise FileNotFoundError(f"{strategy}: not published; have {sorted(list_pools())}")
    # Largest by size, and if a specific n is asked for, the smallest file that holds it --
    # every size is a prefix, so a smaller file gives identical sequences and loads faster.
    pick = hits[-1][1]
    if n is not None:
        fits = [p for d, p in hits if d >= n]
        if fits:
            pick = fits[0]
    seqs, y = _open(pick, n)"""

OLD_MAIN = """        sizes = "/".join(f"{s // 1000}k" for s in info["prefix_sizes"])
        print(f"  {info['strategy']:20s} {info['n']:>7,}  prefixes {sizes:16s} {info['doc']}")"""

NEW_MAIN = """        sizes = "/".join(f"{s // 1000}k" for s in info.get("prefix_sizes") or [])
        print(
            f"  {info.get('strategy', name):24s} {info.get('n') or 0:>9,}  "
            f"prefixes {sizes:34s} {info.get('doc', '')}"
        )"""

OLD_SORT = 'for name, info in sorted(pools.items(), key=lambda kv: (kv[1]["group"], kv[0])):'
NEW_SORT = (
    'for name, info in sorted(pools.items(), key=lambda kv: (kv[1].get("group", ""), kv[0])):'
)


def main() -> None:
    src = P.read_text()
    if "_pool_files" in src:
        print("already patched")
        return
    for old, new in (
        (OLD_LIST, NEW_LIST),
        (OLD_HITS, NEW_HITS),
        (OLD_SORT, NEW_SORT),
        (OLD_MAIN, NEW_MAIN),
    ):
        if old not in src:
            sys.exit("patch target missing, refusing to edit:\n%s" % old[:120])
        src = src.replace(old, new, 1)
    shutil.copy2(P, P.with_suffix(".py.bak_nov21"))
    P.write_text(src)
    P.chmod(0o644)
    P.with_suffix(".py.bak_nov21").chmod(0o644)
    print("patched %s (backup .bak_nov21)" % P)


if __name__ == "__main__":
    main()
