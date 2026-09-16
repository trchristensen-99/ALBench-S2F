"""Does a pool on disk carry the parameters the plan now asks for?

Exists because a config edit silently leaves stale pools behind. The tuned evoaug arm
had to be spotted by hand and deleted; without a check like this, any later change to a
strategy's parameters would keep reusing data generated under the OLD ones and the
resulting curve would be mislabelled rather than wrong-looking.

Compares against the registry default when the plan omits a parameter, so a pool
stamped {} is equivalent to one stamped with the defaults spelled out. Exit 0 = the
pool matches and can be reused, 1 = regenerate it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    p = Path(args.pool)
    if not p.exists():
        print(f"  {p.name}: absent")
        return 1

    from albench.registry import REGISTRY

    spec = REGISTRY.get(args.strategy)
    if spec is None:
        print(f"  unknown strategy {args.strategy!r}")
        return 1

    wanted = dict(kv.split("=", 1) for kv in args.set)
    with np.load(p, allow_pickle=True) as z:
        raw = str(z["params"]) if "params" in z.files else "{}"
    try:
        stamped = json.loads(raw)
    except Exception:
        stamped = {}

    def norm(v):
        s = str(v)
        try:
            return float(s)
        except ValueError:
            return s

    # Every parameter the strategy has, resolved: explicit value else registry default.
    for key, param in spec.params.items():
        want = norm(wanted.get(key, param.default))
        have = norm(stamped.get(key, param.default))
        if want != have:
            print(f"  {p.name}: {key} is {have!r} on disk but the plan wants {want!r}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
