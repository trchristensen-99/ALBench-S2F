#!/usr/bin/env python
"""Hourly token usage of every Claude Code session found under one or more `~/.claude/projects` roots, to see what fills the account's 5-hour session limit.

The AutoResearch loop calls the Claude Code CLI (`claude -p`) from the cluster with the lab account's token, and the same account is used interactively; a "session limit" message
(`You've hit your session limit · resets 4:50pm`) stops both. Every assistant message in a session log carries its `usage`; this sums them per hour (America/New_York), per root and per model,
de-duplicating repeated log lines by (message id, request id). Output: JSON {root: {hour: {model: {calls, input, output, cache_create, cache_read}}}}.

    python scripts/analysis/llm_usage_windows.py --root ~/.claude/projects --since 2026-10-29 --out usage_local.json
    python scripts/analysis/llm_usage_windows.py --root /grid/koo/home/christen/.claude/projects --root /grid/wsbs/home_norepl/christen/.claude/projects --since 2026-10-29 --out usage_cluster.json   (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def scan(root: Path, since: float) -> dict:
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    seen: set = set()
    for f in root.rglob("*.jsonl"):
        try:
            if f.stat().st_mtime < since:
                continue
            kind = "subagent" if "subagents" in f.parts else "session"
            proj = f.relative_to(root).parts[0]
            tag = (
                "isolated_cli"
                if proj.startswith("-tmp-llm-cli")
                else ("repo_cli" if "ALBench" in proj else proj[:30])
            )
            with open(f, errors="replace") as fh:
                for line in fh:
                    if '"usage"' not in line:
                        continue
                    try:
                        o = json.loads(line)
                    except ValueError:
                        continue
                    m = o.get("message") or {}
                    u = m.get("usage")
                    if not u or o.get("type") != "assistant":
                        continue
                    k = (m.get("id"), o.get("requestId"))
                    if k in seen:
                        continue
                    seen.add(k)
                    ts = datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")).astimezone(
                        NY
                    )
                    if ts.timestamp() < since:
                        continue
                    hour = ts.strftime("%Y-%m-%d %H:00")
                    d = out[f"{tag}/{kind}"][hour][m.get("model") or "?"]
                    d["calls"] += 1
                    d["input"] += u.get("input_tokens") or 0
                    d["output"] += u.get("output_tokens") or 0
                    d["cache_create"] += u.get("cache_creation_input_tokens") or 0
                    d["cache_read"] += u.get("cache_read_input_tokens") or 0
        except Exception:  # noqa: BLE001 - one unreadable log must not lose the rest
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--since", default="2026-10-29")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=NY).astimezone(timezone.utc).timestamp()
    res: dict = {}
    for r in a.root:
        p = Path(os.path.expanduser(r))
        for k, v in scan(p, since).items():
            res[f"{p}:{k}"] = v
    json.dump(res, open(a.out, "w"))
    print(
        "wrote",
        a.out,
        {k: sum(x["calls"] for h in v.values() for x in h.values()) for k, v in res.items()},
    )


if __name__ == "__main__":
    main()
