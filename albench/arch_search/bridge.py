"""Cluster-side half of the LLM bridge.

The cluster cannot call Claude: the organisation disabled Claude Code subscription
auth there, and an API key bills separately from the plan. But the CLI works on the
researcher's laptop against the normal plan. So instead of paying twice, the search
loop writes a REQUEST file and blocks; a poller on the laptop serves it and writes
the RESPONSE back over the same shared filesystem.

Requests are content-addressed, so a resumed job re-uses an answer it already has
rather than spending plan usage on a question that was already asked.

This is deliberately filesystem-based: no inbound network to the cluster, no
daemon listening, and the whole thing degrades to "jobs wait" when the laptop is
closed rather than failing the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get(
    "LLM_BRIDGE_DIR",
    "/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/llm_bridge"))
REQ, RESP, CLAIMED = ROOT / "requests", ROOT / "responses", ROOT / "claimed"


def _key(system: str, user: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (model, system, user):
        h.update(part.encode()); h.update(b"\x00")
    return h.hexdigest()[:20]


def request_completion(system: str, user: str, model: str = "sonnet",
                       timeout_s: float = 3600, poll_s: float = 5.0,
                       log=print) -> str:
    """Ask the laptop-side poller for a completion; block until it answers.

    Raises TimeoutError if nobody serves it — the caller should treat that the
    same as any other LLM failure (fail soft, do not silently fall back to random).
    """
    for d in (REQ, RESP, CLAIMED):
        d.mkdir(parents=True, exist_ok=True)
    k = _key(system, user, model)
    rpath = RESP / f"{k}.json"
    if rpath.exists():                      # already answered in a previous run
        return json.loads(rpath.read_text())["text"]

    qpath = REQ / f"{k}.json"
    if not qpath.exists():
        tmp = qpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"key": k, "model": model, "system": system, "user": user,
             "created": time.time(), "host": os.uname().nodename}, indent=2))
        os.replace(tmp, qpath)
        log(f"  [bridge] queued request {k} ({len(user)} chars) — waiting for laptop poller")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if rpath.exists():
            payload = json.loads(rpath.read_text())
            if payload.get("error"):
                raise RuntimeError(f"bridge returned error: {payload['error'][:200]}")
            log(f"  [bridge] got response {k} after {time.time()-t0:.0f}s")
            return payload["text"]
        time.sleep(poll_s)
    raise TimeoutError(
        f"no response for request {k} after {timeout_s:.0f}s. Is the laptop poller "
        f"running? (scripts/llm_bridge_poller.sh)")


def pending() -> list[str]:
    return sorted(p.stem for p in REQ.glob("*.json")
                  if not (RESP / f"{p.stem}.json").exists())


class BridgeUnavailable(RuntimeError):
    """The laptop poller did not answer within the budget.

    Raised instead of returning something usable, because the alternative --
    silently substituting a random config -- would corrupt the arm being measured.
    An AutoResearch arm that quietly becomes random search is worse than an arm
    that stops: the first produces a wrong answer, the second produces none.
    The caller should checkpoint and exit so a watchdog can resume later.
    """


def request_or_pause(system: str, user: str, model: str = "sonnet",
                     timeout_s: float = 1800, log=print) -> str:
    """request_completion, but raise BridgeUnavailable instead of TimeoutError.

    Overnight the laptop may sleep or the VPN may drop. Progress is already
    durable -- every finished model has written its npz + meta -- so the correct
    response to a dead bridge is to stop cleanly and let the watchdog retry, not
    to burn GPU hours on configs nobody proposed.
    """
    try:
        return request_completion(system, user, model=model, timeout_s=timeout_s, log=log)
    except TimeoutError as e:
        raise BridgeUnavailable(str(e)) from e


def prefetch(system: str, user: str, model: str = "sonnet") -> str | None:
    """Queue a request without waiting; returns the cached answer if present.

    Lets a round ask for the NEXT round proposals while the current one trains,
    so a serve that takes 40s costs nothing once the pipeline is primed.
    """
    for d in (REQ, RESP, CLAIMED):
        d.mkdir(parents=True, exist_ok=True)
    k = _key(system, user, model)
    rpath = RESP / f"{k}.json"
    if rpath.exists():
        return json.loads(rpath.read_text()).get("text")
    qpath = REQ / f"{k}.json"
    if not qpath.exists():
        tmp = qpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": k, "model": model, "system": system,
                                   "user": user, "created": time.time(),
                                   "host": os.uname().nodename}, indent=2))
        os.replace(tmp, qpath)
    return None
