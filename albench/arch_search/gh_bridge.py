"""LLM bridge over GitHub — the VPN-independent transport.

The filesystem bridge required the laptop to SSH into the cluster, so a VPN drop
(or an overnight Duo prompt nobody is awake to answer) stopped the search. Both
machines can reach github.com directly, so routing the queue through a private
repo removes the VPN from the path entirely: the cluster pushes a request, the
laptop polls GitHub and pushes the answer back, and neither needs the other to be
reachable.

Verified prerequisites: the cluster resolves api.github.com (HTTP 200) and has an
authenticated `gh`.

Requests are content-addressed, so a resumed job reuses an answer already on
GitHub rather than spending plan usage asking again.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time

REPO = os.environ.get("LLM_BRIDGE_REPO", "trchristensen-99/albench-llm-bridge")


ASK_AHEAD = int(os.environ.get("LLM_ASK_AHEAD", "1"))   # 1 = off; N = ask N rounds at once
BANK = os.path.expanduser("~/.config/albench/llm_bank")


def bank_take(tag: str) -> str | None:
    """Pop one banked response for this cell, if any."""
    import glob as _g
    files = sorted(_g.glob(os.path.join(BANK, f"{tag}_*.txt")))
    if not files:
        return None
    f = files[0]
    try:
        txt = open(f).read()
        os.remove(f)
        return txt
    except Exception:
        return None


def bank_put(tag: str, texts: list[str]) -> int:
    os.makedirs(BANK, exist_ok=True)
    n = 0
    for i, t in enumerate(texts):
        p = os.path.join(BANK, f"{tag}_{int(time.time())}_{i:03d}.txt")
        try:
            open(p, "w").write(t); n += 1
        except Exception:
            pass
    return n


class BridgeUnavailable(RuntimeError):
    """Nobody served the request in time. Caller should checkpoint and exit."""


def _key(system: str, user: str, model: str, nonce: str = "") -> str:
    """Content address, plus a caller-supplied nonce.

    Pure content addressing was wrong for an active search. When history failed to
    reload after a resume the prompt came back byte-identical, the cache returned
    the SAME proposal, and a 60-round cell trained one config 58 times while
    looking like it had explored 60. The nonce (the cell plus how many models it
    has already produced) keeps genuine crash-resume cheap -- same round, same
    key -- while guaranteeing a NEW round always asks a new question.
    """
    h = hashlib.sha256()
    for p in (model, system, user, nonce):
        h.update(p.encode()); h.update(b"\x00")
    return h.hexdigest()[:20]


def _token() -> str:
    """GitHub token from a 0600 file on shared storage.

    Deliberately NOT `gh`: that binary exists only on the login node, so every
    compute job died with FileNotFoundError. The REST API is plain HTTPS and
    python is present everywhere, which removes the dependency entirely.
    """
    for p in (os.environ.get("ALBENCH_GH_TOKEN_FILE"),
              os.path.expanduser("~/.config/albench/gh_token")):
        if p and os.path.exists(p):
            t = open(p).read().strip()
            if t:
                return t
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    raise BridgeUnavailable(
        "no GitHub token: expected ~/.config/albench/gh_token (0600) or GH_TOKEN")


def _api(path: str, method: str = "GET", payload: dict | None = None,
         timeout: int = 30, _retry: bool = False):
    import urllib.error
    import urllib.request

    url = f"https://api.github.com/repos/{REPO}/{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "albench-llm-bridge",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _note_rate(r.headers)
            body = r.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        _note_rate(e.headers)
        if e.code == 404:
            return None
        # A 403 with zero budget left is not a permissions failure, it is the
        # hourly quota. Crashing here kills a GPU cell for something that fixes
        # itself on a clock, so wait it out once and retry.
        if e.code == 403 and _RATE.get("remaining") == 0 and not _retry:
            wait = max(0.0, _RATE.get("reset", 0.0) - time.time()) + 5.0
            if 0 < wait <= 3900:
                print(f"  [bridge] hourly API quota exhausted; waiting "
                      f"{wait/60:.1f} min for reset", flush=True)
                time.sleep(wait)
                return _api(path, method=method, payload=payload,
                            timeout=timeout, _retry=True)
        raise
    except Exception:
        return None



# The GitHub REST budget is 5,000 requests/hour PER USER, shared by every search
# cell on the cluster. Twelve cells polling every 20s exhausted it in under an
# hour and every LLM arm died at once. Track the budget from response headers so
# the waiters can slow down before they hit zero instead of after.
_RATE_LIMITED_RETRY = True
_CALLS = {"n": 0}
_RATE = {"remaining": None, "reset": 0.0}


def _note_rate(headers) -> None:
    _CALLS["n"] += 1
    if _CALLS["n"] % 100 == 0:
        print(f"  [bridge] {_CALLS['n']} API calls this process "
              f"(budget remaining {_RATE.get('remaining')})", flush=True)
    try:
        rem = headers.get("x-ratelimit-remaining")
        rst = headers.get("x-ratelimit-reset")
        if rem is not None:
            _RATE["remaining"] = int(rem)
        if rst is not None:
            _RATE["reset"] = float(rst)
    except Exception:
        pass


def rate_state() -> dict:
    return dict(_RATE)


def _rate_guard(reserve: int = 150) -> None:
    """Block until the budget refills if we are close to exhausting it.

    `reserve` keeps headroom for the PUTs that actually submit work, so a waiter
    polling in a loop can never starve a sibling cell that needs to post a result.
    """
    rem = _RATE.get("remaining")
    if rem is None or rem > reserve:
        return
    wait = max(0.0, _RATE.get("reset", 0.0) - time.time()) + 5.0
    if wait > 0:
        print(f"  [bridge] API budget {rem} left; sleeping {wait/60:.1f} min "
              f"until the hourly reset", flush=True)
        time.sleep(min(wait, 3900))



def _beat_stale(age: float | None, stale_s: float) -> bool:
    """True only if a stale heartbeat really means the laptop is gone.

    The poller publishes its heartbeat through the SAME per-user GitHub quota the
    search cells spend. When the quota is exhausted the poller cannot beat either,
    so its beat goes stale while the laptop is wide awake and still serving. Cells
    used to read that as "laptop asleep" and exit, turning one quota exhaustion
    into a cluster-wide shutdown of every LLM arm.

    If our own budget is spent, the quota explains the stale beat; do not blame
    the laptop.
    """
    if age is None or age <= stale_s:
        return False
    # The poller spends ~3 calls per cycle from the SAME budget, so it stops
    # beating well before the budget reaches zero -- it loses the race for the
    # last few hundred calls while a cheap single read from here still succeeds.
    # Treat a thin budget, not just an empty one, as explaining the stale beat.
    rem = _RATE.get("remaining")
    if rem is not None and rem < 500:
        return False
    return True


def _get(path: str) -> str | None:
    d = _api(f"contents/{path}")
    if not d or "content" not in d:
        return None
    try:
        return base64.b64decode(d["content"]).decode()
    except Exception:
        return None


_PUT_ATTEMPTS = 6


def _put(path: str, text: str, message: str) -> bool:
    """Write a file, retrying the read-sha/PUT race.

    The GitHub contents API needs the CURRENT sha to update a file, so this is a
    read-modify-write. With many search cells writing concurrently another writer
    can land between our read and our PUT, and GitHub rejects the stale sha with
    409 Conflict. One such 409 used to kill a whole cell, because LLM_FAIL_HARD=1
    (correctly) refuses to fall back to random search.

    Re-read the sha and retry with exponential backoff plus jitter; the jitter
    matters because the colliding writers are cells that all started together and
    would otherwise retry in lockstep.
    """
    import random
    import urllib.error

    for attempt in range(_PUT_ATTEMPTS):
        body = {"message": message,
                "content": base64.b64encode(text.encode()).decode()}
        cur = _api(f"contents/{path}")
        if cur and cur.get("sha"):      # updating an existing file needs its sha
            body["sha"] = cur["sha"]
        try:
            return _api(f"contents/{path}", method="PUT", payload=body) is not None
        except urllib.error.HTTPError as e:
            if e.code not in (409, 422) or attempt == _PUT_ATTEMPTS - 1:
                raise
            time.sleep(min(8.0, 0.4 * 2 ** attempt) * (0.5 + random.random()))
    return False


def bridge_age_s() -> float | None:
    """Seconds since the laptop poller last published a heartbeat, or None.

    Lets a job tell a BUSY bridge from a SHUT laptop: the poller beats before and
    after each request it serves, so a long backlog keeps the beat fresh while a
    closed lid lets it go stale.
    """
    raw = _get("heartbeat.json")
    if not raw:
        return None
    try:
        return max(0.0, time.time() - float(json.loads(raw)["ts"]))
    except Exception:
        return None


def request_completion(system: str, user: str, model: str = "sonnet",
                       timeout_s: float = 5400, poll_s: float = 45.0,
                       stale_s: float = 1500, nonce: str = "", log=print) -> str:
    k = _key(system, user, model, nonce)
    cached = _get(f"responses/{k}.json")
    if cached:
        payload = json.loads(cached)
        if payload.get("text"):
            return payload["text"]

    if not _get(f"requests/{k}.json"):
        ok = _put(f"requests/{k}.json",
                  json.dumps({"key": k, "model": model, "system": system, "user": user,
                              "created": time.time(), "host": os.uname().nodename}, indent=2),
                  f"request {k}")
        if not ok:
            raise BridgeUnavailable(f"could not push request {k} to {REPO}")
        log(f"  [gh-bridge] pushed request {k} ({len(user)} chars)")

    # Distinguish a SLOW bridge from a SHUT laptop. Waiting the full timeout on a
    # laptop that is closed holds a GPU idle for an hour and a half and repeats on
    # every resubmit; a stale heartbeat means nobody is listening, so give the slot
    # back immediately and let the watchdog retry once the poller is alive again.
    age = bridge_age_s()
    if _beat_stale(age, stale_s):
        raise BridgeUnavailable(
            f"poller heartbeat is {age/60:.0f} min old (> {stale_s/60:.0f} min): "
            f"laptop is asleep or the poller is down. Exiting so the GPU is freed; "
            f"the watchdog will resume this cell.")

    t0 = time.time()
    # Poll interval GROWS while waiting. A fixed 20s across a dozen concurrent
    # cells spends the whole hourly API budget on "not ready yet"; the answer
    # typically takes minutes, so the early polls are nearly all wasted.
    cur_poll = poll_s
    last_hb = 0.0
    while time.time() - t0 < timeout_s:
        _rate_guard()
        # re-check mid-wait: the laptop may close after we started waiting.
        # Tied to elapsed time rather than loop count, since the interval grows.
        if time.time() - last_hb > 300:
            last_hb = time.time()
            age = bridge_age_s()
            if _beat_stale(age, stale_s):
                raise BridgeUnavailable(
                    f"poller went silent mid-wait ({age/60:.0f} min). Exiting to free the GPU.")
        raw = _get(f"responses/{k}.json")
        if raw:
            payload = json.loads(raw)
            if payload.get("error"):
                raise BridgeUnavailable(f"bridge error: {payload['error'][:200]}")
            if payload.get("text"):
                log(f"  [gh-bridge] answered {k} after {time.time()-t0:.0f}s")
                return payload["text"]
        time.sleep(cur_poll)
        cur_poll = min(cur_poll * 1.4, 120.0)
    raise BridgeUnavailable(
        f"no response for {k} after {timeout_s:.0f}s — is the laptop poller running?")
