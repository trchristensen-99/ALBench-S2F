#!/usr/bin/env python3
"""Cross-queue dispatcher: keep every registered RESUMABLE cell running on whichever queue / GPU type has room.

Why this exists: the account may run 2 H100 + 2 V100 on `fast` (4 h), 8 jobs on `default` (12 h) and 20 H100 + 20 V100 on
`slow_nice`, but single-queue submission left most of that idle (H100s are mostly taken by other users; V100s sit idle).
Registered cells are resumable (arch cells rebuild their history from disk; the harness A/B skips runs already logged), so a
cell runs as a chain of short jobs and may hop between queues / GPU types at every job boundary.

Safety: <= MAX_SUBS submissions per cycle, per-queue caps, exponential give-up (3 consecutive jobs that die in < 5 min pause
that cell), a kill switch file, and a full action log. It never touches jobs it did not submit unless the registry lists
their name under `ext` (then it only waits for them).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

R = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
SB = "/cm/shared/apps/slurm/current/bin"
REG = R / "scratchpad/dispatch_registry.json"
STOP = R / "scratchpad/dispatch_STOP"
OUT = R / "outputs/dispatch"
STATE_F, LOG_F = OUT / "state.json", OUT / "dispatch.log"
JOBS_DIR = R / "scratchpad/dispatch_jobs"
USER = "christen"
FAST_CAP = {"h100": 2, "v100": 2}          # QOS fast: gres/gpu:h100=2, gres/gpu:v100=2 per user
LLM_DOWN = False                           # set each cycle: LLM submissions paused (session limit); retrain cells may then use default (not fast: that would only preempt our own slow_nice jobs)
DEFAULT_SUBMIT_CAP = 15                    # QOS default: 16 submitted per user (dependency-held jobs count); keep 1 spare
STALL_S = {"h100": 3 * 3600, "v100": 6 * 3600}   # a RUNNING job with no new result file for this long is flagged (never auto-killed)
H100_RESERVE = 0                           # slow_nice H100 slots that V100-eligible cells may not take
SLOW_CAP = {"h100": 20, "v100": 20}        # QOS slow_nice: 20 + 20 GPUs per user (counts include pending)
WALL = {"fast": "04:00:00", "default": "12:00:00", "slow_nice": "2-00:00:00", "mig": "1-12:00:00"}
MIGQ_CAP = 14                              # QOS mig (partition migq): 14 slices per user, outside the slow_nice total of 40, 12 h walls, preempts slow_nice
_MIGQ = {"free": 0, "n": 0}                # free MIG slices and our mig-QOS jobs, refreshed at the start of every cycle
MIG_CAP = 28                               # MIG slices (H100 NVL 1g.12gb, 10.8 GB; measured 3 to 6 times slower than a full H100) on the otherwise idle migq partition; the partition accepts QOS default / fast / slow_nice only (the mig QOS is not allowed there); small retrain cells only
MIG_NO = ("ind30z_", "a600_", "a1m_", "mrr_", "pool", "fx35_c100", "fx35_c60", "fx35_c200", "sw38_100k")  # cells that must not run on a slice: the collaborator ensembles (urgent), large-D anchors, pool builds
# wall-clock budget (hours) handed to arch cells per job. A round (2 models) must finish before the wall limit; V100 rounds
# can take hours (per-model cap x5), so fast/V100 only starts a new round during the first 0.6 h.
BUDGET = {("fast", "h100"): 3.0, ("fast", "v100"): 0.6, ("default", "h100"): 10.5, ("default", "v100"): 6.0,
          ("slow_nice", "h100"): 20, ("slow_nice", "v100"): 12, ("mig", "mig"): 6.0}
MAX_SUBS, PD_MIGRATE_S, CYCLE_S = 24, 1200, 180
MAX_LLM_ACTIVE = 40           # overnight: laptop idle, V100 LLM cells draw few tokens per GPU-hour; LLM cells share ONE account session limit (hit at ~20:10 with 16 cells + laptop)
LLM_COOLDOWN_S = 1800         # after any cell exits 75 (rate limited) pause ALL LLM submissions this long


CLAUDE_LOGS = Path("/grid/wsbs/home_norepl/christen/.claude/projects")
LIMIT_RE = re.compile(r"hit your .{0,30}limit|usage limit|rate limit|limit reached", re.I)
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_reset(text: str) -> float | None:
    """'You've hit your session limit · resets 9pm (America/New_York)' -> epoch seconds. Also 'Oct 24, 9am' and 'Mon 9am'."""
    m = re.search(r"resets\s+([^()\n]+?)\s*\(([\w/_+-]+)\)", text)
    if not m:
        return None
    when, tzname = m.group(1).strip(), m.group(2)
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = ZoneInfo("America/New_York")
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", when, re.I)
    if not tm:
        return None
    now = dt.datetime.now(tz)
    cand = now.replace(hour=int(tm.group(1)) % 12 + (12 if tm.group(3).lower() == "pm" else 0), minute=int(tm.group(2) or 0), second=0, microsecond=0)
    md, wd = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})", when), re.search(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)", when)
    if md:
        try:
            cand = cand.replace(month=dt.datetime.strptime(md.group(1), "%b").month, day=int(md.group(2)))
        except Exception:
            pass
    elif wd:
        cand += dt.timedelta(days=(WD.index(wd.group(1)) - now.weekday()) % 7)
        if cand <= now:
            cand += dt.timedelta(days=7)
    elif cand <= now:
        # a reset time that passed minutes ago means the limit is lifting now (the CLI rounds it): wait 5 min, not 24 h
        cand = now + dt.timedelta(minutes=5) if (now - cand) < dt.timedelta(hours=2) else cand + dt.timedelta(days=1)
    return cand.timestamp()


def llm_probe() -> tuple[bool, str]:
    """One tiny CLI call through the same token path the cells use. True when the account accepts calls again."""
    try:
        # PATH: started from cron the dispatcher has no ~/.local/bin, so `claude` was "not found" (rc 127) and every probe looked like a usage limit (Nov 2: LLM cells paused for 1 h 15 min on a lifted limit)
        r = subprocess.run("bash -c 'export PATH=$HOME/.local/bin:$PATH; set +u; . ~/.config/albench/llm.env 2>/dev/null; set -u; echo Reply with the word OK. | claude -p --model claude-sonnet-5-5'",
                           shell=True, capture_output=True, text=True, timeout=150)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        if r.returncode != 0:
            log(f"LLM probe rc={r.returncode}: {out.strip()[:200]!r}")
        return (r.returncode == 0 and "OK" in (r.stdout or "") and not LIMIT_RE.search(r.stdout or "")), out
    except Exception as e:  # noqa: BLE001
        return False, f"probe error {type(e).__name__}: {e}"


def llm_tokens_last_hour() -> tuple[int, int, int]:
    """(calls, input+output+cache-write tokens, output tokens) over the last 60 min, from the CLI's own session logs."""
    now, calls, tot, out = time.time(), 0, 0, 0
    seen = set()
    for f in CLAUDE_LOGS.rglob("*.jsonl"):
        try:
            if f.stat().st_mtime < now - 3900:
                continue
            for line in open(f, errors="replace"):
                if '"usage"' not in line:
                    continue
                o = json.loads(line)
                m = o.get("message") or {}
                u = m.get("usage")
                if not u or o.get("type") != "assistant":
                    continue
                k = (m.get("id"), o.get("requestId"))
                if k in seen:
                    continue
                seen.add(k)
                if datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")).timestamp() < now - 3600:
                    continue
                calls += 1
                tot += (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                out += u.get("output_tokens") or 0
        except Exception:
            continue
    return calls, tot, out


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def log(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_F, "a") as fh:
        fh.write(line + "\n")


def my_jobs() -> list[dict]:
    jobs = []
    for line in sh(f'{SB}/squeue -u {USER} -h -o "%i|%j|%q|%t|%b|%V|%M"').splitlines():
        try:
            jid, name, qos, st, gres, sub, el = line.split("|")
        except ValueError:
            continue
        m = re.match(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", el.strip())
        el_s = (int(m.group(1) or 0) * 86400 + int(m.group(2) or 0) * 3600 + int(m.group(3)) * 60 + int(m.group(4))) if m else 0
        g = "mig" if "1g.12gb" in gres else ("h100" if "h100" in gres else ("v100" if "v100" in gres else "other"))  # MIG slices of an H100 NVL (partition migq, QOS mig) also contain "h100"
        try:
            age = time.time() - datetime.strptime(sub, "%Y-%m-%dT%H:%M:%S").timestamp()
        except Exception:
            age = 0.0
        jobs.append(dict(id=jid, name=name, qos=qos, st=st, gpu=g, age=age, el=el_s))
    return jobs


def progress(c: dict) -> tuple[int, float]:
    """(units of finished work, mtime of the newest result) for a cell: model metas, or logged A/B runs."""
    d = c["done"]
    if d["type"] == "hab_keys":
        p = R / d["file"]
        if not p.exists():
            return 0, 0.0
        have = set()
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            have.add((r["cfg"] if "cfg" in r else f"F{r.get('arch')}", r["seed"], r["arm"]))
        need = {(d["cfg"], s, a) for s in d["seeds"] for a in d["arms"]}
        return len(need & have), p.stat().st_mtime
    metas = [f for f in (R / c["out_dir"]).glob("r*_meta.json") if not f.name.endswith("_screen.json")]
    return len(metas), max((f.stat().st_mtime for f in metas), default=0.0)


def free_gpus() -> dict:
    free = {"h100": 0, "v100": 0}
    for line in sh(f"{SB}/scontrol show node -o").splitlines():
        if re.search(r"State=\S*(DOWN|DRAIN|FAIL|NOT_RESPONDING)", line):
            continue
        g = re.search(r"\bGres=gpu:(\w+):(\d+)", line)
        if not g or g.group(1) not in free:
            continue
        # this Slurm exposes no GresUsed; AllocTRES carries the untyped allocated GPU count (nodes are homogeneous)
        u = re.search(r"AllocTRES=\S*?gres/gpu=(\d+)", line)
        free[g.group(1)] += max(0, int(g.group(2)) - (int(u.group(1)) if u else 0))
    return free


def is_done(c: dict) -> bool:
    d = c["done"]
    if d["type"] == "file":
        return (R / d["path"]).exists()
    if d["type"] == "hab_keys":
        have, p = set(), R / d["file"]
        if p.exists():
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # an arm that errored is "attempted": do not loop forever on a deterministic failure
                have.add((r["cfg"] if "cfg" in r else f"F{r.get('arch')}", r["seed"], r["arm"]))
        return {(d["cfg"], s, a) for s in d["seeds"] for a in d["arms"]} <= have
    return False


def _mig_cap() -> int:
    """MIG slices to use; scratchpad/mig_cap.txt overrides the default without restarting the dispatcher (the migq partition has 28 slices)."""
    try:
        return int((R / "scratchpad/mig_cap.txt").read_text().strip())
    except Exception:
        return MIG_CAP


def _defer_prefixes() -> tuple:
    """Cell-name prefixes the dispatcher should not submit right now (one per line in scratchpad/defer_prefixes.txt; edit without a restart)."""
    try:
        return tuple(x.strip() for x in (R / "scratchpad/defer_prefixes.txt").read_text().splitlines() if x.strip())
    except Exception:
        return ()


MIG_BAD = {
    "V1.json": ["evoaug_paper2023_rep3_r04_arch_09"],
    "MM.json": ["motif_ct_enriched_30k_rep4_r10_arch_19"],
}  # menu entries that run out of memory on a MIG slice (menu file name -> entry ids)
_MENU_IDS: dict = {}


def _bad_shard(c: dict) -> bool:
    mm, ms = re.search(r"--menu (\S+)", c["cmd"]), re.search(r"--shard (\d+)/(\d+)", c["cmd"])
    if not (mm and ms) or mm.group(1).rsplit("/", 1)[-1] not in MIG_BAD:
        return False
    try:
        if mm.group(1) not in _MENU_IDS:
            _MENU_IDS[mm.group(1)] = [e.get("id") or e.get("name") for e in json.load(open(mm.group(1)))]
        ids = _MENU_IDS[mm.group(1)]
        k, n = int(ms.group(1)), int(ms.group(2))
        return any(i in ids and ids.index(i) % n == k for i in MIG_BAD[mm.group(1).rsplit("/", 1)[-1]])
    except Exception:
        return False


def mig_ok(c: dict) -> bool:
    """Menu-retrain cells up to 100k sequences run acceptably on a MIG slice (about 1/7 of an H100: the per-model time cap is scaled x8 in menu_retrain.py); LLM cells and big or urgent cells do not."""
    if "LLM_MODEL" in c.get("env", {}) or c["name"].startswith(MIG_NO) or "menu_retrain.py" not in c["cmd"]:
        return False
    m = re.search(r"--D (\d+)", c["cmd"])
    if not (bool(m) and int(m.group(1)) <= 100000):
        return False
    if _bad_shard(c):  # the shard that holds a menu entry known to need more than a 10.8 GB slice
        return False
    try:  # a model that ran out of the slice memory (10.8 GB) in an earlier attempt: rerun the cell on a full GPU
        fj = R / c["out_dir"] / "failed.jsonl"
        if fj.exists() and "out of memory" in fj.read_text().lower():
            return False
    except Exception:
        pass
    return True


def choose(c: dict, mine: list[dict], free: dict, exclude_id: str | None = None) -> tuple[str, str]:
    allowed = c["gpus"]
    _gpu_only = ("LLM_MODEL" not in c.get("env", {}) and not c["name"].startswith(("pool30_", "pool100_")) and not LLM_DOWN) or c["name"].startswith("a1m_")  # while the LLM is paused the idle fast slots take retrain cells; 1M anchors (3 h models) never use the 4 h fast wall  # keep the faster fast/default queues for the LLM runs (the long pole)
    _gpu_only_def = False  # everything may use the default queue now that no big retrain backlog competes for it (H100-first order below)
    # H100s are the scarce queue: once three of our H100 jobs are already waiting, V100-eligible cells go to V100s first (30k models train fine there)
    _pd_h100 = sum(1 for j in mine if j["gpu"] == "h100" and j["st"] == "PD" and j["id"] != exclude_id)
    order = ("v100", "h100") if ("v100" in allowed and _pd_h100 >= 3) else ("h100", "v100")
    js = [j for j in mine if j["id"] != exclude_id]
    cnt = lambda q, g: sum(1 for j in js if j["qos"] == q and j["gpu"] == g)  # noqa: E731
    for g in order:  # fast: priority 4000; it preempts default/slow_nice by design
        if (not _gpu_only or (g == "h100" and "LLM_MODEL" not in c.get("env", {}) and not c["name"].startswith("a1m_"))) and g in allowed and cnt("fast", g) < FAST_CAP[g]:
            return "fast", g
    n_def = sum(1 for j in js if j["qos"] == "default")
    n_def_run = sum(1 for j in js if j["qos"] == "default" and j["st"] == "R")
    if not _gpu_only_def and n_def < DEFAULT_SUBMIT_CAP and n_def_run < 8:  # default starts at once only while fewer than 8 of ours run (QOS per-user cap); otherwise slow_nice below
        for g in order:
            if g == "h100" and g in allowed and (free[g] > 0 or not any(j["qos"] == "default" and j["st"] == "PD" for j in js)):  # default is kept for H100 starts (priority over slow_nice); V100 work uses slow_nice and fast
                return "default", g
    if "v100" in allowed and n_def < DEFAULT_SUBMIT_CAP and cnt("default", "v100") < 4 and free["v100"] > 0 and _pd_h100 >= 3:  # H100 queues are stuffed: the 4 default V100 slots of our QOS take V100-eligible cells too
        return "default", "v100"
    if c["name"].startswith(("ind30q_", "ind30n1_", "ind30h_")) and "v100" in allowed and n_def < DEFAULT_SUBMIT_CAP and n_def_run < 8 and free["v100"] > 0:
        return "default", "v100"  # quick-turnaround cells use spare default slots on V100 while H100 starts are blocked (LLM cells hold the slow_nice V100 slots)
    if mig_ok(c) and sum(1 for j in js if j["gpu"] == "mig") < _mig_cap():  # an idle lane nobody contests: before queueing behind other users on slow_nice
        return "mig", "mig"
    _cap = lambda g: SLOW_CAP[g] - (H100_RESERVE if (g == "h100" and len(allowed) > 1) else 0)  # noqa: E731  keep slow_nice H100 slots for H100-only cells (large-D anchors, 300k searches)
    for g in order:
        if g in allowed and free[g] > 0 and cnt("slow_nice", g) < _cap(g):
            return "slow_nice", g
    for g in order:
        if g in allowed and cnt("slow_nice", g) < _cap(g):
            return "slow_nice", g
    if not _gpu_only_def and n_def < DEFAULT_SUBMIT_CAP:  # slow_nice full too: queue behind the default cap rather than overfilling slow_nice
        return "default", allowed[0]
    return "wait", ""


RETRAIN_PFX = ("mix_", "un30_", "un10_", "un100_", "as_", "ind30", "nest_", "mrf", "mr2_", "mrr_", "fx35_", "fx36_", "fe35_", "fr35_", "lg35", "pk35_", "sw38_", "bench38_", "mt38_", "mv1c_")
LLM_SMALL = ("g30_", "rnd30_", "u_", "mot30_", "g10_", "u10_", "mx30_", "ex30_", "rf30_", "xr30_", "ag30_", "ac30_", "an30_", "ae30_", "xe30_", "xd30_", "xe10_", "xd10_")


def wall_and_budget(c: dict, qos: str, gpu: str) -> tuple[str, float]:
    """slow_nice jobs get a short wall time so the scheduler can backfill them (a 2-day request never fits a gap); cells are resumable."""
    n = c["name"]
    if qos == "slow_nice" and n.startswith("sw38_100k"):  # a 100k model can take 5 h on a V100
        return "10:00:00", BUDGET[(qos, gpu)]
    if qos == "slow_nice" and n.startswith(RETRAIN_PFX):
        return "03:30:00", BUDGET[(qos, gpu)]
    if qos == "slow_nice" and n.startswith(LLM_SMALL):
        return "08:00:00", 7.0
    if qos == "slow_nice" and n.startswith(("g100x_", "u100_", "g300s_", "a600_", "a1m_")):
        return "14:00:00", 12.5
    return WALL[qos], BUDGET[(qos, gpu)]


def sbatch_text(c: dict, qos: str, gpu: str, migq: bool = False) -> str:
    _wall, _bud = wall_and_budget(c, qos, gpu)
    cmd = c["cmd"].replace("{BUDGET}", str(_bud))
    env = " ".join(f"{k}={v}" for k, v in c.get("env", {}).items())
    if qos == "mig":
        _hdr = f"#SBATCH --partition=migq --qos={'mig' if migq else 'slow_nice'} --gres=gpu:nvidia_h100_nvl_1g.12gb:1\n#SBATCH --cpus-per-task=4 --mem=40G --time={'12:00:00' if migq else _wall} --requeue"
    else:
        # 4 CPUs and 48 GB fit the many nodes that have a free GPU but only 4 free CPUs; AlphaGenome jobs keep 8 and 90 GB
        _cm = "--cpus-per-task=8 --mem=90G" if "bench33_ag.py" in c["cmd"] else "--cpus-per-task=4 --mem=48G"
        _hdr = f"#SBATCH --partition=gpuq --qos={qos} --gres=gpu:{gpu}:1\n#SBATCH {_cm} --time={_wall} --requeue"
    return f"""#!/bin/bash
#SBATCH --job-name=d_{c['name']}
{_hdr}
#SBATCH --output={OUT}/logs/d_{c['name']}_%A.log
cd {R}
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
[ -r ~/.config/albench/llm.env ] && . ~/.config/albench/llm.env
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1
{('export ' + env) if env else ''}
{('export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MENU_MAX_FWD_MS=1000000000') if qos == 'mig' else ''}
mkdir -p {R / c['out_dir']}
{cmd}
"""


def submit(c: dict, qos: str, gpu: str, dry: bool) -> str:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    f = JOBS_DIR / f"d_{c['name']}_{qos}_{gpu}.sbatch"
    migq = qos == "mig" and _MIGQ["free"] > 0 and _MIGQ["n"] < MIGQ_CAP  # a free slice is required: a mig-QOS job would otherwise preempt our own slow_nice slices
    f.write_text(sbatch_text(c, qos, gpu, migq))
    if dry:
        return "DRY"
    out = sh(f"{SB}/sbatch --parsable {f}")
    if migq and re.match(r"^\d+", out):
        _MIGQ["free"] -= 1
        _MIGQ["n"] += 1
    return out.split(";")[0] if re.match(r"^\d+", out) else ""


def check_finished(c: dict, st: dict, mine_ids: set) -> None:
    """Did the last dispatcher job end quickly without finishing? 3 in a row pauses the cell."""
    jid = st.get("last_job")
    if not jid or jid in mine_ids or st.get("checked") == jid:
        return
    st["checked"] = jid
    if st.get("expected_cancel") == jid:
        return
    row = sh(f"{SB}/sacct -n -P -X -j {jid} -o State,Elapsed,ExitCode").splitlines()
    if not row:
        return
    state, el, ec = (row[0].split("|") + ["", "", ""])[:3]
    if ec.startswith("75:"):  # rate limited (usage/session limit): not a failure; back off all LLM cells
        st["rate_limited"] = True
        try:
            tail = "\n".join(open(OUT / "logs" / f"d_{c['name']}_{jid}.log", errors="replace").read().splitlines()[-80:])
        except Exception:
            tail = ""
        st["reset_epoch"] = parse_reset(tail)
        when = datetime.fromtimestamp(st["reset_epoch"]).strftime("%m-%d %H:%M") if st["reset_epoch"] else "unknown"
        log(f"  {c['name']}: job {jid} exited 75 (LLM limit) after {el}; reset time parsed from CLI: {when}")
        return
    h, m, s = (el.split(":") + ["0", "0"])[-3:] if el.count(":") == 2 else ("0", "0", "0")
    secs = int(h) * 3600 + int(m) * 60 + int(s)
    if secs < 300 and not state.startswith("COMPLETED"):
        st["short_fails"] = st.get("short_fails", 0) + 1
        log(f"  {c['name']}: job {jid} ended in {secs}s ({state}); consecutive short failures = {st['short_fails']}")
        if st["short_fails"] >= 3:
            st["paused"] = True
            log(f"  {c['name']}: PAUSED after 3 consecutive short failures - inspect {OUT}/logs/d_{c['name']}_*.log")
    else:
        st["short_fails"] = 0


def cycle(reg: list[dict], dry: bool) -> None:
    mine, free = my_jobs(), free_gpus()
    _MIGQ["free"], _MIGQ["n"] = free.get("mig", 0), sum(1 for j in mine if j["qos"] == "mig" and j["gpu"] == "mig")
    state = json.loads(STATE_F.read_text()) if STATE_F.exists() else {}
    subs, ids = 0, {j["id"] for j in mine}
    g = state.setdefault('_global', {})
    if g.get('probe_needed') and time.time() >= g.get('llm_cooldown_until', 0) and not dry:
        ok, out = llm_probe()
        if ok:
            g['probe_needed'] = False
            log('LLM probe OK: limit has reset; LLM cells may be resubmitted')
        else:
            nxt = parse_reset(out) or time.time() + 1200
            g['llm_cooldown_until'] = max(nxt, time.time() + 240) + 30  # retry soon: the CLI reset time is only approximate
            log(f"LLM probe still limited; waiting until {datetime.fromtimestamp(g['llm_cooldown_until']):%m-%d %H:%M}")
    global LLM_DOWN
    _gl = state.get("_global", {})
    LLM_DOWN = bool(time.time() < _gl.get("llm_cooldown_until", 0) or _gl.get("probe_needed"))
    summary = {"done": 0, "active": 0, "submitted": 0, "paused": 0}
    for c in reg:
        st = state.setdefault(c["name"], {})
        if is_done(c):
            summary["done"] += 1
            continue
        if c["name"].startswith(_defer_prefixes()):
            continue
        if st.get("paused"):
            summary["paused"] += 1
            continue
        st.pop('rate_limited', None)
        check_finished(c, st, ids)
        if st.pop('rate_limited', False):
            g = state.setdefault('_global', {})
            until = (st.pop('reset_epoch', None) or time.time() + LLM_COOLDOWN_S) + 120  # wait for the reset, +2 min
            g['llm_cooldown_until'] = max(g.get('llm_cooldown_until', 0), until)
            g['probe_needed'] = True
            log(f"  LLM submissions paused until {datetime.fromtimestamp(g['llm_cooldown_until']):%m-%d %H:%M} (then a probe call must succeed)")
        is_llm = 'LLM_MODEL' in c.get('env', {})
        jn = "d_" + c["name"]
        active = [j for j in mine if j["name"] in [jn] + c.get("ext", [])]
        if active:
            summary["active"] += 1
            j = active[0]
            if j["st"] == "R" and j["gpu"] in STALL_S:  # flag (never kill) a running job that stopped producing results
                n_done, newest = progress(c)
                idle = time.time() - max(newest, time.time() - j["el"])
                if idle > STALL_S[j["gpu"]] and time.time() - st.get("stall_logged", 0) > 3600:
                    st["stall_logged"] = time.time()
                    log(f"  STALL? {c['name']}: running {j['el'] // 60} min on {j['qos']}/{j['gpu']} but no new result for {int(idle // 60)} min "
                        f"({n_done} done)")
            if j["name"] == jn and j["st"] == "PD" and j["qos"] == "default" and j["age"] > PD_MIGRATE_S:  # default is group-limited (waits for hours with "Priority"); a free fast slot is better
                q, g = choose(c, mine, free, exclude_id=j["id"])
                if q == "default" and j["age"] > 1200 and "h100" in c["gpus"] and free["h100"] > 0 and sum(1 for x in mine if x["qos"] == "slow_nice" and x["gpu"] == "h100") < SLOW_CAP["h100"]:
                    q, g = "slow_nice", "h100"  # default has starved this job for over an hour while H100s are free: stay on slow_nice for 6 h (the slow_nice branch below must not move it back)
                    st["stay_slow_until"] = time.time() + 6 * 3600
                if q in ("fast", "slow_nice"):
                    log(f"  {c['name']}: pending {int(j['age'] / 60)} min on default/{j['gpu']}; migrating to {q}/{g}")
                    if not dry:
                        st["expected_cancel"] = j["id"]
                        sh(f"{SB}/scancel {j['id']}")
                    mine = [x for x in mine if x["id"] != j["id"]]
                    jid = submit(c, q, g, dry)
                    st["last_job"] = jid
                    mine.append(dict(id=jid, name=jn, qos=q, st="PD", gpu=g, age=0))
            if j["name"] == jn and j["st"] == "PD" and j["qos"] == "slow_nice" and j["age"] > PD_MIGRATE_S:
                _pinned = time.time() < st.get("stay_slow_until", 0)
                _mine_c = mine + [dict(id=f"_pin{i}", name="_pin", qos="default", st="PD", gpu="h100", age=0) for i in range(DEFAULT_SUBMIT_CAP)] if _pinned else mine  # pinned: pretend the default queue is full so choose() offers fast / MIG / slow_nice only
                q, g = choose(c, _mine_c, free, exclude_id=j["id"])
                if q != "wait" and (q != "slow_nice" or (g != j["gpu"] and free[g] > 0)) and not (q == "default" and time.time() < st.get("stay_slow_until", 0)):  # pinned cells may still hop between H100 and V100 on slow_nice, but never back to default
                    log(f"  {c['name']}: pending {int(j['age'] / 60)} min on slow_nice/{j['gpu']}; migrating to {q}/{g}")
                    if not dry:
                        st["expected_cancel"] = j["id"]
                        sh(f"{SB}/scancel {j['id']}")
                    mine = [x for x in mine if x["id"] != j["id"]]
                    jid = submit(c, q, g, dry)
                    st["last_job"] = jid
                    mine.append(dict(id=jid, name=jn, qos=q, st="PD", gpu=g, age=0))
            continue
        if subs >= MAX_SUBS:
            continue
        if any(not (R / p_).exists() for p_ in c.get("needs", [])):  # dependency gate: wait until the pool/run files this cell reads exist
            continue
        if is_llm:
            if time.time() < state.get('_global', {}).get('llm_cooldown_until', 0) or state.get('_global', {}).get('probe_needed'):
                continue
            llm_active = sum(1 for cc in reg if 'LLM_MODEL' in cc.get('env', {}) for j in mine if j['name'] in ['d_' + cc['name']] + cc.get('ext', []))
            if llm_active >= MAX_LLM_ACTIVE:
                continue
        q, g = choose(c, mine, free)
        if q == "wait":
            continue
        jid = submit(c, q, g, dry)
        if not jid:
            log(f"  {c['name']}: sbatch FAILED for {q}/{g}")
            continue
        st["last_job"], st["last_target"] = jid, f"{q}/{g}"
        log(f"  {c['name']}: submitted job {jid} on {q}/{g} (free h100={free['h100']} v100={free['v100']})")
        mine.append(dict(id=jid, name=jn, qos=q, st="PD", gpu=g, age=0))
        free[g] = max(0, free.get(g, 0) - 1)
        subs += 1
        summary["submitted"] += 1
        summary["active"] += 1
    if not dry:
        STATE_F.write_text(json.dumps(state, indent=1))
        calls_h, tok_h, out_h = llm_tokens_last_hour()
        gl = state.get("_global", {})
        paused = ("PAUSED until " + datetime.fromtimestamp(gl["llm_cooldown_until"]).strftime("%m-%d %H:%M")) if time.time() < gl.get("llm_cooldown_until", 0) or gl.get("probe_needed") else "ok"
        lines = [f"dispatcher status {datetime.now():%Y-%m-%d %H:%M:%S}  free GPUs {free}  summary {summary}",
                 f"LLM last 60 min (HPC CLI logs): {calls_h} calls, {tok_h:,} tokens (in+out+cache-write), {out_h:,} output | LLM submissions: {paused}"]
        for c in reg:
            n_done, newest = progress(c)
            jn = "d_" + c["name"]
            j = next((x for x in mine if x["name"] in [jn] + c.get("ext", [])), None)
            where = f"{j['qos']}/{j['gpu']}/{j['st']}" if j else "-"
            ago = f"{int((time.time() - newest) / 60)}m ago" if newest else "never"
            flag = "DONE" if is_done(c) else ("PAUSED" if state.get(c["name"], {}).get("paused") else "")
            lines.append(f"  {c['name']:<22} {flag:<7} {where:<22} results={n_done:<4} newest={ago}")
        (OUT / "status.txt").write_text("\n".join(lines) + "\n")
    mq = {}
    for j in mine:
        mq[(j["qos"], j["gpu"], j["st"])] = mq.get((j["qos"], j["gpu"], j["st"]), 0) + 1
    log(f"cycle: {summary} | my jobs {dict(sorted(mq.items()))} | free GPUs {free}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    while True:
        if STOP.exists():
            log("STOP file present; exiting")
            return
        try:
            cycle(json.loads(REG.read_text()), args.dry)
        except Exception as e:  # never die silently
            log(f"cycle error: {type(e).__name__}: {e}")
        if args.once:
            return
        time.sleep(CYCLE_S)


if __name__ == "__main__":
    main()
