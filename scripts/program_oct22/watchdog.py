"""HPC-side watchdog for the dispatcher program (cron, every 30 minutes). Runs on the login node, so it is light: file reads, squeue, the status script.

Why it exists: the monitoring I do from a laptop session needs the VPN. When the VPN drops nothing watches the queue and the routine failures (a model
that ran out of the MIG slice memory, a burst of CPU-gate failures that wrote a COMPLETE marker with nothing trained) wait for the next reconnect.
This script handles those, keeps a status snapshot where I can read it on reconnect, and can email when something needs a human.

  outputs/watch/status_latest.txt   full status (overwritten every run)
  outputs/watch/history.log         one line per run
  outputs/watch/events.log          every automatic action and every alert
  outputs/watch/EMAIL_ON            create this file to get mail (to the local account, `mail <user>`): group finished, dispatcher down, nothing running

Automatic actions (each at most twice per marker, so a model that fails for a real reason cannot loop):
  * a shard marker COMPLETE_kofn with failed>=1 in a directory whose failed.jsonl records an out-of-memory error, and no job of that cell active:
    delete the marker; the dispatcher keeps cells with such a record off the 10.8 GB MIG slices (mig_ok) and reruns the missing model on a full GPU;
  * a marker with failed>=5 and failed > done (CPU forward-time gate storm on a loaded node): delete the marker, the cell reruns the missing models.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

R = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
W = R / "outputs/watch"
SQ = "/cm/shared/apps/slurm/current/bin/squeue"
MR = R / "outputs/menu_retrain_v0"
now = time.strftime("%Y-%m-%d %H:%M")


def log(path: str, msg: str) -> None:
    with open(W / path, "a") as fh:
        fh.write(f"{now} {msg}\n")


def sh(cmd: list[str], timeout: int = 300) -> str:
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=R
        ).stdout
    except Exception as ex:  # noqa: BLE001
        return f"ERR {ex}"


def mail(subject: str, body: str) -> None:
    if (W / "EMAIL_ON").exists():
        user = os.environ.get("USER", "christen")
        try:
            subprocess.run(["mail", "-s", subject, user], input=body, text=True, timeout=60)
        except Exception as ex:  # noqa: BLE001
            log("events.log", f"mail failed: {ex}")


EXPECTED = {
    "10k": 20,
    "30k": 20,
    "100k": 12,
}  # configurations trained per cell of the Oct-28 night program


def health() -> tuple[list[str], list[str]]:
    """One line per scaling / top-up cell: models trained against expected, cap hits, failure reasons, mean validation r; plus flags."""
    lines, flags = [], []
    for d in sorted(MR.glob("sc31_*")) + sorted(MR.glob("al31_*")):
        m = re.match(r"sc31_(\d+k)_", d.name)
        exp = EXPECTED[m.group(1)] if m else 12
        metas = []
        for f in d.glob("r00_menu_*_meta.json"):
            try:
                metas.append(json.load(open(f)))
            except Exception:  # noqa: BLE001 - file still being written
                pass
        fails = collections.Counter()
        if (d / "failed.jsonl").exists():
            for ln in open(d / "failed.jsonl"):
                try:
                    fails[json.loads(ln)["why"][:28]] += 1
                except Exception:  # noqa: BLE001
                    pass
        caps = sum(1 for x in metas if x.get("cap_hit"))
        val = sum(x["val_pearson"] for x in metas) / max(len(metas), 1)
        ep = min((x.get("epochs_trained") or 0 for x in metas), default=0)
        markers = len(list(d.glob("COMPLETE_*")))
        lines.append(
            f"{d.name:44s} models {len(metas):2d}/{exp} markers {markers} cap_hit {caps} mean_val {val:.3f} min_epochs {ep} failed {dict(fails) or '-'}"
        )
        n_shards = (
            int(
                re.search(r"of(\d+)", next(iter(d.glob("COMPLETE_*")), Path("x_of0")).name).group(1)
            )
            if markers
            else 0
        )
        if markers and markers == n_shards and len(metas) < exp:
            flags.append(
                f"{d.name}: all shards complete but {len(metas)}/{exp} models ({dict(fails)})"
            )
        if metas and len(metas) >= 3 and val < 0.55:
            flags.append(f"{d.name}: mean validation r {val:.3f} is low")
        if caps > 0.25 * max(len(metas), 1) and len(metas) >= 4:
            flags.append(f"{d.name}: {caps}/{len(metas)} models hit the time cap")
    return lines, flags


def main() -> None:
    W.mkdir(parents=True, exist_ok=True)
    sf = W / "state.json"
    state = json.load(open(sf)) if sf.exists() else {"notified": [], "retries": {}, "idle": 0}
    status = sh(["nice", str(R / ".venv/bin/python"), "scripts/program_oct22/status_oct26.py"])
    (W / "status_latest.txt").write_text(f"{now}\n{status}")
    jobs = sh([SQ, "-u", os.environ.get("USER", "christen"), "-h", "-o", "%j %T"]).splitlines()
    running = [j for j in jobs if j.endswith(" RUNNING")]
    pending = [j for j in jobs if j.endswith(" PENDING")]
    tmux = (
        subprocess.run(["tmux", "has-session", "-t", "dispatch"], capture_output=True).returncode
        == 0
    )
    log(
        "history.log",
        f"dispatcher={'up' if tmux else 'DOWN'} running={len(running)} pending={len(pending)}",
    )
    alerts = []
    if not tmux:
        alerts.append("the dispatcher is not running (cron relaunches it within 10 minutes)")
    state["idle"] = state["idle"] + 1 if (not running and pending) else 0
    if state["idle"] >= 2:
        alerts.append(f"nothing is running while {len(pending)} jobs wait (two checks in a row)")
    # groups that finished since the last run
    for line in status.splitlines():
        m = re.match(r"^(.*?)\s+(\d+)/(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if m and m.group(2) == m.group(3) and m.group(1).strip() not in state["notified"]:
            state["notified"].append(m.group(1).strip())
            if state.get("initialised"):
                alerts.append(f"group finished: {m.group(1).strip()} ({m.group(3)} cells)")
    state["initialised"] = True
    # automatic fixes
    reg = json.load(open(R / "scratchpad/dispatch_registry.json"))
    by_marker = {c["done"]["path"]: c for c in reg if c.get("done", {}).get("type") == "file"}
    active = {j.split()[0] for j in jobs}
    for fj in glob.glob(str(MR / "*/failed.jsonl")):
        d = Path(fj).parent
        txt = Path(fj).read_text().lower()
        has_oom = "out of memory" in txt
        for mk in d.glob("COMPLETE_*of*"):
            m = re.search(r"done=(\d+) failed=(\d+)", mk.read_text())
            if not m:
                continue
            done, failed = int(m.group(1)), int(m.group(2))
            rel = str(mk.relative_to(R))
            cell = by_marker.get(rel)
            if cell is None or f"d_{cell['name']}" in active:
                continue
            why = None
            if has_oom and failed >= 1:
                why = "out-of-memory model"
            elif failed >= 5 and failed > done:
                why = "forward-time gate storm"
            if why and state["retries"].get(rel, 0) < 2:
                state["retries"][rel] = state["retries"].get(rel, 0) + 1
                mk.unlink()
                log(
                    "events.log",
                    f"deleted {rel} ({why}; done={done} failed={failed}) -> cell reruns the missing models",
                )
    hl, hf = health()
    (W / "health_oct30.txt").write_text(
        f"{now}\n" + "\n".join(hl + [""] + (["FLAGS"] + hf if hf else ["no flags"])) + "\n"
    )
    new = [x for x in hf if x not in state.get("flags", [])]
    if new:
        alerts += new
    state["flags"] = hf
    json.dump(state, open(sf, "w"))
    if alerts:
        log("events.log", "ALERT " + "; ".join(alerts))
        mail("ALBench HPC watchdog", "\n".join(alerts) + "\n\n" + status[-3000:])


if __name__ == "__main__":
    main()
