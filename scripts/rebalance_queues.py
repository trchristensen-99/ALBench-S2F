"""Keep the HP search optimally spread across the cluster's GPU pools.

Three non-destructive passes (scontrol update on PENDING jobs only; reversible).
Faster hardware always gets first dibs — every free H100 slot is filled before
any V100 slot, because H100 trains the LegNet HP search markedly faster:
  1) Fill the high-priority H100 sub-caps: fast (2 H100, 4h) then default
     (4 H100, 12h). Max walltime per tier is always used.
  2) Fill the high-priority V100 sub-caps: fast (2 V100, 4h) then default
     (4 V100, 12h). Same qos = same scheduling priority as the H100 tiers, so
     these jobs jump the queue ahead of slow_nice — they just run on Volta.
  3) Overflow remaining cells onto the slow_nice V100 pool (sub-cap 20, 48h,
     priority 100). The LegNet HP search trains fp32 (no bf16) so it runs fine
     on Volta; this opens a second ~20-GPU pool that would otherwise sit empty.

Per-user MaxTRESPU (sacctmgr): fast = 2 H100 + 2 V100, default = 4 H100 + 4 V100,
slow_nice = 20 H100 + 20 V100. We saturate the H100 sub-caps first, then the
V100 sub-caps, so no fast GPU ever idles while a slow one runs.

Large-D note: D=1M cells are eligible for ALL tiers (incl. fast 4h). A 50+ round
D=1M search never completes in a single window (not even slow_nice 48h), and the
search checkpoints per round (r*_meta.json) with the watchdog resubmitting to
resume. So a short window is productive (it advances a few rounds and saves
them), never wasted — there is no walltime risk when the work is checkpointed
and resumable. Better to advance on an idle GPU now than queue."""

import re
import subprocess

BIN = "/cm/shared/apps/slurm/current/bin"
FAST_H100_CAP, DEF_H100_CAP = 2, 4  # per-user MaxTRESPU H100 sub-caps (fast / default)
FAST_V100_CAP, DEF_V100_CAP = 2, 4  # per-user MaxTRESPU V100 sub-caps (fast / default)
SN_V100_CAP = 20  # slow_nice V100 sub-cap (overflow pool)
MAX_D = 1_000_000  # all tiers eligible up to D=1M (checkpoint+resume makes short windows safe)


def squeue(fmt):
    return subprocess.run(
        [f"{BIN}/squeue", "--me", "-h", "-o", fmt], capture_output=True, text=True, timeout=20
    ).stdout


def scontrol_qos(jid, qos, tl):
    """Promote to a higher-priority qos on the job's current (H100) gres."""
    r = subprocess.run(
        [f"{BIN}/scontrol", "update", f"jobid={jid}", f"TimeLimit={tl}", f"qos={qos}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def scontrol_v100_qos(jid, qos, tl):
    """Move to a higher-priority qos AND flip the gres to V100, in one update."""
    r = subprocess.run(
        [
            f"{BIN}/scontrol",
            "update",
            f"jobid={jid}",
            f"TimeLimit={tl}",
            f"qos={qos}",
            "TresPerNode=gres/gpu:v100:1",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def scontrol_v100(jid):
    """Flip gres to V100, leaving the job on its current (slow_nice) qos."""
    r = subprocess.run(
        [f"{BIN}/scontrol", "update", f"jobid={jid}", "TresPerNode=gres/gpu:v100:1"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def main():
    fast_h = fast_v = def_h = def_v = sn_v = 0
    pend = []  # (jid, D, name, qos, gres)
    for ln in squeue("%i|%j|%T|%q|%b").strip().split("\n"):
        if not ln.strip():
            continue
        parts = ln.split("|")
        jid, name, state, qos, gres = parts[0], parts[1], parts[2], parts[3], parts[4]
        is_v = "v100" in gres
        if qos == "fast":
            fast_v, fast_h = (fast_v + 1, fast_h) if is_v else (fast_v, fast_h + 1)
        elif qos == "default":
            def_v, def_h = (def_v + 1, def_h) if is_v else (def_v, def_h + 1)
        elif qos == "slow_nice" and is_v:
            sn_v += 1
        if name.startswith("hp_") and not name.startswith("hp_watchdog") and state == "PENDING":
            m = re.search(r"_d(\d+)_", name)
            if m:
                pend.append((jid, int(m.group(1)), name, qos, gres))
    pend.sort(key=lambda x: x[1])  # smallest D first
    print(
        f"start: fastH={fast_h}/{FAST_H100_CAP} defH={def_h}/{DEF_H100_CAP} "
        f"fastV={fast_v}/{FAST_V100_CAP} defV={def_v}/{DEF_V100_CAP} "
        f"snV={sn_v}/{SN_V100_CAP}  pending hp_={len(pend)}"
    )

    promoted = set()
    mfh = mdh = mfv = mdv = msv = 0

    # Pass 1 — saturate the high-priority H100 sub-caps first (fastest hardware wins).
    for jid, D, name, qos, gres in pend:
        if qos in ("fast", "default") or "v100" in gres or D > MAX_D:
            continue
        if fast_h < FAST_H100_CAP:
            ok, err = scontrol_qos(jid, "fast", "04:00:00")
            if ok:
                fast_h += 1
                mfh += 1
                promoted.add(jid)
                print(f"  fastH   <- {name} ({jid})")
            else:
                print(f"  ERR fastH {name}: {err[:120]}")
        elif def_h < DEF_H100_CAP:
            ok, err = scontrol_qos(jid, "default", "12:00:00")
            if ok:
                def_h += 1
                mdh += 1
                promoted.add(jid)
                print(f"  defH    <- {name} ({jid})")
            else:
                print(f"  ERR defH {name}: {err[:120]}")
        if fast_h >= FAST_H100_CAP and def_h >= DEF_H100_CAP:
            break

    # Pass 2 — only once H100 is full, fill the high-priority V100 sub-caps.
    # Picks up both slow_nice-H100 cells (flip gres -> V100) and cells already
    # parked on slow_nice-V100 (just bump qos up to fast/default).
    for jid, D, name, qos, gres in pend:
        if jid in promoted or qos != "slow_nice" or D > MAX_D:
            continue
        if fast_v < FAST_V100_CAP:
            ok, err = scontrol_v100_qos(jid, "fast", "04:00:00")
            if ok:
                fast_v += 1
                mfv += 1
                promoted.add(jid)
                print(f"  fastV   <- {name} ({jid})")
            else:
                print(f"  ERR fastV {name}: {err[:120]}")
        elif def_v < DEF_V100_CAP:
            ok, err = scontrol_v100_qos(jid, "default", "12:00:00")
            if ok:
                def_v += 1
                mdv += 1
                promoted.add(jid)
                print(f"  defV    <- {name} ({jid})")
            else:
                print(f"  ERR defV {name}: {err[:120]}")
        if fast_v >= FAST_V100_CAP and def_v >= DEF_V100_CAP:
            break

    # Pass 3 — overflow the rest onto the idle slow_nice V100 pool (48h, priority 100).
    for jid, D, name, qos, gres in pend:
        if sn_v >= SN_V100_CAP:
            break
        if jid in promoted or qos != "slow_nice" or "v100" in gres or D > MAX_D:
            continue
        ok, err = scontrol_v100(jid)
        if ok:
            sn_v += 1
            msv += 1
            promoted.add(jid)
            print(f"  snV     <- {name} ({jid})")
        else:
            print(f"  ERR snV {name}: {err[:120]}")

    print(
        f"\nmoved: fastH+={mfh} defH+={mdh} fastV+={mfv} defV+={mdv} snV+={msv}  "
        f"now fastH={fast_h} defH={def_h} fastV={fast_v} defV={def_v} snV={sn_v}"
    )


if __name__ == "__main__":
    main()
