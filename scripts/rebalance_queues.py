"""Keep the HP search optimally spread across the cluster's GPU pools.

Two non-destructive passes (scontrol update on PENDING jobs only; reversible):
  1) Promote smallest-D pending hp_ cells from slow_nice onto the high-priority
     fast/default H100 tiers so they grab free H100s sooner. Max walltime per
     tier is always used: fast=4h, default=12h.
  2) Flip pending hp_ cells from H100 onto the otherwise-IDLE V100 pool
     (slow_nice v100 sub-cap = 20). The LegNet HP search trains fp32 (no bf16),
     so it runs fine on Volta; this opens a second ~20-GPU pool that would
     otherwise sit empty while jobs queue on contended H100s.

Large-D note: D=1M cells are now eligible for ALL tiers (incl. fast 4h). A 50+
round D=1M search never completes in a single window (not even slow_nice 48h),
and the search checkpoints per round (r*_meta.json) with the watchdog resubmitting
to resume. So a short window is productive (it advances a few rounds and saves
them), never wasted — there is no walltime risk to be mindful of when the work is
checkpointed and resumable. Better to advance on an idle GPU now than queue."""

import re
import subprocess

BIN = "/cm/shared/apps/slurm/current/bin"
FAST_CAP, DEF_CAP = 2, 4  # real per-user GPU caps; spread the tail, don't overstuff one pool
FAST_MAX_D, DEF_MAX_D = 1_000_000, 1_000_000
V100_TARGET = 20  # fill the idle V100 pool up to the slow_nice v100 sub-cap
V100_MAX_D = 1_000_000  # checkpoint+resume makes V100's slower speed safe even at D=1M


def squeue(fmt):
    return subprocess.run(
        [f"{BIN}/squeue", "--me", "-h", "-o", fmt], capture_output=True, text=True, timeout=20
    ).stdout


def scontrol_qos(jid, qos, tl):
    r = subprocess.run(
        [f"{BIN}/scontrol", "update", f"jobid={jid}", f"TimeLimit={tl}", f"qos={qos}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def scontrol_v100(jid):
    r = subprocess.run(
        [f"{BIN}/scontrol", "update", f"jobid={jid}", "TresPerNode=gres/gpu:v100:1"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def main():
    fast = default = v100 = 0
    pend = []  # (jid, D, name, qos, gres)
    for ln in squeue("%i|%j|%T|%q|%b").strip().split("\n"):
        if not ln.strip():
            continue
        parts = ln.split("|")
        jid, name, state, qos, gres = parts[0], parts[1], parts[2], parts[3], parts[4]
        if qos == "fast":
            fast += 1
        elif qos == "default":
            default += 1
        if "v100" in gres:
            v100 += 1
        if name.startswith("hp_") and not name.startswith("hp_watchdog") and state == "PENDING":
            m = re.search(r"_d(\d+)_", name)
            if m:
                pend.append((jid, int(m.group(1)), name, qos, gres))
    pend.sort(key=lambda x: x[1])  # smallest D first
    print(
        f"start: fast={fast}/{FAST_CAP} default={default}/{DEF_CAP} "
        f"v100={v100}/{V100_TARGET}  pending hp_={len(pend)}"
    )

    # Pass 1 — fill high-priority H100 tiers. Skip v100 jobs (leave them on the
    # idle pool with their long slow_nice walltime).
    mf = md = 0
    promoted = set()
    for jid, D, name, qos, gres in pend:
        if qos in ("fast", "default") or "v100" in gres:
            continue
        if fast < FAST_CAP and D <= FAST_MAX_D:
            ok, err = scontrol_qos(jid, "fast", "04:00:00")
            if ok:
                fast += 1
                mf += 1
                promoted.add(jid)
                print(f"  fast    <- {name} ({jid})")
            else:
                print(f"  ERR fast {name}: {err[:120]}")
        elif default < DEF_CAP and D <= DEF_MAX_D:
            ok, err = scontrol_qos(jid, "default", "12:00:00")
            if ok:
                default += 1
                md += 1
                promoted.add(jid)
                print(f"  default <- {name} ({jid})")
            else:
                print(f"  ERR default {name}: {err[:120]}")
        if fast >= FAST_CAP and default >= DEF_CAP:
            break

    # Pass 2 — fill the idle V100 pool from remaining slow_nice H100 pending.
    mv = 0
    for jid, D, name, qos, gres in pend:
        if v100 >= V100_TARGET:
            break
        if jid in promoted or qos != "slow_nice" or "v100" in gres or D > V100_MAX_D:
            continue
        ok, err = scontrol_v100(jid)
        if ok:
            v100 += 1
            mv += 1
            print(f"  v100    <- {name} ({jid})")
        else:
            print(f"  ERR v100 {name}: {err[:120]}")

    print(
        f"\nmoved: fast+={mf} default+={md} v100+={mv}  "
        f"now fast={fast} default={default} v100={v100}"
    )


if __name__ == "__main__":
    main()
