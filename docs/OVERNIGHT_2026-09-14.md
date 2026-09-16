> **HISTORICAL SNAPSHOT (2026-09-14).** The `run_*.sbatch` files named below
> were retired on 2026-09-15 in favour of `scripts/pipelines/`; see
> docs/REPRODUCE.md. Kept for the run log, not as instructions.

# Overnight run — 2026-09-14, read this first

## What is running

**Reservoir generation, 210 screen cells.**
`run_screengen.sbatch` — SLURM array, `cpuq`, `slow_nice` qos, 36 concurrent.
Outputs land in `outputs/screen/cache/*.npz`, one per cell, params stamped inside.

**Watchdog** (`run_screen_watchdog.sbatch`, job 3186828).
Every 10 min it counts completed cells, and if the queue is empty while cells remain
it resubmits the array. It exits cleanly once all 210 exist, or after 12 resubmits if
something is failing repeatedly. It already self-healed once tonight, at 103/210.

## First thing in the morning

```bash
ssh bamdev4
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
tail -20 outputs/screen/STATUS.txt                     # the timeline
find outputs/screen/cache -name '*.npz' -size +0c | wc -l   # should be 210
grep -l '^FAIL' logs/screengen/*.out | head             # any failures
```

`STATUS.txt` ends with either `ALL CELLS COMPLETE` or a resubmit-budget message. If
the latter, the failures are in `logs/screengen/`.

## Safety properties (tested, not assumed)

- **Resubmission is idempotent.** Each array task skips a cell whose output already
  exists and is non-empty — verified by rerunning a completed index, which printed
  `SKIP already present`. So the watchdog cannot redo finished work or corrupt it.
- **A partial write cannot masquerade as done.** The size test is `-size +0c`, and
  `albench generate` writes via numpy's own save, so a truncated file fails the
  reload rather than counting as complete.
- **The watchdog is bounded.** MAX_RESUB=12 stops an infinite resubmit loop if
  something is deterministically broken.

## Oracle labelling — RUNNING, but only the 30k half

The test passed after the handoff was written: the AG oracle loads in 173 s and
labels at **29.7 seq/s**, giving finite, sensibly-distributed labels
(mean 0.542, sd 1.115).

That rate reframes the plan. Per cell it is **16.8 min at D=30k but 2.8 h at
D=300k**, so labelling all 210 cells would be **~323 GPU-h for a SINGLE oracle
model** — and ~3,200 with the 10-model ensemble. Not feasible, and not necessary:
the screen exists to find the best PARAMETERS per strategy, and parameter effects
should be visible at 30k. Scale is what the 300k arm is for, and that comes after
the parameters are chosen.

So: **105 cells at D=30k are being labelled** (`run_label30k.sbatch`, job 3187122,
6 concurrent, ~29 GPU-h total) with its own watchdog (job 3187127, status in
`outputs/screen/STATUS_LABEL.txt`). The 97 cells at D=300k are generated but
**deliberately unlabelled** — decide what that arm needs before spending ~270 GPU-h
on it.

```bash
tail -20 outputs/screen/STATUS_LABEL.txt
find outputs/screen/cache -name '*__labeled.npz' -size +0c | wc -l   # want 105
```

Labelled cells carry `oracle_labels`, plus the original `strategy` / `params` /
`seed` and an `oracle_id` stamp, and are written atomically via os.replace so a
killed job cannot leave a half-written file that looks complete.

A glob to watch out for: `*d30000*` also matches `d300000`. My first submission did
exactly that and would have labelled the expensive 300k cells too; it was cancelled
and the list rebuilt with `*__d30000__*`.

**Student training.** Still driven by the existing bakeoff infrastructure, not by
`albench`. Connecting the two is the next piece of work.

## Note on cost

The 210 cells are ~53 GPU-h of student training when it comes to that. Oracle
labelling dominates and is charged separately — at the measured AG rate a single
300k cell is hours, so labelling all 210 cells is NOT something to launch without
sizing it first.
