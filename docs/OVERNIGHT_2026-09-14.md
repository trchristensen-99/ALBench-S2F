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

## What is NOT running, and why

**Oracle labelling.** Generation produces sequences only. The labelling test
(`logs/testlabel_*.out`, job 3186904) was still loading the AlphaGenome model when
the laptop went down, so I did not commit the night to a path I had not seen succeed
end-to-end. Sequences are the prerequisite and are cheap; labelling is the expensive
step and is worth launching deliberately once that test is confirmed.

Check it first thing:
```bash
grep -E 'OK:|PROJECTED|Error' logs/testlabel_3186904.out
```
If it reports a seq/s figure, labelling works and can be launched as an array over
`outputs/screen/cache/*.npz`. If it errored, the likely cause is the known-fragile
`_load_oracle` path in `experiments/exp1_1_scaling.py`.

**Student training.** Still driven by the existing bakeoff infrastructure, not by
`albench`. Connecting the two is the next piece of work.

## Note on cost

The 210 cells are ~53 GPU-h of student training when it comes to that. Oracle
labelling dominates and is charged separately — at the measured AG rate a single
300k cell is hours, so labelling all 210 cells is NOT something to launch without
sizing it first.
