"""Nov-11: replicates 5 and 6 of the fixed-geometry ladder at 10k, 30k and 100k.

`register_nov11b.py` took the three affordable rungs to four replicates this morning and its
reasoning is unchanged: the designed-set gain of selection over the control is being claimed at
roughly the size of the program's own single-run spread (sigma ~= 0.008), so the error bar is the
headline's weakest joint. Four replicates give a standard error of about 0.004; six give about
0.0033. That is a smaller step than the one from two to four, and it would not be worth buying on
its own.

It is worth buying now because the capacity is otherwise idle. The registry has run down to about
eleven unstarted cells against 39 active, and the dispatcher has been sitting on two to four free
H100s and five free V100s for eight consecutive cycles with only three jobs pending on Priority --
the queue is thin, not blocked. Idle GPUs convert to nothing; these convert to a tighter interval
on the number the scaling section leads with.

    al57  10k -> 20k     ~8 GPU-h
    al56  30k -> 60k     ~12 GPU-h
    al55  100k -> 200k   ~55 GPU-h

al53 (300k -> 600k) is again left alone, and for a second reason now: its replicates 1 and 2 are
still pending on the 48 h wall, so deepening that rung would queue work behind work that has not
finished.

About 75 GPU-hours. Same cells, same guard: every new cell's command must equal the rep-1 command
with the replicate number substituted, checked against the defining module rather than copied.

    python scripts/program_oct22/register_nov11c.py [--dry]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov11b as b  # noqa: E402

if __name__ == "__main__":
    b.NEW_REPS = (5, 6)
    b.BACKUP_SUFFIX = ".bak_nov11c"
    b.main()
