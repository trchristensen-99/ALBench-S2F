# ar2 — a finished experiment, kept for reproducibility

AutoResearch 2: an agentic controller that proposed, trained and gatekept student
architectures through its own CLI, run as 9 + 24 dispatcher cells and completed. Nothing in
`albench/` imports it, no current registry cell runs it, and no number in the current study
depends on re-running it. It is kept because its outputs are cited and its tests document
what the controller guaranteed.

**Do not build on this.** New shared code goes in `albench/` (see `AGENTS.md`); the loop
that is live today is `albench/arch_search/arch_autoresearch.py` with
`albench/arch_search/agent_loop.py`.

It stays at the top level rather than moving under a `study/` tree for one practical
reason: `ar2` is an importable package and both its own modules and its tests import it as
`ar2.*`, so moving it would mean either rewriting every import or putting a second
directory on `sys.path`. That is more churn than an archived experiment is worth.

Its tests run with the rest of the suite (`pytest tests/ar2`). One of them,
`test_stress_set_and_rule`, had been failing since `stress_check`'s explosion threshold
moved from 10 to 25 training SDs; the test was asserting the old threshold and was corrected
on 2026-11-20 rather than deleted, because the rule it checks is still the rule.
