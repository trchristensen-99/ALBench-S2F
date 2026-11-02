# What limits the LLM runs, and what was changed (Nov 2, 2026)

Every loop cell calls the Claude Code CLI (`claude -p`) from the cluster with the lab account's token. All cells and the interactive sessions share one **5-hour session limit** (the CLI says `You've hit your session limit · resets 4:50pm (America/New_York)`; 553 such messages in the job logs, no weekly-limit or model-specific message). When it is hit, cells exit with code 75, the dispatcher pauses LLM submissions and probes until the window resets. Four things made it worse than it had to be; three are fixed.

## 1. What fills the window

Hourly usage was summed from the CLI's own session logs (`scripts/analysis/llm_usage_windows.py`; laptop and both cluster homes) for the four windows that ended in a limit message:

| window (hit at) | cluster calls | cluster output | cluster cache write | cluster cache read | laptop output | laptop cache read | output, both |
|---|---|---|---|---|---|---|---|
| Oct 29 18:10-23:10 (22:05) | 934 | 2.71 M | 8.8 M | 22.0 M | 0.08 M | 23 M | 2.79 M |
| Oct 29 23:10-Oct 30 04:10 (03:32) | 838 | 3.00 M | 7.8 M | 20.8 M | 0.05 M | 15 M | 3.05 M |
| Oct 30 04:10-09:10 (07:35) | 888 | 2.70 M | 8.8 M | 21.6 M | 0.01 M | 13 M | 2.72 M |
| Nov 2 11:50-16:50 (16:45) | 960 | 2.27 M | 10.1 M | 23.2 M | 0.49 M | 157 M | 2.77 M |

* The loop cells are almost the whole load, and **total output tokens at the moment of the limit were the same in all four windows, 2.7 to 3.05 M** (cluster plus laptop), while cache reads differed seven-fold (23 M to 157 M) and cache writes by 30%. The budget behaves like roughly 2.8 M output tokens per 5 hours, with cache reads counting for little. A window lasts 5 hours only when fewer than about 550 k output tokens per hour are generated; the cells generated 290 to 620 k per hour with 9 to 11 running.
* Interactive use (this session included) displaced about 15% of the window on Nov 2 and almost nothing on Oct 29-30.

## 2. Where the tokens come from

A trivial prompt (five words) costs this much per call on the cluster, by CLI configuration (tokens and the CLI's own `total_cost_usd`):

| configuration | cache write | cache read | uncached input | cost |
|---|---|---|---|---|
| old: repository as working directory, tools on | 12,640 | 19,113 | 2 | $0.054 |
| empty directory, `--tools ""` (deployed at 17:50) | 5,124 | 3,401 | 2 | $0.021 |
| + `--system-prompt` instead of the Claude Code default | 0 | 0 | 327 | $0.0007 |

So 10 of the 10.1 M cache-write tokens per window were fixed overhead: every call restarted a full Claude Code session. The second cost is hidden reasoning. The same real proposer prompt (16 k characters, six proposals back) generates **5,353 output tokens at the default effort and 2,683 with `--effort medium`** (2,644 with thinking disabled, 2,404 at `--effort low`) for a visible answer of about 1,600 tokens. The loop's own counter (`approx_tokens_out`, characters / 4) shows 0.9 k per call; the CLI reported 2.4 k per call on average, a factor of 2.7 that is almost all thinking. Per round the loop used about 11 k output tokens; the visible text is about 3.5 k.

## 3. Changes (all deployed Nov 2; `experiments/llm_autoresearch.py::_call_claude_cli`)

* empty working directory, `--tools ""`, the loop's own system prompt in `--system-prompt` (no Claude Code prompt, no `AGENTS.md`, no repository access), `--effort medium` (`LLM_CLI_EFFORT`, `default` to leave it out, `LLM_CLI_ISOLATE=0` for the old behaviour). A real proposer call through the new path returned six complete proposals in 18 s (35 s before). Tests: `tests/test_llm_cli.py`.
* Expected effect: output per call halves, fixed overhead per call drops from about 32 k tokens to about 0.3 k. If the limit is output-weighted as the table suggests, the same window holds about twice as many rounds; if it also weights cache writes, more. It is measured by rerunning `llm_usage_windows.py` over the next window and counting output tokens per round.
* The dispatcher's LLM probe could not find `claude` after the cron restart (no `~/.local/bin` on its PATH), logged `still limited`, and kept all LLM cells paused from 16:50 to 18:00 on a limit that had lifted at 16:50 (fixed, `cluster_live/dispatcher.py`). Its "LLM last 60 min" line also read the old home directory and undercounted (fixed: both homes).

## 4. Further levers, if more throughput is needed

| lever | effect | cost |
|---|---|---|
| `ideas_per_round` 6 to 4 | the proposer writes about half of the visible output; about -15% output per round | fewer candidates for the coordinator to choose from |
| a lighter model for the coder (about 1 k visible tokens x 2 per round) | depends on how the limit weights models (unknown) | code quality, repair rate |
| API key instead of the subscription token for the cells (`ANTHROPIC_API_KEY`, no session limit, billed per token) | removes the 5-hour wall; the CLI's own cost accounting implies about $60 to 75 per 5-hour window at the old load (about $12 to 14 per hour with 9 to 11 cells) and roughly half that after the changes above | money; a decision for you |
| fewer concurrent cells | nothing: the budget is per window, concurrency only decides how fast it is used | - |

Not a lever: the cache reads (the laptop used 157 M in the Nov 2 window without moving the limit).
