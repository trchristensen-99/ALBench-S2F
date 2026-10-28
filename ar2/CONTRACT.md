# ar2 implementation contract (internal, for the three people/agents building it in parallel)

Source of truth for behaviour: "AutoResearch v2.docx" (text at the session scratchpad `autoresearch_v2.txt`, sections 1-17). This file only fixes the
seams between the three work packages so they can be built independently and plugged together. Where this file and the spec disagree on an interface, this
file wins; where it is silent, follow the spec.

Package root: `ar2/` at the repository root (Python package `ar2`), tests in `tests/ar2/`. No imports from `albench` inside `ar2` except `albench.ensemble.stack`
(ElasticNet stack) in report/export code and the vendored Muon optimizer (`albench.ensemble.menu`/`muon.py`) when `optimizer: muon` is asked for.
Python >= 3.11, `from __future__ import annotations`, type hints on public functions, ruff-clean (`ruff format` + `ruff check`).
Local tests run on CPU with `pytest tests/ar2 -q` and must not need a GPU, the network, the `claude` CLI or the HPC. Do NOT ssh to or touch the HPC.

## Work packages and file ownership (never edit another package's files; ask via the interface below)

* **A - compute side** (everything that runs inside a job): `ar2/data/`, `ar2/evaluator/`, `ar2/gatekeeper.py`, `tests/ar2/test_data.py`,
  `test_evaluator.py`, `test_gatekeeper.py`, `tests/ar2/fixtures/broken_models/`.
* **B - orchestration**: `ar2/__init__.py`, `ar2/config.py`, `ar2/config/default.yaml`, `ar2/memory/`, `ar2/jobs/`, `ar2/controller.py`, `ar2/tuner.py`,
  `ar2/orchestrator.py`, `ar2/sim.py`, `ar2/report.py`, `ar2/cli.py`, `ar2/__main__.py`, `tests/ar2/test_memory.py`, `test_controller.py`, `test_sim.py`,
  `test_jobs.py`, `test_report.py`.
* **C - proposer and prompts**: `ar2/proposer/`, `ar2/prompts/v1/*.md`, `ar2/views.py`, `ar2/tagger.py`, `ar2/lessons.py`, `tests/ar2/test_views.py`,
  `test_proposer.py`, `test_lessons.py`.

## Workspace layout (one search = one directory `ws/`; orchestrator is its single writer)

```
ws/config.yaml (resolved)  ws/ar2.db  ws/meta/   ws/candidates/<code_hash>/{model.py,card.json,tags.json,gate.json}
ws/runs/<run_id>/{spec.json,metrics.json,curves.csv,preds_val_loop.npy,log.txt,[ckpt.pt]}
ws/calls/<call_id>/{prompt.md,response.json,stdout.txt,stderr.txt}   ws/queue/{pending,running,done}/   ws/prompt_proposals/   ws/stress/stress_seqs.npy
```
`code_hash` = sha1[:16] of the normalised source (comments/whitespace removed via `ast.unparse(ast.parse(src))`). `run_id` = `r_` + sha1[:12] of
`code_hash|recipe_hash|pool_id|seed|purpose`. `recipe_hash` = sha1[:12] of canonical JSON (sorted keys) of the recipe dict (spec 4.3; `hparams` inside it).

## Seam 1 - evaluator job (A provides, B consumes)

`python -m ar2.evaluator.train --spec ws/runs/<run_id>/spec.json` (exit code 0 even for diverged/oom/exploded runs; they are results). spec.json:
```json
{"run_id": "r_...", "code_dir": "ws/candidates/<hash>", "recipe": {...spec 4.3...}, "seed": 0, "purpose": "first_look",
 "pool_id": "...", "D": 30000, "out_dir": "ws/runs/<run_id>",
 "train_manifest": "path.json", "val_loop_manifest": "path.json", "val_select_manifest": null,
 "time_cap_s": 755.0, "cap_extension_used": false, "speed_factor": 1.0,
 "stress_path": "ws/stress/stress_seqs.npy", "save_preds": true, "save_ckpt": false, "label_noise_sd": 0.0, "rc_average": true,
 "early_term": {"checkpoints": [0.5, 0.75], "margin": 0.03, "reference": {"onecycle": {"0.5": 0.61, "0.75": 0.70}}} }
```
`early_term.reference` is the best val r at each checkpoint (fraction of max_epochs) among that family's earlier runs that used the same schedule; absent or empty
means no early termination. `speed_factor` (1 on H100, 5 V100, 8 MIG slice) multiplies the time cap. The job writes `curves.csv`, `log.txt`, optional preds/ckpt, and
`metrics.json` last (tmp file + rename) with exactly the spec 9.2 fields plus `cost_equiv` (epochs_run / max_epochs, 1.0 for a full run) and `cap_s_used`.
`status` in {ok, diverged, exploded, oom, time_cap, error, early_stopped_by_tuner}. A Phase-D spec carries `val_select_manifest`; metrics.json then also has
`val_select_r`; ordinary specs must not contain it.

Data (A): pools/validation are npz files accepted through a manifest JSON: `{"path": ".../x.npz", "sha256": null, "n": N, "role": "train|val_loop|val_select",
"pool_id": "...", "reservoir": "...", "generation_seed": 42, "D": 30000, "label_source": "oracle_v3", "tasks": ["K562"], "seq_key": "sequences",
"label_keys": {"K562": "oracle_labels"}, "chrom_key": null, "index_file": null}`. `index_file` (optional .npy) selects rows (used to split one npz into val_loop/val_select and to
take the first D rows of a bigger pool). `loader.load_split(manifest_path, expect_role) -> Split(seqs, labels [N,T] float32, meta)`; refuses roles other than the three;
`loader.verify_no_overlap(train, *others)` by sequence hash incl. reverse complements; rejects any length != 200.

## Seam 2 - gatekeeper (A provides, B and C consume)

`ar2.gatekeeper.run_gate(candidate_dir: Path, cfg: dict, *, hparams: dict | None = None, existing_hashes: set[str] | None = None, device: str = "cpu", speed_probe: bool = False,
speed_factor: float = 1.0, time_cap_s: float | None = None) -> GateResult` with `GateResult(ok: bool, code: str | None, action: "pass"|"repair"|"reject"|"amp_off",
detail: str, traceback: str, measured: dict)`. `code` in {files, unsafe, duplicate, build, params, shape, constant, coverage, batch_leak, determinism, learn, amp, speed,
near_duplicate} (spec section 7; near_duplicate is checked by the orchestrator after tagging via `gatekeeper.near_duplicate(tags, hparams, others)`). `measured` has n_params,
throughput_seq_per_s, peak_mem_gb, amp_safe, code_hash. Static checks run in-process on source text (`static_check(source) -> (ok, code, detail)`); every dynamic check runs
in a fresh sandboxed subprocess (no network env, resource limits, timeout) that is the only place candidate code is imported. Check randomisation per spec.
`write_gate_json(candidate_dir, result)`. `normalise_source(src) -> str` and `code_hash(src) -> str` live here too.

## Seam 3 - proposer, prompts, tagger (C provides, B consumes)

```python
class ProposerResult: ok: bool; files: dict[str, str]; parsed: object | None; raw_stdout: str; raw_stderr: str; model_id: str | None; cli_version: str | None;
                      usage: dict; duration_s: float; exit_code: int; error: str | None
class Proposer(Protocol): def call(self, prompt: str, workdir: Path, expect: Literal["sketches","candidate","tags","lesson"], timeout_s: int) -> ProposerResult
make_proposer(cfg: dict) -> Proposer                 # backend: claude_code_cli | mock
ar2.views.ViewContext  (dataclass, plain data; B builds it from the DB): D, n_recipes, max_params, time_cap_min, noise_floor, taxonomy (str), coverage: list[dict(family, candidates, state, note)],
   near: list[dict(summary, family)] (do-not-repeat, already truncated by B to 12 or 20), lessons_global: list[str], lessons_family: list[str],
   family: dict | None (REFINE: family, confirmed_mean, confirmed_se, top_mean, top_se, n_seeds, params, throughput, diagnostics: dict, recipe_sensitivity: dict, earlier_changes: list[dict(text, effect, se, accepted)]),
   failure_view: list[dict(code, description)] | None, extra_constraints: list[str], forced_families: list[str] | None, speed_note: str | None
ar2.views.render(mode, ctx, assignment: dict) -> str     # mode in SKETCH|EXPLORE|REFINE|REPAIR|TAGGER|LESSON; asserts no '{{' left and token estimate <= max_prompt_tokens
ar2.views.prepare_workdir(workdir: Path, mode: str, *, parent_code: str | None, reference_code: str) -> None   # writes CONTRACT.md, reference_model.py, selfcheck.py, parent_model.py
ar2.tagger.tag(proposer, code: str, taxonomy: list[str], cfg: dict, workdir: Path) -> dict   # {family, core_block, uses_attention, ..., n_params filled later by B, justification}
ar2.tagger.rule_check(code: str, tags: dict) -> dict      # AST rules (uses nn.MultiheadAttention -> uses_attention True, nn.LSTM/GRU -> recurrence ...) override/annotate the LLM tags
ar2.lessons.make_evidence(kind, ...) -> dict; ar2.lessons.word(proposer, evidence: dict, cfg, workdir) -> str   # numbers appended by code, never by the LLM
ar2.prompts.PROMPT_FILES, ar2.views.prompt_version(cfg) -> str ("v1+<sha1 of all template files>")
```
Reference models for `prepare_workdir`: three small valid models (cnn_residual, rnn_hybrid, transformer_hybrid) in `ar2/proposer/reference_models/`, rotated by a counter.
The CLI adapter: `claude -p` reading the prompt from stdin, `--output-format json`, `--model <full id>`, `--allowedTools "Read Write Edit Bash(python selfcheck.py)"`, `--max-turns`,
run with cwd = a scratch dir under `scratch_root` (outside the repo) and an isolated `CLAUDE_CONFIG_DIR`; at init it runs `claude --version` / `claude --help` and fails with a
message naming any missing option. Auth on the cluster comes from the environment (`CLAUDE_CODE_OAUTH_TOKEN`), so the adapter must pass the environment through but nothing else
that is user-level. Rate-limit / auth errors must be raised as `ProposerPaused(retry_after_s)`; 5 consecutive failures as `ProposerHalt`.

## Seam 4 - jobs (B provides)

`ar2.jobs.base.JobBackend` as in spec 4.6. `ar2.jobs.local.LocalBackend` (subprocess per GPU id, for tests/smoke) and `ar2.jobs.queue.QueueBackend` (file queue
`ws/queue/{pending,running,done}`; `submit` writes `pending/<run_id>.json` idempotently; external workers started with `python -m ar2.jobs.worker --queue ws/queue
[--idle-exit-s 1800]` claim a job by atomic rename to `running/`, run the evaluator subprocess, move to `done/`; a worker also runs `speed` gate probes: job kind
`gate_speed`). The worker sets `speed_factor` from `torch.cuda.get_device_name` (H100 1, V100 5, MIG/1g.12gb 8) and patches it into the spec before running; stale `running/`
entries (worker died: no heartbeat file update for 10 minutes) are requeued by the orchestrator.

## Hard requirements shared by all packages

* Test data never appears anywhere. Validation = `val_loop` only; `val_select` only in Phase D specs; never printed into a prompt.
* Every module must be importable without torch where the spec allows (B, C should not import torch at import time).
* No step may mutate the repository outside `ws/` and the scratch root.
* Report honestly in your final message: what is implemented, what is a stub, which spec tests pass.
