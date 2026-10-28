-- ar2 ledger and derived tables (spec 10.2). Ledger tables (calls, candidates, runs, decisions, sketches, lesson_text, final_scores, meta) are append-only:
-- a row's status fields may advance, nothing is deleted. Derived tables (families, lessons) are rebuilt from the ledger by `ar2 rebuild`.
-- Columns marked (+) are additions to the spec's schema that the orchestrator needs; the spec's columns keep their names and types.

CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY, mode TEXT, assignment TEXT, prompt_version TEXT,
  prompt_path TEXT, response_path TEXT, model_id TEXT, cli_version TEXT,
  status TEXT, duration_s REAL, tokens_in INT, tokens_out INT, created_at TEXT);

CREATE TABLE IF NOT EXISTS candidates (
  code_hash TEXT PRIMARY KEY, call_id TEXT, parent_code_hash TEXT,
  family TEXT, family_claimed TEXT, tags TEXT, card TEXT,
  gate_status TEXT, gate_code TEXT, gate_detail TEXT, repair_attempts INT,
  n_params INT, throughput REAL, amp_safe INT, source TEXT, created_at TEXT,
  phase TEXT,          -- (+) phase in which the candidate was created: A | M | P
  sketch_id TEXT,      -- (+) the sketch it implements, if any
  hparams TEXT);       -- (+) HPARAMS defaults (json) as reported by the gatekeeper

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, code_hash TEXT, recipe TEXT, recipe_hash TEXT,
  pool_id TEXT, D INT, seed INT, purpose TEXT,        -- first_look|tune|deep_tune|confirm|refine|screen|procedure|final
  status TEXT, stop_reason TEXT, val_r REAL, val_mse REAL, val_spearman REAL,
  train_r REAL, diagnostics TEXT, epochs_run INT, cost_equiv REAL,
  wall_time_s REAL, job_id TEXT, submitted_at TEXT, finished_at TEXT,
  phase TEXT,          -- (+) A | S | P | M | D
  category TEXT,       -- (+) explore | refine | deep_tune | fixed (anchors, screen, final): the budget category of 5.7
  family TEXT,         -- (+) family of the candidate at submission time
  schedule TEXT,       -- (+) lr schedule (early termination compares runs of the same schedule only)
  attempt INT,         -- (+) infrastructure resubmissions so far
  meta TEXT,           -- (+) json: replaces (OOM replacement), option/anchor (screen), parent_code_hash (refine), censored, still_improving ...
  UNIQUE (code_hash, recipe_hash, pool_id, seed, purpose));

CREATE TABLE IF NOT EXISTS families (               -- derived
  family TEXT PRIMARY KEY, state TEXT, n_candidates INT, n_runs INT,
  best_code_hash TEXT, best_recipe_hash TEXT, screen_best REAL,
  confirmed_mean REAL, confirmed_sd REAL, n_confirm INT,
  recipe_spread REAL, refinements INT, refinements_without_gain INT,
  triage TEXT, second_chance_used INT, censored INT, updated_at TEXT,
  detail TEXT);        -- (+) json with the rest of the FamilyState

CREATE TABLE IF NOT EXISTS lessons (                -- derived
  lesson_id TEXT PRIMARY KEY, scope TEXT, statement TEXT,
  effect REAL, se REAL, n INT, evidence_run_ids TEXT,
  status TEXT,                        -- confirmed|retracted
  created_at_cost REAL,
  kind TEXT, key TEXT);               -- (+) evidence kind and the comparison it belongs to (retraction groups by scope+key)

CREATE TABLE IF NOT EXISTS decisions (
  step INT, mode TEXT, target TEXT, reason TEXT, archive_hash TEXT, created_at TEXT,
  payload TEXT);       -- (+) json with the action parameters, the guard/adaptation signal values

CREATE TABLE IF NOT EXISTS final_scores (           -- Phase D only; never read by views.py (only ar2/memory/final.py and the report touch it)
  code_hash TEXT, recipe_hash TEXT, seed INT, val_select_r REAL, val_loop_r REAL);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);  -- schema version, config hash, pinned model, sigma_seed, prompt hash

CREATE TABLE IF NOT EXISTS sketches (                -- (+) ledger of SKETCH output; status advances unused -> assigned -> used
  sketch_id TEXT PRIMARY KEY, call_id TEXT, family TEXT, name TEXT, body TEXT,
  novelty REAL, status TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS lesson_text (             -- (+) wording written by the LLM for an evidence record; numbers always come from code
  evidence_key TEXT PRIMARY KEY, statement TEXT, call_id TEXT, created_at TEXT);

CREATE INDEX IF NOT EXISTS runs_code ON runs (code_hash);
CREATE INDEX IF NOT EXISTS runs_family ON runs (family, purpose);
CREATE INDEX IF NOT EXISTS cand_family ON candidates (family);
