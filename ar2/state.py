"""Plain data the controller reasons over, and the actions it returns.

Why this module exists: the controller (ar2/controller.py) must be provable on a simulator without a database, an LLM or a GPU, so it never touches SQLite. The
orchestrator builds a :class:`Snapshot` from the ledger (facts only: runs, candidates, calls, decisions), calls ``Controller.decide(snapshot, ...)`` and executes the
returned :class:`Action` list. The controller re-derives every family state from the facts (see ``controller.derive_family_state``), so the ``families`` table is a
cache that ``ar2 rebuild`` can reproduce.

Everything here is JSON-serialisable through ``to_dict`` / ``from_dict``; no field holds an object that is not plain data.

Units: scores are Pearson r on val_loop; budgets and spend are in *run-equivalents* (1 per full run, ``epochs_run / max_epochs`` for a run that stopped early).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

MODES = (
    "SKETCH",  # LLM call: 10 one-line ideas (no training)
    "EXPLORE",  # LLM call: implement one sketch (or, for an anchor with source=reference, register the reference model without a call)
    "TUNE",  # no call: submit recipes (purpose first_look | tune | second_chance | refine) for a family's candidate
    "CONFIRM",  # no call: retrain best candidate+recipe on seeds (confirm 1-3)
    "REFINE",  # LLM call: one structural change to a confirmed family's best model
    "REPAIR",  # LLM call: fix a gatekeeper failure
    "TUNE_DEEP",  # no call: local perturbations of a menu-eligible family's best recipe
    "PROCEDURE",  # LLM call (phase P): nominate / implement a loss or training plug-in
    "SCREEN",  # no call: procedure-screen runs on the anchors (phase S) and their adoption decision
    "FINAL",  # no call: phase D reconfirmation on seeds 4-6, scored once on val_select
    "NOTE",  # not executable: a logged decision (guard fired, outcome of a refinement, phase change, stop, menu). target['kind'] says what
)
FAMILY_STATES = (
    "empty",  # in the taxonomy, no candidate yet
    "first_look",  # a candidate passed the gate; its 3 recipes are not all finished
    "parked",  # best of first look more than keep_margin below the top confirmed mean; one second chance, then retired
    "tuning",  # within keep_margin of the top: 8 recipes in total
    "confirming",  # tuned and within confirm_margin of the best confirmed family (anchors always): 3 seed runs
    "confirmed",  # 3 seeds finished
    "plateaued",  # plateau_after consecutive refinements not accepted
    "retired",  # confirmed mean more than refine_margin below the top, or a parked family that failed its second chance
    "tuned_only",  # 8 recipes done, best screening score more than confirm_margin below the best confirmed family
)
PHASES = ("A", "S", "P", "M", "D", "DONE")
CATEGORIES = ("explore", "refine", "deep_tune")
RUN_CATEGORIES = CATEGORIES + ("anchor", "screen", "procedure", "final")
PURPOSES = (
    "first_look",
    "tune",
    "second_chance",
    "confirm",
    "refine",
    "deep_tune",
    "screen",
    "procedure",
    "final",
)
REFINE_ANGLES = (
    "capacity (width and depth)",
    "receptive field and pooling",
    "normalisation and activation",
    "regularisation inside the architecture",
    "positional and strand handling",
    "efficiency at equal accuracy",
)
STATUSES = ("ok", "diverged", "exploded", "oom", "time_cap", "error", "early_stopped_by_tuner")


class _Plain:
    """Mixin: dict round trip for dataclasses made only of plain data (nested plain dataclasses are rebuilt through ``_nested``)."""

    _nested: dict[str, Any] = {}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        kw = {}
        names = {f.name for f in fields(cls)}
        for k, v in d.items():
            if k not in names:
                continue
            spec = cls._nested.get(k)
            if spec is None or v is None:
                kw[k] = v
            elif spec[0] == "one":
                kw[k] = spec[1].from_dict(v)
            elif spec[0] == "list":
                kw[k] = [spec[1].from_dict(x) for x in v]
            elif spec[0] == "dict":
                kw[k] = {kk: spec[1].from_dict(x) for kk, x in v.items()}
        return cls(**kw)


@dataclass
class RunInfo(_Plain):
    """One finished (or open) run, as the controller needs it. B1 fills it from the ``runs`` table + metrics.json.

    run_id, code_hash, recipe_hash: ledger keys.  family: the tagger's family for ``code_hash``.
    recipe: the recipe dict (spec 4.3).  schedule: recipe['schedule'] (kept separate for same-schedule comparisons).
    seed: 0 for screening and tuning, 1-3 confirmation, 4-6 final.  purpose: one of PURPOSES.
    category: explore | refine | deep_tune | anchor | screen | procedure | final - which budget line the run is charged to (B1 stores it on the run row).
    phase: A | S | P | M | D - the phase the run was charged to.
    status: ``ok`` once finished normally; ``pending`` / ``running`` while open (then val_r is None); otherwise one of STATUSES.
    val_r: validation r (None while open). train_r: training-fit r on a 10k subsample (None when unknown). cost_equiv: run-equivalents spent so far.
    epochs_run, max_epochs, best_epoch: from metrics.json.  still_improving: spec 9.2.  improvement_per_epoch: mean r gain over the last 5 epochs (for 8.5).
    censored: the run stopped on the time cap.  exploded: the stress screen failed (cannot be a family's best).
    checkpoint_best: {"0.5": best val r up to 50% of max_epochs, "0.75": ...} from curves.csv, for early-termination references.
    diagnostics: extra metrics.json fields shown to the proposer (pred_sd_ratio, mse_by_activity_quintile, ...).  cap_multiplier: 1.5 when the run used the extended time cap.
    """

    run_id: str = ""
    code_hash: str = ""
    family: str = ""
    recipe_hash: str = ""
    recipe: dict = field(default_factory=dict)
    schedule: str = "onecycle"
    seed: int = 0
    purpose: str = "first_look"
    category: str = "explore"
    phase: str = "M"
    status: str = "ok"
    val_r: float | None = None
    train_r: float | None = None
    cost_equiv: float = 0.0
    epochs_run: int = 0
    max_epochs: int = 40
    best_epoch: int = 0
    still_improving: bool = False
    improvement_per_epoch: float = 0.0
    censored: bool = False
    exploded: bool = False
    checkpoint_best: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    cap_multiplier: float = 1.0

    @property
    def finished(self) -> bool:
        return self.status not in ("pending", "running")


@dataclass
class ChildInfo(_Plain):
    """An in-flight REFINE child of a confirmed family (None in ``FamilyState.pending_refinement`` when there is none).

    code_hash / parent_code_hash: ledger keys.  angle: the REFINE angle used.  recipes_planned: recipes already created for the child (3 when screening is fully planned).
    screen_scores: finished seed-0 val r values (exploded runs excluded).  screen_open: screening runs not finished.
    confirm_planned: the 3 confirmation runs were created.  confirm_vals: finished confirmation val r.  confirm_open: confirmation runs not finished.
    best_recipe: the child's best screening recipe (used for confirmation).  outcome_recorded: B1 has applied the NOTE refinement_outcome for this child.
    parent_screen_best / parent_confirmed_mean: the parent's best screening score and confirmed mean when the child was created (the references of spec 5.1 / 5.2).
    """

    code_hash: str = ""
    parent_code_hash: str = ""
    angle: str = ""
    recipes_planned: int = 0
    screen_scores: list = field(default_factory=list)
    screen_open: int = 0
    confirm_planned: bool = False
    confirm_vals: list = field(default_factory=list)
    confirm_open: int = 0
    best_recipe: dict = field(default_factory=dict)
    outcome_recorded: bool = False
    parent_screen_best: float | None = None
    parent_confirmed_mean: float | None = None


@dataclass
class DeepInfo(_Plain):
    """Deep-tune bookkeeping of one family.

    recipes_done: deep-tune recipes finished (all batches).  batch_index: the current undecided batch; batch_runs: runs created in it (0 = idle, a new batch may start; a full batch has batch_size);
    batch_scores: val r of the current batch's finished valid runs; batch_open: runs of the batch not finished.
    batch_best_recipe: best recipe of the current batch so far.  prev_screen_best: the family's best screening score before this batch.
    confirm_recipe: recipe under confirmation (None when none).  confirm_vals / confirm_open: its 3 seed runs.  attempts / accepted: finished confirm decisions.
    """

    recipes_done: int = 0
    batch_scores: list = field(default_factory=list)
    batch_open: int = 0
    batch_best_recipe: dict = field(default_factory=dict)
    prev_screen_best: float | None = None
    confirm_recipe: dict | None = None
    confirm_vals: list = field(default_factory=list)
    confirm_open: int = 0
    attempts: int = 0
    accepted: int = 0
    batch_index: int = 0
    batch_size: int = 0
    batch_runs: int = 0
    confirm_recipe_hash: str | None = None


@dataclass
class FamilyState(_Plain):
    """Facts about one family. The controller derives the lifecycle state from these (``state`` here is the cached value, only informative).

    family: name from the taxonomy (``other:<name>`` is its own family).  state: cached lifecycle state.  anchor: one of the three anchor families.
    n_candidates / n_runs: counts in the ledger.  best_code_hash / best_recipe / best_recipe_hash: the family's current best candidate and its tuned recipe (best
    non-exploded screening run; replaced when a refinement or deep-tune recipe is accepted).  screen_best: best non-exploded seed-0 val r over all of the family's recipes.
    confirmed_mean / confirmed_sd / n_confirm: mean, SD and count of the 3 confirmation runs of best_code_hash with best_recipe (the selecting run excluded) - None until confirmed.
    recipes_done: finished screening recipes (purposes first_look, tune, second_chance of the current best candidate lineage; a diverged recipe counts).
    recipes_open: screening runs planned or running, not finished.  recipes_planned: recipes created so far for the lineage (done + open).
    first_look_done / first_look_open: of the 3 first-look recipes.  first_look_best: best val r among them (None when none finished OK).
    second_chance_used: the triage recipes were created.  second_chance_done / second_chance_open: of those 3.
    recipe_results: [{recipe_hash, val_r, schedule, status, code_hash, seed, purpose}] for finished screening runs (seed 0), incl. diverged (val_r None).
    confirm_open: confirmation runs of the family's own best (explore confirm) not finished.  confirm_vals: its finished values.
    refinements / refinements_without_gain: decided refinements and the current run of rejected ones.  refinement_history: [{child, accepted, effect, se}].
    pending_refinement: ChildInfo of the in-flight refinement, if any.  angle_index: refinements attempted so far (rotates the REFINE angle).
    censored: best run was censored by the time cap.  censored_extended: the 1.5x extension was already used.  triage: diagnosis chosen for the second chance.
    deep: DeepInfo.  menu_flagged: output of ``menu_select`` (informative).  last_diagnostics: diagnostics of the best run for prompts (train_r, val_r, pred_sd_ratio ...).
    final_vals: finished phase-D seed runs (val_loop r); final_open: phase-D runs not finished.  summary: one-line text of the best model (for views).
    """

    family: str = ""
    state: str = "empty"
    anchor: bool = False
    n_candidates: int = 0
    n_runs: int = 0
    best_code_hash: str | None = None
    best_recipe: dict = field(default_factory=dict)
    best_recipe_hash: str | None = None
    screen_best: float | None = None
    confirmed_mean: float | None = None
    confirmed_sd: float | None = None
    n_confirm: int = 0
    recipes_done: int = 0
    recipes_open: int = 0
    recipes_planned: int = 0
    first_look_done: int = 0
    first_look_open: int = 0
    first_look_best: float | None = None
    second_chance_used: bool = False
    second_chance_done: int = 0
    second_chance_open: int = 0
    recipe_results: list = field(default_factory=list)
    confirm_open: int = 0
    confirm_vals: list = field(default_factory=list)
    refinements: int = 0
    refinements_without_gain: int = 0
    refinement_history: list = field(default_factory=list)
    pending_refinement: ChildInfo | None = None
    angle_index: int = 0
    censored: bool = False
    censored_extended: bool = False
    triage: dict = field(default_factory=dict)
    deep: DeepInfo = field(default_factory=DeepInfo)
    menu_flagged: bool = False
    last_diagnostics: dict = field(default_factory=dict)
    final_vals: list = field(default_factory=list)
    final_open: int = 0
    summary: str = ""
    tunable: dict = field(default_factory=dict)
    hparams: dict = field(default_factory=dict)
    suggested_recipe: dict = field(default_factory=dict)
    gate_failed_open: int = 0

    _nested = {"pending_refinement": ("one", ChildInfo), "deep": ("one", DeepInfo)}


@dataclass
class Snapshot(_Plain):
    """Everything ``Controller.decide`` reads. B1 builds it at the start of every ``ar2 step`` from the ledger.

    step: step counter (the controller's randomness is a pure function of (cfg.controller.seed, step, tag), so decisions replay exactly).
    phase: the phase B1 believes it is in (informative; the controller recomputes it with ``controller.current_phase``).
    D: training-set size.  budgets: {anchors, screen, procedures, main, final} from the config.
    spend: finished run-equivalents per phase {"A","S","P","M","D"}.  open_runs: planned-or-running runs per phase (each counted as 1.0 against the budget).
    spend_by_category: finished run-equivalents of phase M per category {"explore","refine","deep_tune"} (anchor/screen/procedure/final are phases, not categories).
    open_by_category: open runs of phase M per category.
    balance: the live balance settings (config ``balance`` section with any ``ar2 set`` overrides already applied).
    families: {name: FamilyState} for every family that has a candidate OR is in the taxonomy (empty families may be omitted; the controller adds them from ``taxonomy``).
    taxonomy: the family names of spec 4.4 (without other:<name>).  anchors: the three anchor families.
    pending_counts: {"first_look","tune","confirm","refine","deep_tune","final","screen"} open runs per kind;  runs_in_flight: submitted runs not finished;
    calls_in_flight: LLM calls running;  calls_in_flight_targets: [{mode, family, code_hash}] for those calls (so one family is not assigned twice);  sketches_unused: [{id, family, name, mechanism}] sketches not yet implemented;  re_since_sketch: run-equivalents spent since the
    last SKETCH call;  repair_queue: [{code_hash, family, gate_code, attempts}] gate failures awaiting REPAIR (attempts < max_repairs);
    new_candidates_unplanned: [{code_hash, family, origin: explore|refine|anchor, parent_code_hash, suggested_recipe, hparams, tunable}] candidates that passed gate and
    tagging but have no recipes yet (the controller returns TUNE actions for them).
    candidate_summaries: [{family, summary}] (for sketch novelty and the do-not-repeat count).
    recent_new_family_outcomes: outcomes of the last new families, oldest first: "parked" | "retired" | "tuned_only" | "confirmed" | "menu".
    recent_exploit_outcomes: accepted? (bool) of the last refinements and deep-tune attempts, oldest first (each entry {"kind": "refine"|"deep_tune", "accepted": bool}).
    recent_calls: {"EXPLORE": [...], "SKETCH": [...], "REFINE": [...]} last 10 calls per mode, each {assigned_family, landed_family, landed_family_size, first_attempt_gate_ok,
    duplicate, speed_fail, collapse}.  guard_state: opaque dict the controller wrote last time (B1 stores it in meta and returns it here).
    sigma_seed: measured pooled within-family SD of confirmation runs (None until 3 families are confirmed; prior used then).
    top_confirmed_mean: best confirmed mean over families (informative).  archive_hash: hash of the archive state (logged with each decision).
    recipe_defaults: {"loss": {...}, "ema": None|0.999, "screen_done": bool} adopted by the procedure screen.
    screen_results: [{"option","anchor","seed","val_r"}] finished screen runs; screen_open: open screen runs; screen_baseline: {"anchor": {"seed": val_r}} (confirm seeds 1, 2).
    screen_decision_done: the adoption NOTE was recorded;  procedures: {"nominated": [str], "implemented": int, "screened": int} for phase P.
    last_progress_spend: cumulative phase-M spend when a new family last became menu-eligible or a refinement/deep-tune was last accepted.
    cum_main_spend: cumulative phase-M spend (== spend["M"]).  total_calls_budget_left: optional hard cap on LLM calls (None = unlimited).
    stop_requested: the operator stopped the run.  final_menu_done: phase D has been fully scored.
    """

    step: int = 0
    phase: str = "A"
    D: int = 100_000
    budgets: dict = field(default_factory=dict)
    spend: dict = field(default_factory=dict)
    open_runs: dict = field(default_factory=dict)
    spend_by_category: dict = field(default_factory=dict)
    open_by_category: dict = field(default_factory=dict)
    balance: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
    taxonomy: list = field(default_factory=list)
    anchors: list = field(
        default_factory=lambda: ["cnn_residual", "rnn_hybrid", "transformer_hybrid"]
    )
    pending_counts: dict = field(default_factory=dict)
    runs_in_flight: int = 0
    calls_in_flight: int = 0
    calls_in_flight_targets: list = field(default_factory=list)
    sketches_unused: list = field(default_factory=list)
    re_since_sketch: float = 0.0
    repair_queue: list = field(default_factory=list)
    new_candidates_unplanned: list = field(default_factory=list)
    candidate_summaries: list = field(default_factory=list)
    recent_new_family_outcomes: list = field(default_factory=list)
    recent_exploit_outcomes: list = field(default_factory=list)
    recent_calls: dict = field(default_factory=dict)
    guard_state: dict = field(default_factory=dict)
    sigma_seed: float | None = None
    top_confirmed_mean: float | None = None
    archive_hash: str = ""
    recipe_defaults: dict = field(default_factory=dict)
    screen_results: list = field(default_factory=list)
    screen_open: int = 0
    screen_baseline: dict = field(default_factory=dict)
    screen_decision_done: bool = False
    procedures: dict = field(default_factory=dict)
    last_progress_spend: float = 0.0
    cum_main_spend: float = 0.0
    total_calls_budget_left: int | None = None
    stop_requested: bool = False
    final_menu_done: bool = False
    anchors_budget_exhausted: bool = False

    _nested = {"families": ("dict", FamilyState)}


@dataclass
class Action(_Plain):
    """One decision of the controller. B1 writes every Action (executable or not) to the ``decisions`` table and executes the executable ones.

    mode: one of MODES.  target: what it acts on - keys depend on the mode:
      SKETCH   {"forced_families": [...]|None}
      EXPLORE  {"family", "sketch": {...}|None, "anchor": bool, "source": "llm"|"reference", "extra_constraints": [...]}
      TUNE     {"code_hash", "family", "recipe_ids": [ordinals], "cap_multipliers": [float per recipe]}
      CONFIRM  {"code_hash", "family", "recipe_hash", "target": "family"|"child"|"deep", "cap_multiplier": float}
      REFINE   {"family", "parent_code_hash", "angle", "extra_constraints": [...]}
      REPAIR   {"code_hash", "family", "gate_code", "attempt"}
      TUNE_DEEP {"family", "code_hash", "batch": int}
      PROCEDURE {"step": "nominate"|"implement", "idea": str|None}
      SCREEN   {"kind": "run"|"adopt", "anchor", "option", "code_hash"} / adopt: {"adopted": [...], "effects": {...}}
      FINAL    {"family", "code_hash", "recipe_hash", "val_select": True}
      NOTE     {"kind": "guard"|"refinement_outcome"|"deep_outcome"|"family_state"|"phase"|"stop"|"menu"|"adaptation", ...}
    reason: a human-readable sentence (stored in ``decisions.reason``).
    recipes: full recipe dicts (spec 4.3) to create as pending runs (TUNE / CONFIRM / TUNE_DEEP / SCREEN / FINAL); the controller never emits more than the budget allows.
    seeds: seed per recipe (aligned with ``recipes``).  purpose: run purpose for the created runs.  category / phase: the budget line the runs are charged to.
    adaptation: 12.1 adjustments for the prompt of an LLM action: {"extra_constraints": [...], "forced_families": [...], "failure_view": bool, "extend_do_not_repeat": int|None,
    "rotate_reference": bool, "speed_note": bool} plus "signals": {name: value} that triggered them (logged).  executable: False for NOTE.
    expected_cost: run-equivalents this action is expected to cost (informative; used for balance projections).
    """

    mode: str = "NOTE"
    target: dict = field(default_factory=dict)
    reason: str = ""
    recipes: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    purpose: str = ""
    category: str = ""
    phase: str = ""
    adaptation: dict = field(default_factory=dict)
    executable: bool = True
    expected_cost: float = 0.0
