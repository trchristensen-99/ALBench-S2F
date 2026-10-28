"""Simulator for the controller (spec 16.2): a planted landscape, a proposer-free executor and the assertions of section 16.2.

The executor plays the role of the orchestrator + proposer + evaluator in memory: it holds a ledger (runs, candidates, recorded outcomes), builds a ``Snapshot`` from it
with the same ``build_family_state`` the orchestrator uses, calls ``Controller.decide``, applies the returned actions (creating pending runs, simulated LLM calls,
recording outcome notes) and advances a clock. A full search takes well under a second, so the controller can be proven on 20 seeds before any GPU time or LLM call is spent.

Landscape (hidden): every family has a base score and a hidden optimal learning rate; a run scores base + candidate quality - 0.02 * (log10(lr / lr_opt))**2 + N(0, 0.004).
A run whose lr is more than 1.6 decades from the optimum diverges. Planted cases: two families that beat the anchors by 0.015 but whose optimal lr is far from the default,
one family that always diverges, a claimant that is a duplicate of an anchor (filed under the anchor's family by the tagger), ten families within noise of the anchors or worse.
Landscape kinds: "planted" (the above), "flat" (every family equal), "never_new" (new families never succeed, refinements do), "never_refine" (new families succeed,
refinements never do).
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ar2 import tuner
from ar2.controller import (
    ANCHORS,
    TAXONOMY,
    Controller,
    build_family_state,
    family_view,
    forecast,
    menu_select,
    merged,
    top_confirmed,
)
from ar2.state import Action, FamilyState, RunInfo, Snapshot

PLANTED = ("state_space", "unet")
DIVERGER = "tcn_tied"
CLAIMANT = "other:dup_of_cnn_residual"
OTHERS = ("other:alpha", "other:beta", "other:gamma")
SCREEN_EFFECT = {
    "ema": 0.020,
    "corr": 0.018,
    "binned": -0.01,
}  # planted effects of the procedure-screen options (huber, hetero, multitask: none)


def _h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


@dataclass
class Landscape:
    kind: str
    seed: int
    base: dict = field(default_factory=dict)
    lr_opt: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(10_000 + self.seed)
        fam = list(TAXONOMY) + list(OTHERS)
        anchors = {
            "cnn_residual": (0.700, 2e-3),
            "rnn_hybrid": (0.690, 1e-3),
            "transformer_hybrid": (0.680, 5e-4),
        }
        for f in fam:
            if f in anchors:
                self.base[f], self.lr_opt[f] = anchors[f]
            else:
                self.base[f] = 0.0
                self.lr_opt[f] = 10 ** rng.uniform(-3.5, -2.3)
        if self.kind == "flat":
            for f in fam:
                self.base[f], self.lr_opt[f] = 0.70, 1e-3
            return
        others = [f for f in fam if f not in anchors and f not in PLANTED and f != DIVERGER]
        for i, f in enumerate(
            others
        ):  # ten families: half within noise of the anchors, half clearly worse
            self.base[f] = (
                0.69 + rng.uniform(-0.004, 0.006) if i % 2 == 0 else rng.uniform(0.60, 0.65)
            )
        self.base[DIVERGER] = 0.72
        if self.kind == "planted":
            self.base["state_space"], self.lr_opt["state_space"] = (
                0.715,
                8e-5,
            )  # far below the 5e-4 default of a low-lr family
            self.base["unet"], self.lr_opt["unet"] = 0.715, 9e-3  # far above the 2e-3 default
        elif self.kind == "never_new":
            for f in fam:
                if f not in anchors:
                    self.base[f] = rng.uniform(0.55, 0.62)
        elif self.kind == "never_refine":
            for f in fam:
                if f not in anchors and f != DIVERGER:
                    self.base[f] = 0.705 + rng.uniform(0.0, 0.01)
        else:  # "broad": good families are rare
            self.base["state_space"], self.lr_opt["state_space"] = 0.712, 3e-4
            self.base["unet"], self.lr_opt["unet"] = 0.712, 3e-3

    def refine_gain(self, rng: np.random.Generator, family: str, anchor: bool) -> float:
        if self.kind in ("never_refine", "flat"):
            return float(rng.normal(-0.004, 0.002))
        if self.kind == "never_new":
            return float(0.012 if rng.random() < 0.8 else -0.004)
        return float(0.012 if rng.random() < (0.5 if anchor else 0.3) else -0.004)


@dataclass
class SimResult:
    kind: str
    seed: int
    snapshot: Snapshot
    families: dict
    menu: dict
    spend: dict
    spend_by_category: dict
    stop_reason: str
    flat: bool
    steps: int
    phases: list
    guard_events: list
    counts: dict
    dup_families: list
    refine_views_ok: bool
    screen_adopted: list | None
    seconds: float


class SimExecutor:
    """The orchestrator, proposer and evaluator of a simulated search."""

    def __init__(
        self, cfg: dict | None, kind: str = "planted", seed: int = 0, *, steps: int = 600
    ) -> None:
        self.cfg = merged(cfg)
        self.cfg["controller"]["seed"] = int(self.cfg["controller"]["seed"]) + seed
        self.kind, self.seed, self.max_steps = kind, seed, steps
        self.ctrl = Controller(self.cfg)
        self.land = Landscape(kind, seed)
        self.rng = np.random.default_rng(seed)
        self.runs: dict[str, RunInfo] = {}
        self.order: list[str] = []
        self.cands: dict[str, dict] = {}
        self.outcomes: dict[str, list] = {}
        self.sketches: list[dict] = []
        self.calls: list[dict] = []  # in flight: {mode, family, code_hash, finish, payload}
        self.call_log: dict[str, list] = {"EXPLORE": [], "SKETCH": [], "REFINE": []}
        self.step = 0
        self.tick_n = 0
        self.re_since_sketch = 0.0
        self.new_family_outcomes: list[str] = []
        self.exploit_outcomes: list[dict] = []
        self.seen_state: dict[str, str] = {}
        self.eligible_seen: set[str] = set()
        self.last_top: float | None = None
        self.last_progress = 0.0
        self.defaults: dict = {}
        self.screen_done = False
        self.final_menu_done = False
        self.guard_events: list[dict] = []
        self.phases: list[str] = []
        self.decisions: list[Action] = []
        self.dup_families: list[str] = []
        self.refine_views_ok = True
        self.screen_adopted: list | None = None
        self.counts = {"explore": 0, "refine": 0, "deep_recipes": 0, "sketch_calls": 0}
        self.recorded_sketch_families: set[str] = set()
        self.stop_reason = ""
        self.flat = False
        self.queue: list[str] = []
        self._mut = 0
        self.progress_log: list = []
        self._fam_cache = None

    # ----------------------------------------------------------- ledger -> snapshot

    def _runs_of(self, fam: str) -> list[RunInfo]:
        return [self.runs[r] for r in self.order if self.runs[r].family == fam]

    def families(self) -> dict[str, FamilyState]:
        if self._fam_cache is not None and self._fam_cache[0] == self._mut:
            return self._fam_cache[1]
        out = self._build_families()
        self._fam_cache = (self._mut, out)
        return out

    def _build_families(self) -> dict[str, FamilyState]:
        out = {}
        for fam in sorted({c["family"] for c in self.cands.values()}):
            cs = [c for c in self.cands.values() if c["family"] == fam]
            out[fam] = build_family_state(
                fam,
                self._runs_of(fam),
                cs,
                self.outcomes.get(fam, []),
                self.cfg,
                anchor=fam in ANCHORS,
            )
        return out

    def snapshot(self) -> Snapshot:
        fams = self.families()
        top = top_confirmed(fams)
        pre = Controller(self.cfg)
        for f in fams.values():
            pass
        fams_full = pre.prepare(
            Snapshot(families=fams, taxonomy=list(TAXONOMY), anchors=list(ANCHORS))
        )
        spend = {p: 0.0 for p in "ASPMD"}
        openr = {p: 0 for p in "ASPMD"}
        by_cat = {c: 0.0 for c in ("explore", "refine", "deep_tune")}
        open_cat = {c: 0 for c in ("explore", "refine", "deep_tune")}
        inflight = 0
        screen_open = 0
        for r in self.runs.values():
            if r.finished:
                spend[r.phase] += r.cost_equiv
                if r.phase == "M" and r.category in by_cat:
                    by_cat[r.category] += r.cost_equiv
            else:
                openr[r.phase] += 1
                if r.phase == "M" and r.category in open_cat:
                    open_cat[r.category] += 1
                if r.status == "running":
                    inflight += 1
                if r.phase == "S":
                    screen_open += 1
        unplanned = [
            {
                "code_hash": ch,
                "family": c["family"],
                "origin": c["origin"],
                "parent_code_hash": c.get("parent_code_hash"),
                "suggested_recipe": c["suggested"],
                "hparams": {},
                "tunable": {},
            }
            for ch, c in self.cands.items()
            if c["gate_status"] == "passed"
            and not any(r.code_hash == ch for r in self.runs.values())
        ]
        repair_q = [
            {
                "code_hash": ch,
                "family": c["family"],
                "gate_code": "shape",
                "attempts": c["attempts"],
            }
            for ch, c in self.cands.items()
            if c["gate_status"] == "failed"
        ]
        screen_results, baseline = [], {}
        for r in self.runs.values():
            if r.purpose == "screen" and r.finished and r.val_r is not None:
                screen_results.append(
                    {
                        "option": r.diagnostics["option"],
                        "anchor": r.diagnostics["anchor"],
                        "seed": r.seed,
                        "val_r": r.val_r,
                    }
                )
        for a in ANCHORS:
            f = fams.get(a)
            if f and f.n_confirm >= 3:
                baseline[a] = {
                    r.seed: r.val_r
                    for r in self._runs_of(a)
                    if r.purpose == "confirm"
                    and r.code_hash == f.best_code_hash
                    and r.val_r is not None
                    and r.seed in (1, 2)
                }
        return (
            Snapshot(
                step=self.step,
                D=self.cfg["data"]["D"],
                budgets=dict(self.cfg["controller"]["budget"]),
                spend=spend,
                open_runs=openr,
                spend_by_category=by_cat,
                open_by_category=open_cat,
                balance=dict(self.cfg["balance"]),
                families=fams,
                taxonomy=list(TAXONOMY),
                anchors=list(ANCHORS),
                runs_in_flight=inflight,
                calls_in_flight=len(self.calls),
                calls_in_flight_targets=[
                    {"mode": c["mode"], "family": c.get("family"), "code_hash": c.get("code_hash")}
                    for c in self.calls
                ],
                sketches_unused=list(self.sketches),
                re_since_sketch=self.re_since_sketch,
                repair_queue=repair_q,
                new_candidates_unplanned=unplanned,
                candidate_summaries=[
                    {"family": c["family"], "summary": c["summary"]} for c in self.cands.values()
                ],
                recent_new_family_outcomes=list(self.new_family_outcomes),
                recent_exploit_outcomes=list(self.exploit_outcomes),
                recent_calls={m: list(v[-10:]) for m, v in self.call_log.items()},
                guard_state=self.ctrl.state_dict(),
                top_confirmed_mean=top,
                recipe_defaults=dict(self.defaults),
                screen_results=screen_results,
                screen_open=screen_open,
                screen_baseline=baseline,
                screen_decision_done=self.screen_done,
                procedures={"done": True},
                last_progress_spend=self.last_progress,
                cum_main_spend=spend["M"],
                final_menu_done=self.final_menu_done,
            )
            if fams_full is not None
            else None
        )

    # ----------------------------------------------------------- applying actions

    def _new_cand(
        self,
        family: str,
        origin: str,
        *,
        parent: str | None = None,
        quality: float = 0.0,
        claimed: str | None = None,
        angle: str = "",
    ) -> str:
        ch = f"c{len(self.cands):04d}{_h(self.seed, len(self.cands)) % 10**6:06d}"
        lr_opt = self.land.lr_opt.get(family, 1e-3)
        sug_lr = float(lr_opt * 10 ** self.rng.normal(0, 0.6))
        n = len([c for c in self.cands.values() if c["family"] == family])
        self.cands[ch] = {
            "code_hash": ch,
            "family": family,
            "claimed": claimed or family,
            "origin": origin,
            "parent_code_hash": parent,
            "angle": angle,
            "gate_status": "passed",
            "quality": quality,
            "attempts": 0,
            "suggested": {
                "lr": sug_lr,
                "weight_decay": 0.01,
                "batch_size": 256,
                "warmup_frac": 0.05,
            },
            "summary": f"{family} variant {n} " + " ".join(self._words(family, 5)),
        }
        return ch

    def _words(self, family: str, k: int) -> list[str]:
        r = np.random.default_rng(_h(family, self.rng.integers(10**6)))
        pool = [
            w
            for w in (
                "conv",
                "kernel",
                "attention",
                "gate",
                "mixer",
                "token",
                "state",
                "scan",
                "skip",
                "pool",
                "dilated",
                "recurrent",
                "residual",
                "motif",
                "equivariant",
                "spacing",
                "norm",
                "patch",
                "embed",
                "bilinear",
            )
        ]
        return [pool[i] for i in r.integers(len(pool), size=k)] + family.replace(":", "_").split(
            "_"
        )

    def _score(
        self, c: dict, recipe: dict, seed: int, purpose: str, option: str | None = None
    ) -> tuple[float | None, float | None, str, float]:
        fam = c["family"]
        if fam == DIVERGER:
            return None, None, "diverged", 0.15
        lr = recipe["lr"]
        gap = abs(math.log10(lr / self.land.lr_opt.get(fam, 1e-3)))
        if gap > 1.6:
            return None, None, "diverged", 0.15
        noise = np.random.default_rng(_h(c["code_hash"], tuner.recipe_hash(recipe), seed)).normal(
            0, 0.004
        )
        val = self.land.base[fam] + c["quality"] - 0.02 * gap**2 + noise
        if self.kind == "flat":
            val = (
                0.70 + noise
            )  # nothing matters here: no family, recipe or procedure option has an effect
        else:
            if recipe.get("ema"):
                val += SCREEN_EFFECT["ema"]
            name = (recipe.get("loss") or {}).get("name", "mse")
            val += SCREEN_EFFECT.get({"mse_corr": "corr"}.get(name, name), 0.0)
        cost = float(
            np.random.default_rng(
                _h("cost", c["code_hash"], tuner.recipe_hash(recipe), seed)
            ).uniform(0.55, 1.0)
        )
        return float(val), float(min(0.99, val + 0.10)), "ok", cost

    def _create_runs(self, a: Action) -> None:
        for i, rec in enumerate(a.recipes):
            seed = a.seeds[i] if i < len(a.seeds) else 0
            ch = a.target.get("code_hash")
            if a.mode == "SCREEN" and a.target.get("anchor"):
                ch = a.target["code_hash"]
            anchor = a.target.get("anchor") if a.mode == "SCREEN" else None
            if a.mode == "SCREEN" and a.target.get("anchors_per_recipe"):
                anchor = a.target["anchors_per_recipe"][i]
                ch = next(f.best_code_hash for f in self.families().values() if f.family == anchor)
            if a.mode == "FINAL":
                ch = a.target["code_hash"]
            rid = (
                "r_"
                + hashlib.sha1(
                    f"{ch}|{tuner.recipe_hash(rec)}|{seed}|{a.purpose}|{a.category}|{i}".encode()
                ).hexdigest()[:12]
            )
            if rid in self.runs:
                continue
            fam = self.cands[ch]["family"]
            opt = a.target["options"][i] if a.mode == "SCREEN" else None
            r = RunInfo(
                run_id=rid,
                code_hash=ch,
                family=fam,
                recipe_hash=tuner.recipe_hash(rec),
                recipe=copy.deepcopy(rec),
                schedule=rec["schedule"],
                seed=seed,
                purpose=a.purpose,
                category=a.category,
                phase=a.phase,
                status="pending",
                cap_multiplier=(a.target.get("cap_multipliers") or [1.0] * len(a.recipes))[i]
                if a.mode == "TUNE"
                else 1.0,
                diagnostics={"option": opt, "anchor": anchor or fam},
            )
            self.runs[rid] = r
            self.order.append(rid)
            self.queue.append(rid)

    def apply(self, acts: list[Action]) -> None:
        self._mut += 1
        for a in acts:
            self.decisions.append(a)
            if a.mode == "NOTE":
                self._note(a)
            elif a.mode in ("TUNE", "CONFIRM", "TUNE_DEEP", "SCREEN", "FINAL") and a.recipes:
                self._create_runs(a)
                if a.mode == "TUNE_DEEP":
                    self.counts["deep_recipes"] += len(a.recipes)
            elif a.mode == "SCREEN" and a.target.get("kind") == "adopt":
                self.defaults = dict(a.target["defaults"])
                self.screen_done = True
                self.screen_adopted = list(a.target["adopted"])
            elif a.mode == "EXPLORE" and a.target.get("anchor"):
                ch = self._new_cand(a.target["family"], "anchor")
                self.cands[ch]["suggested"]["lr"] = tuner.default_recipe(
                    a.target["family"], self.cfg
                )["lr"]
            elif a.mode == "EXPLORE":
                self.calls.append(
                    {
                        "mode": "EXPLORE",
                        "family": a.target["family"],
                        "finish": self.tick_n + 1,
                        "sketch": a.target["sketch"],
                        "assigned": a.target["family"],
                    }
                )
                self.counts["explore"] += 1
            elif a.mode == "SKETCH":
                self.calls.append(
                    {
                        "mode": "SKETCH",
                        "finish": self.tick_n + 1,
                        "forced": (a.target or {}).get("forced_families"),
                    }
                )
                self.counts["sketch_calls"] += 1
            elif a.mode == "REFINE":
                fam = a.target["family"]
                f = self.families()[fam]
                top = top_confirmed(self.families())
                try:
                    fv = family_view(f, top, 0.005)
                    ok = all(
                        k
                        in (
                            "family",
                            "confirmed_mean",
                            "confirmed_se",
                            "n_seeds",
                            "top_mean",
                            "earlier_changes",
                        )
                        for k in fv
                    )
                except ValueError:
                    ok = False
                self.refine_views_ok &= ok
                self.calls.append(
                    {
                        "mode": "REFINE",
                        "family": fam,
                        "finish": self.tick_n + 1,
                        "parent": a.target["parent_code_hash"],
                        "angle": a.target["angle"],
                    }
                )
                self.counts["refine"] += 1
            elif a.mode == "REPAIR":
                self.calls.append(
                    {
                        "mode": "REPAIR",
                        "family": a.target.get("family"),
                        "code_hash": a.target["code_hash"],
                        "finish": self.tick_n + 1,
                    }
                )
            elif a.mode == "PROCEDURE":
                pass

    def _note(self, a: Action) -> None:
        k = a.target.get("kind")
        if k == "refinement_outcome" or k == "deep_outcome":
            fam = a.target["family"]
            self.outcomes.setdefault(fam, []).append(dict(a.target))
            self.exploit_outcomes.append(
                {
                    "kind": "refine" if k == "refinement_outcome" else "deep_tune",
                    "accepted": bool(a.target["accepted"]),
                }
            )
            if a.target["accepted"]:
                self.last_progress = self.snapshot_spend_m()
                self.progress_log.append((round(self.last_progress, 1), k))
        elif k == "guard":
            self.guard_events.append(dict(a.target))
        elif k == "phase":
            self.phases.append(a.target["to"])
        elif k == "menu" and a.target.get("done"):
            self.final_menu_done = True
        elif k == "stop":
            pass

    def snapshot_spend_m(self) -> float:
        return sum(r.cost_equiv for r in self.runs.values() if r.finished and r.phase == "M")

    # ----------------------------------------------------------- clock

    def _finish_call(self, c: dict) -> None:
        m = c["mode"]
        if m == "SKETCH":
            fams_state = self.families()
            counts = {
                f: (fams_state[f].n_candidates if f in fams_state else 0)
                for f in list(TAXONOMY) + list(OTHERS)
            }
            order = sorted(counts, key=lambda f: (counts[f], self.rng.random()))
            forced = c.get("forced")
            pick = list(forced) if forced else order[:6]
            pick += [str(x) for x in self.rng.choice(list(counts), size=3, replace=False)]
            pick.append(CLAIMANT if self.rng.random() < 0.5 else str(self.rng.choice(list(counts))))
            out = []
            for i, f in enumerate(pick[:10]):
                self.recorded_sketch_families.add(f)
                out.append(
                    {
                        "id": f"s{self.tick_n}_{i}",
                        "family": f,
                        "name": f"{f} idea {i}",
                        "mechanism": " ".join(self._words(f, 6)),
                    }
                )
            self.sketches += out
            self.re_since_sketch = 0.0
            self.call_log["SKETCH"].append(
                {"assigned_family": None, "landed_family": None, "first_attempt_gate_ok": True}
            )
        elif m == "EXPLORE":
            sk = c["sketch"]
            if sk in self.sketches:
                self.sketches.remove(sk)
            claimed = sk["family"]
            fam = "cnn_residual" if claimed == CLAIMANT else claimed
            n_before = len([x for x in self.cands.values() if x["family"] == fam])
            ch = self._new_cand(fam, "explore", claimed=claimed)
            if claimed == CLAIMANT:
                self.dup_families.append(self.cands[ch]["family"])
            ok = self.rng.random() < 0.9
            if not ok:
                self.cands[ch]["gate_status"] = "failed"
            self.call_log["EXPLORE"].append(
                {
                    "assigned_family": claimed,
                    "landed_family": fam,
                    "landed_family_size": n_before,
                    "mechanism": sk.get("mechanism", ""),
                    "first_attempt_gate_ok": ok,
                    "duplicate": claimed == CLAIMANT,
                    "speed_fail": False,
                }
            )
        elif m == "REFINE":
            parent = self.cands[c["parent"]]
            gain = self.land.refine_gain(self.rng, parent["family"], parent["family"] in ANCHORS)
            ch = self._new_cand(
                parent["family"],
                "refine",
                parent=c["parent"],
                quality=parent["quality"] + gain,
                angle=c["angle"],
            )
            self.call_log["REFINE"].append(
                {
                    "assigned_family": parent["family"],
                    "landed_family": parent["family"],
                    "first_attempt_gate_ok": True,
                }
            )
        elif m == "REPAIR":
            cd = self.cands[c["code_hash"]]
            cd["attempts"] += 1
            if self.rng.random() < 0.7:
                cd["gate_status"] = "passed"
            elif cd["attempts"] >= 2:
                cd["gate_status"] = "rejected"

    def tick(self) -> None:
        self.tick_n += 1
        self._mut += 1
        # calls finish
        for c in [c for c in self.calls if c["finish"] <= self.tick_n]:
            self.calls.remove(c)
            self._finish_call(c)
        # running runs finish
        for r in [r for r in self.runs.values() if r.status == "running"]:
            c = self.cands[r.code_hash]
            val, tr, st, cost = self._score(
                c, r.recipe, r.seed, r.purpose, (r.diagnostics or {}).get("option")
            )
            r.status, r.val_r, r.train_r, r.cost_equiv = (
                st,
                val,
                tr,
                cost if r.purpose != "final" else 1.0,
            )
            r.epochs_run = int(r.max_epochs * cost)
            r.cap_multiplier = r.cap_multiplier
            if r.phase == "M" or r.phase == "A":
                self.re_since_sketch += r.cost_equiv
        # start pending runs
        running = sum(1 for r in self.runs.values() if r.status == "running")
        free = int(self.cfg["controller"]["max_parallel_jobs"]) - running
        while free > 0 and self.queue:
            rid = self.queue.pop(0)
            if self.runs[rid].status == "pending":
                self.runs[rid].status = "running"
                free -= 1
        self._track_progress()

    def _track_progress(self) -> None:
        fams = self.families()
        top = top_confirmed(fams)
        pre = Controller(self.cfg).prepare(
            Snapshot(families=fams, taxonomy=list(TAXONOMY), anchors=list(ANCHORS))
        )
        for n, f in pre.items():
            old = self.seen_state.get(n)
            if f.state != old:
                self.seen_state[n] = f.state
                if n not in ANCHORS and f.state in ("parked", "retired", "tuned_only", "confirmed"):
                    if not (f.state == "parked" and False):
                        self.new_family_outcomes.append(f.state)
        menu = menu_select(pre, self.cfg)
        sp = self.snapshot_spend_m()
        c = self.cfg["controller"]
        new = [n for n in menu["within_margin"] if n not in self.eligible_seen]
        for n in new:
            self.eligible_seen.add(n)
        # progress events: a new top beyond noise, or the menu is still being filled (see ar2/sim.py docstring: deviation from the literal spec text, documented)
        if top is not None:
            if self.last_top is None or top > self.last_top + c["noise_floor"]:
                self.last_top = top if self.last_top is None else top
                self.last_progress = sp
                self.progress_log.append((round(sp, 1), "new top"))
            elif new and len(menu["within_margin"]) <= c["menu_min_families"]:
                self.last_progress = sp
                self.progress_log.append((round(sp, 1), "menu filling"))

    # ----------------------------------------------------------- the loop

    def run(self) -> SimResult:
        t0 = time.time()
        for _ in range(self.max_steps):
            self.step += 1
            snap = self.snapshot()
            if self.ctrl.is_finished(snap):
                break
            mpj = int(self.cfg["controller"]["max_parallel_jobs"])
            fj = mpj - snap.runs_in_flight
            fc = int(self.cfg["controller"]["max_parallel_calls"]) - snap.calls_in_flight
            acts = self.ctrl.decide(snap, fj, fc)
            for a in acts:
                if a.mode == "NOTE" and a.target.get("kind") == "stop":
                    self.stop_reason, self.flat = a.reason, bool(a.target.get("flat"))
            self.apply(acts)
            self.tick()
        snap = self.snapshot()
        fams = self.ctrl.prepare(snap)
        stop, why, flat = self.ctrl.stop_check(snap, fams)
        self.stop_reason, self.flat = why, flat
        return SimResult(
            self.kind,
            self.seed,
            snap,
            fams,
            menu_select(fams, self.cfg),
            snap.spend,
            snap.spend_by_category,
            why,
            flat,
            self.step,
            self.phases,
            self.guard_events,
            dict(self.counts),
            self.dup_families,
            self.refine_views_ok,
            self.screen_adopted,
            time.time() - t0,
        )


def simulate(cfg: dict | None = None, kind: str = "planted", seed: int = 0, **kw) -> SimResult:
    return SimExecutor(cfg, kind, seed, **kw).run()


def sweep(cfg: dict | None, kind: str, seeds: range | list[int], **kw) -> list[SimResult]:
    return [simulate(cfg, kind, s, **kw) for s in seeds]


def budget_ok(res: SimResult, cfg: dict | None = None) -> bool:
    b = merged(cfg)["controller"]["budget"]
    tot = sum(res.spend.values()) + sum(res.snapshot.open_runs.values())
    return tot <= sum(b.values()) + 1e-6


def planted_found(res: SimResult) -> bool:
    return all(
        res.families[p].n_confirm >= 3 and p in res.menu["menu"]
        for p in PLANTED
        if p in res.families
    )


def explore_share(res: SimResult) -> float:
    s = res.spend_by_category
    tot = sum(s.values())
    return s["explore"] / tot if tot else 0.0


def forecast_error(results: list[SimResult], cfg: dict | None, e: float) -> dict:
    """Relative error of the plan(cfg) forecast against the simulated counts (new families explored, refinements, deep-tune recipes), mean over runs."""
    f = forecast(cfg, e)
    sim = {
        "new_families": float(np.mean([r.counts["explore"] for r in results])),
        "refinements": float(np.mean([r.counts["refine"] for r in results])),
        "deep_tune_recipes": float(np.mean([r.counts["deep_recipes"] for r in results])),
    }
    err = {k: abs(sim[k] - f[k]) / max(f[k], 1e-9) for k in sim}
    return {
        "forecast": {k: f[k] for k in sim},
        "simulated": sim,
        "relative_error": err,
        "mean_relative_error": float(np.mean(list(err.values()))),
    }
