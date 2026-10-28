"""Deterministic mock proposer: fixtures instead of a language model (spec 4.7, used by every test).

It reads the rendered prompt to find out what is asked (the templates are frozen, so their assignment lines identify the mode) and answers from the library in
``mock_library``: ten sketches over 13 families, a valid model for any family, a refinement (a structural variant of ``parent_model.py``), a repair (the
parent with the planted defect removed), tags (from the AST rules) and lesson wording. It can also plant failures on request so that gatekeeper, controller and
orchestrator tests have something to catch:

    MockProposer(planted={3: "wrong_shape", "my sketch name": "duplicate"}, fail_calls={7}, paused_calls={9: 120.0})

``planted`` keys are the 1-based index among candidate calls (EXPLORE and REFINE) or a sketch name; kinds are listed in ``mock_library.PLANTED_KINDS``.
``fail_calls`` makes the n-th call of any kind return ``ok=False``; ``paused_calls`` raises ``ProposerPaused`` on the n-th call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ar2.proposer import mock_library as lib
from ar2.proposer.base import ProposerPaused, ProposerResult, validate_card
from ar2.tagger import rule_check

SKETCHES: tuple[dict[str, str], ...] = (
    {
        "family": "cnn_dilated",
        "name": "Growing-dilation residual tower",
        "mechanism": "Residual blocks whose dilation doubles per block, so the receptive field covers the insert without pooling.",
        "why_for_mpra": "Spacing between binding sites several tens of nt apart is resolved without losing position information.",
        "nearest_existing": "cnn_residual reference",
        "difference": "Dilation instead of pooling; no squeeze-and-excite.",
        "recipe_needs": "Standard one-cycle schedule, moderate weight decay.",
    },
    {
        "family": "state_space",
        "name": "Bidirectional diagonal SSM",
        "mechanism": "Stacked diagonal state-space layers run as long convolutions in both directions.",
        "why_for_mpra": "A learned decay profile models how far a motif's influence reaches along the insert.",
        "nearest_existing": "none",
        "difference": "No attention or recurrence; global context from a parametric kernel.",
        "recipe_needs": "Lower learning rate, no weight decay on the state parameters.",
    },
    {
        "family": "unet",
        "name": "Encoder-decoder with skips",
        "mechanism": "Three-level encoder, nearest-neighbour decoder, skips concatenated before pooling.",
        "why_for_mpra": "Multi-resolution features combine single motifs and their larger arrangement.",
        "nearest_existing": "cnn_plain",
        "difference": "Skip connections across resolutions.",
        "recipe_needs": "Standard.",
    },
    {
        "family": "motif_grammar",
        "name": "Motif scanner plus spacing module",
        "mechanism": "Orientation-shared PWM scan followed by a pairwise co-occurrence module over smoothed motif activity.",
        "why_for_mpra": "Activity depends on motif counts and their spacing.",
        "nearest_existing": "none",
        "difference": "Explicit interaction layer between motif channels.",
        "recipe_needs": "Higher weight decay on the scanner filters.",
    },
    {
        "family": "mixer_mlp",
        "name": "Patch MLP-mixer",
        "mechanism": "4 nt patches mixed along the token axis then the channel axis by MLPs.",
        "why_for_mpra": "Position-specific interactions without convolution inductive bias.",
        "nearest_existing": "none",
        "difference": "No convolution tower.",
        "recipe_needs": "Lower learning rate and warm-up.",
    },
    {
        "family": "transformer_pure",
        "name": "k-mer patch transformer",
        "mechanism": "8 nt patches embedded linearly, attention blocks with learned positions.",
        "why_for_mpra": "Order and spacing of binding sites through attention.",
        "nearest_existing": "transformer_hybrid reference",
        "difference": "No convolutional stem.",
        "recipe_needs": "Warm-up 10%, lr 5e-4.",
    },
    {
        "family": "cnn_multiscale",
        "name": "Inception-style multi-kernel blocks",
        "mechanism": "Parallel 3, 7 and 15 nt branches concatenated in each block.",
        "why_for_mpra": "Binding sites differ in length.",
        "nearest_existing": "cnn_plain",
        "difference": "Parallel kernel sizes.",
        "recipe_needs": "Standard.",
    },
    {
        "family": "tcn_tied",
        "name": "Weight-tied dilated block",
        "mechanism": "One dilated residual block applied four times with shared weights.",
        "why_for_mpra": "Deep integration with few parameters limits overfitting at small D.",
        "nearest_existing": "cnn_dilated",
        "difference": "Shared weights.",
        "recipe_needs": "Standard.",
    },
    {
        "family": "equivariant",
        "name": "RC-equivariant convolutions",
        "mechanism": "Filter banks tied to their reverse complements.",
        "why_for_mpra": "Binding sites act on either strand.",
        "nearest_existing": "cnn_residual",
        "difference": "Strand symmetry built into the layers.",
        "recipe_needs": "Standard.",
    },
    {
        "family": "rnn_hybrid",
        "name": "Conv stem with bidirectional GRU",
        "mechanism": "Two-layer convolution stem, pooled, then a bidirectional GRU and max/mean readout.",
        "why_for_mpra": "Order-dependent integration of motif hits.",
        "nearest_existing": "cnn_residual",
        "difference": "Recurrent integration.",
        "recipe_needs": "Gradient clipping 1.0.",
    },
    {
        "family": "transformer_hybrid",
        "name": "Conv stem with attention blocks",
        "mechanism": "Motif-resolving stem then three pre-norm attention blocks.",
        "why_for_mpra": "Long-range spacing handled by attention over motif features.",
        "nearest_existing": "cnn_residual",
        "difference": "Attention holds most of the depth.",
        "recipe_needs": "Warm-up 10%, lr 5e-4.",
    },
    {
        "family": "cnn_residual",
        "name": "Squeeze-excite residual tower",
        "mechanism": "LegNet-like blocks with squeeze-and-excite.",
        "why_for_mpra": "Reference-style model.",
        "nearest_existing": "cnn_residual reference",
        "difference": "none",
        "recipe_needs": "Standard.",
    },
    {
        "family": "cnn_plain",
        "name": "Plain conv-pool stack",
        "mechanism": "Three conv-pool stages and a linear head.",
        "why_for_mpra": "Baseline.",
        "nearest_existing": "none",
        "difference": "No residual paths.",
        "recipe_needs": "Standard.",
    },
)
_ATTN = {"transformer_hybrid", "transformer_pure", "state_space", "mixer_mlp"}


def _recipe_for(family: str) -> dict[str, Any]:
    if family in _ATTN:
        return {"lr": 5e-4, "weight_decay": 0.05, "warmup_frac": 0.1, "batch_size": 256}
    return {"lr": 2e-3, "weight_decay": 0.01, "warmup_frac": 0.05, "batch_size": 512}


def _mode(prompt: str) -> str:
    if "propose 10 candidate architectures" in prompt:
        return "SKETCH"
    if "ASSIGNMENT: implement this one architecture." in prompt:
        return "EXPLORE"
    if "make ONE structural change" in prompt:
        return "REFINE"
    if "failed an automatic check" in prompt:
        return "REPAIR"
    if "Classify the PyTorch model below" in prompt:
        return "TAGGER"
    if "Write one sentence that states this finding" in prompt:
        return "LESSON"
    return "UNKNOWN"


def strip_planted(source: str) -> str:
    """Remove the defects ``mock_library.broken_source`` plants (what a good repair would do)."""
    cut = source.find("\n\n_orig_build_model = build_model")
    if cut >= 0:
        source = source[:cut] + "\n"
    source = source.replace("import os\n", "", 1) if source.startswith("import os\n") else source
    bad = source.find("\ndef broken(:")
    return source[:bad] + "\n" if bad >= 0 else source


class MockProposer:
    def __init__(
        self,
        seed: int = 0,
        planted: dict[Any, str] | None = None,
        fail_calls: set[int] | None = None,
        paused_calls: dict[int, float] | None = None,
        tag_override: str | None = None,
        model_id: str = "mock-model-1",
    ) -> None:
        self.seed = seed
        self.planted = dict(planted or {})
        self.fail_calls = set(fail_calls or ())
        self.paused_calls = dict(paused_calls or {})
        self.tag_override = tag_override
        self.model_id = model_id
        self.calls = 0
        self.candidate_calls = 0
        self.sketch_calls = 0
        self.issued: dict[str, int] = {}
        self.first_source: str | None = None
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------------- API

    def call(self, prompt: str, workdir: Path, expect: str, timeout_s: int = 900) -> ProposerResult:
        self.calls += 1
        n = self.calls
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if n in self.paused_calls:
            raise ProposerPaused(f"mock rate limit at call {n}", self.paused_calls[n])
        mode = _mode(prompt)
        self.history.append({"n": n, "mode": mode, "expect": expect})
        if n in self.fail_calls:
            return self._result(prompt, ok=False, error=f"mock failure at call {n}", exit_code=1)
        if mode == "SKETCH":
            return self._sketch(prompt, workdir)
        if mode in ("EXPLORE", "REFINE", "REPAIR"):
            return self._candidate(prompt, workdir, mode)
        if mode == "TAGGER":
            return self._tags(prompt)
        if mode == "LESSON":
            return self._lesson(prompt)
        return self._result(prompt, ok=False, error="mock: unrecognised prompt", exit_code=2)

    # ------------------------------------------------------------------------------------------- pieces

    def _result(self, prompt: str, ok: bool = True, **kw: Any) -> ProposerResult:
        return ProposerResult(
            ok=ok,
            model_id=self.model_id,
            cli_version="mock-0",
            usage={"input_tokens": len(prompt) // 4, "output_tokens": 400},
            **kw,
        )

    def _sketch(self, prompt: str, workdir: Path) -> ProposerResult:
        k = self.sketch_calls
        self.sketch_calls += 1
        rot = (3 * k) % len(SKETCHES)
        pick = [dict(SKETCHES[(rot + i) % len(SKETCHES)]) for i in range(10)]
        for i, s in enumerate(pick):
            s["name"] = s["name"] + (f" (round {k})" if k else "")
        text = json.dumps(pick, indent=1)
        (workdir / "sketches.json").write_text(text)
        return self._result(
            prompt, files={"sketches.json": text}, parsed=pick, raw_stdout="wrote sketches.json"
        )

    def _issue(self, family: str) -> int:
        k = self.issued.get(family, 0)
        self.issued[family] = k + 1
        return k

    def _candidate(self, prompt: str, workdir: Path, mode: str) -> ProposerResult:
        if mode != "REPAIR":
            self.candidate_calls += 1
        kind = None
        sketch: dict[str, Any] = {}
        if mode == "EXPLORE":
            m = re.search(
                r"implement this one architecture\.\n(\{.*?\n\})\n- Implement", prompt, re.S
            )
            if m:
                try:
                    sketch = json.loads(m.group(1))
                except ValueError:
                    sketch = {}
            if not sketch:
                fam = next(
                    (f for f in lib.FAMILIES if f in prompt.split("ASSIGNMENT:")[-1]),
                    "cnn_residual",
                )
                sketch = {"family": fam, "name": fam, "mechanism": "", "why_for_mpra": ""}
            kind = self.planted.get(self.candidate_calls) or self.planted.get(
                sketch.get("name", "")
            )
        elif mode == "REFINE":
            kind = self.planted.get(self.candidate_calls)
        if mode == "EXPLORE":
            fam = sketch.get("family", "cnn_residual")
            fam = fam if fam in lib.FAMILIES else "cnn_residual"
            src = lib.library_variant(fam, self._issue(fam))
            card = {
                "family_claimed": sketch.get("family", fam),
                "summary": f"{sketch.get('name', fam)}: {str(sketch.get('mechanism', ''))[:90]}",
                "hypothesis": sketch.get("why_for_mpra")
                or "The mechanism suits motif-and-spacing data.",
                "what_changed": "",
                "parent_code_hash": "",
                "suggested_recipe": _recipe_for(fam),
                "recipe_rationale": "Attention and state-space models need warm-up and a lower learning rate; CNNs tolerate the default."
                if fam in _ATTN
                else "Default CNN recipe.",
                "expected_failure": "May overfit at small D without enough dropout.",
                "amp_safe": True,
                "length_agnostic": False,
            }
        else:
            parent = (
                (workdir / "parent_model.py").read_text()
                if (workdir / "parent_model.py").exists()
                else lib.library_source("cnn_residual")
            )
            if mode == "REPAIR":
                src = strip_planted(parent)
                card = {
                    "family_claimed": "unknown",
                    "summary": "repaired",
                    "hypothesis": "unchanged",
                }
            else:
                src = lib.make_variant(strip_planted(parent), 1 + self.candidate_calls % 5)
                card = {
                    "family_claimed": "unknown",
                    "summary": "refined child",
                    "hypothesis": "Widening the model raises validation r by more than the seed noise.",
                    "what_changed": "Widened the layers (mock structural change).",
                    "parent_code_hash": "",
                    "suggested_recipe": {},
                    "recipe_rationale": "",
                    "expected_failure": "No gain at this data size.",
                    "amp_safe": True,
                    "length_agnostic": False,
                }
        files: dict[str, str] = {}
        if kind == "collapse":
            src = lib.library_variant("cnn_residual", self._issue("cnn_residual"))
        elif kind == "duplicate":
            src = self.first_source or src
        elif kind in lib.PLANTED_KINDS and kind not in (
            "missing_card",
            "invalid_card",
            "collapse",
            "duplicate",
        ):
            src = lib.broken_source(kind, src)
        if self.first_source is None:
            self.first_source = src
        (workdir / "model.py").write_text(src)
        files["model.py"] = src
        if kind == "missing_card":
            return self._result(
                prompt,
                ok=False,
                files=files,
                error="card.json missing",
                missing=["card.json"],
                raw_stdout="wrote model.py",
            )
        card_text = "{not valid json" if kind == "invalid_card" else json.dumps(card, indent=1)
        (workdir / "card.json").write_text(card_text)
        files["card.json"] = card_text
        if kind == "invalid_card":
            return self._result(
                prompt,
                ok=False,
                files=files,
                error="card.json is not valid JSON",
                missing=["card.json (valid JSON)"],
            )
        errs = validate_card(card)
        if errs:
            return self._result(
                prompt, ok=False, files=files, parsed=card, error="; ".join(errs), missing=errs
            )
        return self._result(
            prompt, files=files, parsed=card, raw_stdout="wrote model.py and card.json"
        )

    def _tags(self, prompt: str) -> ProposerResult:
        code = prompt.split("CODE:\n", 1)[-1]
        tags = rule_check(code, None)
        out = {
            k: tags[k]
            for k in (
                "family",
                "core_block",
                "uses_attention",
                "uses_recurrence",
                "normalisation",
                "activation",
                "pooling",
                "positional_info",
                "downsampling_factor",
                "justification",
            )
        }
        if self.tag_override:
            out["family"] = self.tag_override
        return self._result(prompt, parsed=out, raw_stdout=json.dumps(out))

    def _lesson(self, prompt: str) -> ProposerResult:
        m = re.search(r"RECORD: (\{.*\})", prompt)
        ev = json.loads(m.group(1)) if m else {}
        scope = ev.get("scope", "global")
        change = ev.get("change") or ev.get("subject") or "the change"
        direction = ev.get("direction", "no clear effect")
        verdict = "was accepted" if ev.get("accepted") else "was not accepted"
        out = {"statement": f"{change}: {direction}; it {verdict} ({scope} scope)."}
        return self._result(prompt, parsed=out, raw_stdout=json.dumps(out))
