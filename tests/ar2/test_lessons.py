"""Lessons: numbers come from code (T11), wording from one call, scope, retraction, source rules (spec 10.3)."""

from __future__ import annotations

import json
import math

import pytest

from ar2 import lessons, views
from ar2.proposer.mock import MockProposer

RUNS = {
    "p1": 0.730,
    "p2": 0.733,
    "p3": 0.728,  # parent, three confirmation seeds
    "c1": 0.738,
    "c2": 0.741,
    "c3": 0.737,  # child
    "o1": 0.742,
    "o2": 0.740,
    "o3": 0.735,
    "o4": 0.739,
    "o5": 0.741,
    "o6": 0.736,
    "b1": 0.738,
    "b2": 0.739,
    "b3": 0.733,
    "b4": 0.738,
    "b5": 0.738,
    "b6": 0.735,
    "d1": 1.0,
    "d2": 0.0,
    "d3": 0.0,
    "d4": 0.0,
    "d5": 0.0,
    "d6": 0.0,
    "d7": 1.0,
    "d8": 0.0,
}
REFINE = {"parent": ["p1", "p2", "p3"], "child": ["c1", "c2", "c3"]}


def refinement(**kw):
    return lessons.make_evidence(
        "refinement",
        scope="transformer_hybrid",
        subject="attention pooling in place of mean pooling",
        runs=RUNS,
        groups=REFINE,
        **kw,
    )


def test_refinement_numbers_are_computed_from_the_cited_runs():
    ev = refinement()
    p, c = [RUNS[i] for i in REFINE["parent"]], [RUNS[i] for i in REFINE["child"]]
    mp, mc = sum(p) / 3, sum(c) / 3
    vp = sum((x - mp) ** 2 for x in p) / 2
    vc = sum((x - mc) ** 2 for x in c) / 2
    assert (
        ev["effect"] == pytest.approx(mc - mp)
        and ev["se"] == pytest.approx(math.sqrt(vp / 3 + vc / 3))
        and ev["n"] == 6
    )
    assert (
        ev["n_desc"] == "3 vs 3 seeds" and ev["direction"] == "improved" and ev["accepted"] is True
    )


def test_t11_numbers_in_every_stored_lesson_equal_values_recomputed_from_cited_runs(tmp_path):
    rows = []
    for ev in (
        refinement(),
        lessons.make_evidence(
            "procedure",
            scope="global",
            subject="Huber loss",
            runs=RUNS,
            groups={
                "option": [f"o{i}" for i in range(1, 7)],
                "baseline": [f"b{i}" for i in range(1, 7)],
            },
        ),
        lessons.make_evidence(
            "aggregate",
            scope="global",
            subject="Runs that diverged",
            runs=RUNS,
            groups={"runs": [f"d{i}" for i in range(1, 9)]},
        ),
    ):
        proposer = MockProposer()
        statement = lessons.word(proposer, ev, {}, tmp_path / "w")
        rows.append(lessons.to_row(ev, statement, created_at_cost=10.0))
    for row in rows:
        again = lessons.recompute(row, RUNS)
        assert (
            row["effect"] == pytest.approx(again["effect"])
            and row["se"] == pytest.approx(again["se"])
            and row["n"] == again["n"]
        )
        nums = lessons.numbers_in_statement(row["statement"])
        kind = json.loads(row["evidence_run_ids"])["kind"]
        if kind == "aggregate":
            assert nums[0] == round(again["effect"] * 100)
        else:
            assert float(f"{again['effect']:+.3f}") in nums and float(f"{again['se']:.3f}") in nums
    assert rows[0]["statement"].startswith("[transformer_hybrid]") and rows[0][
        "statement"
    ].endswith("Accepted.")
    assert "(3 vs 3 seeds)" in rows[0]["statement"] and rows[1]["statement"].endswith(
        ("Adopted.", "Not adopted.")
    )


def test_llm_sees_no_numbers_and_cannot_add_them(tmp_path):
    ev = refinement()
    rec = lessons.llm_record(ev)
    assert (
        not any(isinstance(v, float) for v in rec.values())
        and "effect" not in rec
        and "se" not in rec
        and "n" not in rec
    )
    prompt = views.render(
        "LESSON", views.ViewContext(), {"evidence": ev}
    )  # even the full record is filtered by the view
    assert "effect" not in prompt and "0.00" not in prompt and "evidence_run_ids" not in prompt

    class Chatty:
        def call(self, prompt, workdir, expect, timeout_s):
            from ar2.proposer.base import ProposerResult

            return ProposerResult(
                ok=True, parsed={"statement": "Attention pooling lifted r by 0.02 in 9 runs"}
            )

    out = lessons.word(Chatty(), ev, {}, tmp_path)
    assert "0.02" not in out and "9 runs" not in out  # the sentence with numbers was discarded
    assert (
        out.startswith(
            "[transformer_hybrid] Attention pooling in place of mean pooling raised validation r:"
        )
        and "+0.0" in out
    )

    class Failing:
        def call(self, *a, **k):
            raise RuntimeError("down")

    assert lessons.word(Failing(), ev, {}, tmp_path).startswith("[transformer_hybrid]")


def test_unconfirmed_evidence_is_refused():
    runs = dict(RUNS)
    with pytest.raises(ValueError, match="3 confirmation seeds"):
        lessons.make_evidence(
            "refinement",
            scope="x",
            subject="s",
            runs=runs,
            groups={"parent": ["p1", "p2"], "child": ["c1", "c2", "c3"]},
        )
    with pytest.raises(ValueError, match="at least 5 runs"):
        lessons.make_evidence(
            "aggregate", scope="global", subject="s", runs=runs, groups={"runs": ["d1", "d2", "d3"]}
        )
    with pytest.raises(ValueError, match="paired"):
        lessons.make_evidence(
            "procedure",
            scope="global",
            subject="s",
            runs=runs,
            groups={"option": ["o1", "o2"], "baseline": ["b1", "b2"]},
        )
    with pytest.raises(ValueError):
        lessons.make_evidence("hunch", scope="global", subject="s", runs=runs, groups={})


def test_negative_results_are_recorded_with_their_interval():
    runs = {"p1": 0.730, "p2": 0.740, "p3": 0.735, "c1": 0.731, "c2": 0.741, "c3": 0.733}
    ev = lessons.make_evidence(
        "refinement", scope="unet", subject="Skip projection", runs=runs, groups=REFINE | {}
    )
    assert ev["direction"] == "no clear effect" and ev["accepted"] is False
    st = lessons.format_statement(ev, "Skip projection made no clear difference")
    assert (
        st.startswith("[unet] Skip projection made no clear difference: ")
        and "+/-" in st
        and st.endswith("Rejected.")
    )
    ni = lessons.make_evidence(
        "procedure",
        scope="global",
        subject="Huber loss",
        runs=RUNS,
        groups={
            "option": [f"o{i}" for i in range(1, 7)],
            "baseline": [f"b{i}" for i in range(1, 7)],
        },
        status="not_testable_in_silico",
    )
    assert lessons.format_statement(ni, "Huber loss").endswith(
        "Not testable in silico (noise-free labels)."
    )


def test_scope_is_kept_and_selection_puts_the_family_first():
    rows = [
        lessons.to_row(
            refinement(), "[transformer_hybrid] a: +0.007 +/- 0.003 (3 vs 3 seeds). Accepted.", 5.0
        ),
        {
            "lesson_id": "g1",
            "scope": "global",
            "statement": "[global] g1",
            "status": "confirmed",
            "created_at_cost": 9.0,
        },
        {
            "lesson_id": "g2",
            "scope": "global",
            "statement": "[global] g2",
            "status": "confirmed",
            "created_at_cost": 1.0,
        },
        {
            "lesson_id": "u1",
            "scope": "unet",
            "statement": "[unet] u1",
            "status": "confirmed",
            "created_at_cost": 8.0,
        },
        {
            "lesson_id": "r1",
            "scope": "global",
            "statement": "[global] retracted",
            "status": "retracted",
            "created_at_cost": 10.0,
        },
    ]
    fam, glob = lessons.select_for_prompt(rows, "transformer_hybrid")
    assert fam == [rows[0]["statement"]] and glob == [
        "[global] g1",
        "[global] g2",
    ]  # newest first, retracted dropped, unet lesson never shown
    fam2, glob2 = lessons.select_for_prompt(rows, "transformer_hybrid", limit=2)
    assert len(fam2) + len(glob2) == 2
    assert lessons.select_for_prompt(rows, None)[0] == []


def test_retraction_needs_a_reversed_sign_and_an_interval_excluding_the_original():
    old = lessons.to_row(refinement(), "old")
    worse = {"p1": 0.74, "p2": 0.741, "p3": 0.739, "c1": 0.73, "c2": 0.729, "c3": 0.731}
    ev_neg = lessons.make_evidence(
        "refinement",
        scope="transformer_hybrid",
        subject="attention pooling in place of mean pooling",
        runs=worse,
        groups=REFINE,
    )
    new = lessons.to_row(ev_neg, "new")
    assert lessons.should_retract(old, new)
    noisy = {
        "p1": 0.73,
        "p2": 0.76,
        "p3": 0.71,
        "c1": 0.72,
        "c2": 0.75,
        "c3": 0.725,
    }  # reversed sign but a wide interval that still includes the old effect
    ev_noisy = lessons.make_evidence(
        "refinement",
        scope="transformer_hybrid",
        subject="attention pooling in place of mean pooling",
        runs=noisy,
        groups=REFINE,
    )
    assert ev_noisy["effect"] < 0 and not lessons.should_retract(old, lessons.to_row(ev_noisy, "n"))
    other_subject = lessons.to_row(
        lessons.make_evidence(
            "refinement",
            scope="transformer_hybrid",
            subject="something else",
            runs=worse,
            groups=REFINE,
        ),
        "o",
    )
    assert not lessons.should_retract(old, other_subject)
    other_scope = lessons.to_row(
        lessons.make_evidence(
            "refinement",
            scope="unet",
            subject="attention pooling in place of mean pooling",
            runs=worse,
            groups=REFINE,
        ),
        "o",
    )
    assert not lessons.should_retract(old, other_scope)
    assert not lessons.should_retract({**old, "status": "retracted"}, new)


def test_lesson_id_is_stable_and_distinct():
    a, b = refinement(), refinement()
    assert lessons.lesson_id(a) == lessons.lesson_id(b)
    other = lessons.make_evidence(
        "refinement",
        scope="unet",
        subject="attention pooling in place of mean pooling",
        runs=RUNS,
        groups=REFINE,
    )
    assert lessons.lesson_id(a) != lessons.lesson_id(other)
