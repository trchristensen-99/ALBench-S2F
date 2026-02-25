"""Tests for Exp5 utility helpers."""

from __future__ import annotations

from pathlib import Path

from experiments.exp5_best_student import _write_fasta


def test_write_fasta_outputs_records(tmp_path: Path) -> None:
    """FASTA writer should emit one header+sequence per input item."""
    out = tmp_path / "seqs.fasta"
    count = _write_fasta(out, ["ACGT", "TTAA"], prefix="x")
    assert count == 2
    text = out.read_text(encoding="utf-8")
    assert ">x_1" in text
    assert "ACGT" in text
    assert ">x_2" in text
    assert "TTAA" in text
