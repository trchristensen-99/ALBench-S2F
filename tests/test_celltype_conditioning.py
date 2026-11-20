"""The conditioning wrapper has to be inert at init and alive after training.

Both halves matter. Inert at init, because FiLM is attached to a backbone that is already
the study's reference student: if the wrapper perturbed it before learning anything, every
conditioning run would start behind the unconditioned control and the comparison would
measure the wrapper rather than the scheme. Alive after, because a conditioning scheme that
silently ignores the cell is exactly the failure mode the controls exist to catch -- it
returns a full, plausible prediction with nothing in the output to say the cell was never
read.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from albench.celltype import conditioning as C  # noqa: E402
from albench.celltype import data as D  # noqa: E402
from albench.celltype import descriptions as DESC  # noqa: E402

CELLS = ["K562", "HepG2", "SK-N-SH"]


class Tiny(torch.nn.Module):
    """A backbone with the two features the wrapper hooks: BatchNorm1d and a Linear(*, 1)."""

    def __init__(self, ch: int = 8):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, ch, 5, padding=2)
        self.bn = torch.nn.BatchNorm1d(ch)
        self.head = torch.nn.Linear(ch, 1)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        return self.head(h.mean(-1)).reshape(-1)


def _x(n: int = 6, length: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 4, (n, length), generator=g)
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()


@pytest.mark.parametrize("mode", C.CONDITIONING_MODES)
@pytest.mark.parametrize("depth", C.DEPTHS)
def test_every_mode_builds_and_is_inert_at_init(mode, depth):
    torch.manual_seed(0)
    bare = Tiny().eval()
    x = _x()
    want = bare(x).detach()

    model = C.wrap(bare, CELLS, mode, depth=depth, d_cond=8).eval()
    ci = torch.tensor([0, 1, 2, 0, 1, 2])
    got = model(x, cell_idx=ci).detach()
    # FiLM is zero-initialised, so gain 1 and bias 0: the wrapper must reproduce the
    # backbone exactly until something trains it.
    assert torch.allclose(want, got, atol=1e-5), f"{mode}/{depth} perturbed the backbone"


@pytest.mark.parametrize("mode", ["onehot", "embed", "text", "structured", "hybrid"])
def test_conditioning_changes_the_prediction_once_film_is_nonzero(mode):
    torch.manual_seed(0)
    model = C.wrap(Tiny(), CELLS, mode, depth="deep", d_cond=8).eval()
    for film in model.films:
        torch.nn.init.normal_(film.to_gb.weight, std=0.5)
    if mode == "embed":
        torch.nn.init.normal_(model.encoder.free.weight, std=1.0)
    x = _x(3)
    a = model(x, cell_idx=torch.tensor([0, 0, 0])).detach()
    b = model(x, cell_idx=torch.tensor([1, 1, 1])).detach()
    assert not torch.allclose(a, b, atol=1e-4), f"{mode} ignores the cell index"


def test_agnostic_ignores_the_cell_entirely():
    model = C.wrap(Tiny(), CELLS, "agnostic", depth="deep").eval()
    for film in model.films:  # there are none, but be explicit about why
        torch.nn.init.normal_(film.to_gb.weight, std=0.5)
    x = _x(3)
    a = model(x, cell_idx=torch.tensor([0, 0, 0])).detach()
    b = model(x, cell_idx=torch.tensor([2, 2, 2])).detach()
    assert torch.allclose(a, b), "the cell-agnostic control is reading the cell"


def test_only_describable_modes_reach_an_unseen_cell():
    src = DESC.text_embeddings(CELLS)
    for mode in ("onehot", "embed"):
        m = C.wrap(Tiny(), CELLS, mode, d_cond=8)
        with pytest.raises(NotImplementedError):
            m.encoder.encode_novel(src[0])
    for mode in ("text", "hybrid"):
        m = C.wrap(Tiny(), CELLS, mode, d_cond=8)
        z = m.encoder.encode_novel(DESC.text_embeddings(["A549"])[0])
        assert z.shape == (1, 8)


def test_text_shuffled_really_permutes_and_derangement_holds():
    plain = C.wrap(Tiny(), CELLS, "text", d_cond=8)
    shuf = C.wrap(Tiny(), CELLS, "text_shuffled", d_cond=8, seed=3)
    assert hasattr(shuf.encoder, "perm")
    perm = shuf.encoder.perm.numpy()
    assert not (perm == np.arange(len(CELLS))).any(), "a cell kept its own description"
    assert not torch.allclose(plain.encoder.table, shuf.encoder.table)


def test_hybrid_residual_starts_at_zero_and_is_penalised():
    m = C.wrap(Tiny(), CELLS, "hybrid", d_cond=8)
    assert float(m.encoder.residual_penalty()) == 0.0
    torch.nn.init.normal_(m.encoder.free.weight, std=1.0)
    assert float(m.encoder.residual_penalty()) > 0.1


def test_novel_cell_vector_differs_between_cells():
    """If two descriptions gave the same vector, zero-shot transfer would be a constant."""
    m = C.wrap(Tiny(), CELLS, "text", d_cond=8)
    torch.nn.init.normal_(m.encoder.proj.weight, std=0.5)
    a = m.encoder.encode_novel(DESC.text_embeddings(["A549"])[0])
    b = m.encoder.encode_novel(DESC.text_embeddings(["Jurkat"])[0])
    assert not torch.allclose(a, b, atol=1e-4)


def test_encode_sequences_places_the_insert_inside_real_flank():
    seqs = ["ACGT" * 50, "TTTT" * 50]
    idx = D.encode_sequences(seqs, margin=D.MARGIN)
    assert idx.shape == (2, 200 + 2 * D.MARGIN)
    # the insert must appear unchanged at the centre, not at the edge
    centre = idx[0, D.MARGIN : D.MARGIN + 200]
    assert list(centre[:4]) == [0, 1, 2, 3]
    assert idx.max() <= 4


def test_onehot_from_idx_maps_the_pad_symbol_to_all_zeros():
    idx = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    oh = D.onehot_from_idx(idx)
    assert oh.shape == (1, 4, 5)
    assert float(oh[0, :, 4].sum()) == 0.0
    assert float(oh[0, :, :4].sum()) == 4.0


def test_long_format_excludes_the_held_out_cell():
    n = 20
    data = D.MultiCellData(
        idx=np.zeros((n, 10), np.uint8),
        y=np.ones((n, 3), np.float32),
        se=np.ones((n, 3), np.float32),
        mask=np.ones((n, 3), bool),
        chrom=np.array(["chr1"] * n, dtype=object),
        cells=list(CELLS),
    )
    rows = np.ones(n, bool)
    si, ki, y, se = D.long_format(data, rows, ["K562", "HepG2"])
    assert len(si) == 2 * n
    assert set(np.unique(ki)) == {0, 1}, "the held-out cell leaked into training"
