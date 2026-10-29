"""The menu: which architecture + training-recipe configurations to train on a dataset, and the ensemble they form.

Why this exists. The study's deliverable is not one network but a fixed list of model-written architectures, each with the training
recipe it was found with, retrained on whatever labelled set is at hand and combined by a non-negative stack on the validation set.
Until now that lived in ``scripts/program_oct22`` (menu_retrain.py, build_menu.py) and in two ``menu_*.json`` files under ``outputs/``.
This module is the library form of it, so a student for the active-learning loop, a one-off retrain and a collaborator's acquisition
experiment all use the same configurations, the same training loop and the same ensembling.

Three things are kept apart on purpose:

* :class:`Menu` -- the configurations (source code of a ``Net`` class + ``train_config``), their provenance, the order in which they
  should be trained, and how many to train at each dataset size (``menus/per_size.json``, derived from the K-curve analysis).
* :func:`train_member` -- one configuration trained on one dataset with the repo's training loop (reverse-complement averaged MSE,
  shift augmentation, early stopping on the validation set, best weights restored).
* :class:`MenuEnsemble` -- a :class:`albench.model.SequenceModel` that trains the chosen configurations, stacks them, and exposes
  per-member predictions so acquisition functions can use their disagreement.

Input context. Every member sees the 200-bp insert inside its real plasmid context, as the oracle that labels it does:
``FLANK_CTX`` = 20 real bases per side (input length ``INPUT_LEN`` = 240, constants in ``albench.arch_search.harness_standard``). Training
encodes ``FLANK_CTX + SHIFT_MAX`` bases per side and ``flank_context.wrap`` slides the window by up to +-``SHIFT_MAX`` real bases (the
harness's own zero-padded shift is switched off for these members); validation and prediction use the centred window. Each trained member
records the context it was trained with (``input_context``: 20, or 0 for the bare insert) and is always encoded with that context, so
ensembles saved before the standard changed (no ``input_context`` key) still load and predict at 200 bp. Callers pass 200-bp insert
*sequences*; a one-hot array of any width ``200 + 2a`` is accepted too and re-framed from its central 200 columns.

Torch is imported lazily: listing and exporting configurations needs only numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from albench.arch_search.harness_standard import FLANK_CTX, INPUT_LEN, INSERT_LEN, SHIFT_MAX
from albench.model import SequenceModel

MENU_DIR = Path(__file__).resolve().parent / "menus"
ORACLE_ID = "oracle_v3"
SEQ_LEN = INSERT_LEN  # the insert; students are shown input_length(FLANK_CTX) = INPUT_LEN bases of it plus real flank

# Per-model wall-clock cap in H100-equivalent minutes by training-set size (x5 on a V100). These are the caps of the experiments; they
# were never reached by the retrained menu, so they only guard against a runaway configuration.
MAX_MINUTES_H100 = {
    10_000: 15.0,
    30_000: 25.0,
    100_000: 60.0,
    300_000: 120.0,
    600_000: 240.0,
    1_000_000: 360.0,
}


# ---------------------------------------------------------------------------------------------------------------------- menu
@dataclass(frozen=True)
class MenuEntry:
    """One configuration: a model-written network and the recipe it was found with."""

    id: str
    run: str  # the independent AutoResearch run (or "reference") the entry came from
    idea: str
    search_val: (
        float | None
    )  # validation r the entry reached in its own search (None for the reference)
    n_params: int | None
    source: str  # python source defining ``class Net(nn.Module)``; executed in a restricted namespace by validate.build_and_check
    train_config: dict[str, Any]

    @property
    def source_sha1(self) -> str:
        return hashlib.sha1(self.source.strip().encode()).hexdigest()[:12]


class Menu:
    """An ordered list of configurations plus the per-size recipe table."""

    def __init__(
        self,
        entries: Sequence[MenuEntry],
        per_size: dict[str, Any] | None = None,
        name: str = "custom",
    ):
        ids = [e.id for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("menu entry ids must be unique")
        self.entries = list(entries)
        self.per_size = per_size or {}
        self.name = name
        self._by_id = {e.id: e for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, entry_id: str) -> MenuEntry:
        return self._by_id[entry_id]

    # -- ordering ---------------------------------------------------------------------------------------------------
    def ordered(self, rule: str = "first2") -> list[MenuEntry]:
        """Entries in training priority order.

        ``val``: by the validation r each had in its own search. ``first1`` / ``first2``: the best one (two) entries of every
        search run come first, then everything else by search validation r -- the rule the K-curve analysis found to reach an
        approximately optimal ensemble with the fewest configurations. ``given``: the stored order.
        """
        key = lambda e: -(e.search_val if e.search_val is not None else -1.0)  # noqa: E731  (the reference entry has no search val)
        by_val = sorted(self.entries, key=key)
        if rule == "given":
            return list(self.entries)
        if rule == "val":
            return by_val
        if rule not in ("first1", "first2"):
            raise ValueError(f"unknown rule {rule!r}; use val, first1, first2 or given")
        n_first = int(rule[-1])
        runs: dict[str, list[MenuEntry]] = {}
        for e in by_val:
            runs.setdefault(e.run, []).append(e)
        head: list[MenuEntry] = []
        for rank in range(n_first):
            head += sorted((r[rank] for r in runs.values() if len(r) > rank), key=key)
        seen = {e.id for e in head}
        return head + [e for e in by_val if e.id not in seen]

    def priority(self, rule: str | None = None) -> list[MenuEntry]:
        """The training priority order: the study's tabulated order (``per_size.json``) unless a ``rule`` is given."""
        stored = self.per_size.get("order")
        skip = self.skipped
        if rule is None and stored:
            by_id = self._by_id
            listed = [by_id[i] for i in stored if i in by_id]
            order = listed + [
                e
                for e in self.ordered(self.per_size.get("rule", "first2"))
                if e.id not in set(stored)
            ]
        else:
            order = self.ordered(rule or self.per_size.get("rule", "first2"))
        return [e for e in order if e.id not in skip]

    @property
    def skipped(self) -> dict[str, str]:
        """Entries left out of the recipe (id -> reason), see ``per_size.json``: two exceed the CPU forward-time check on many nodes and one
        produces exploding predictions on distribution-shifted sequences, which is exactly what candidate scoring feeds it."""
        return dict(self.per_size.get("skip", {}))

    def for_size(
        self, size: int, *, k: int | None = None, rule: str | None = None
    ) -> list[MenuEntry]:
        """The configurations to train on a dataset of ``size`` sequences: the first K of the priority order.

        K comes from ``per_size.json`` (the nearest tabulated size at or below ``size``, or the smallest if below all); pass ``k`` /
        ``rule`` to override. Trainers walk further down the list when a build fails (see :func:`train_menu`).
        """
        order = self.priority(rule)
        return order[: (k if k is not None else int(self.recipe(size).get("k", len(order))))]

    def index_of(self, entry_id: str) -> int:
        """Position in the stored menu file: seeds the training run exactly as ``scripts/program_oct22/menu_retrain.py`` did."""
        return [e.id for e in self.entries].index(entry_id)

    def recipe(self, size: int) -> dict[str, Any]:
        """Per-size settings (K, patience, seeds, time cap) -- the nearest tabulated size at or below ``size``."""
        table = {int(s): r for s, r in self.per_size.get("sizes", {}).items()}
        if not table:
            return {}
        below = [s for s in table if s <= size]
        return dict(table[max(below) if below else min(table)])

    # -- io ---------------------------------------------------------------------------------------------------------
    def to_json(self) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "cell": e.run,
                "idea": e.idea,
                "orig_val": e.search_val,
                "n_params": e.n_params,
                "source": e.source,
                "train_config": e.train_config,
            }
            for e in self.entries
        ]

    @classmethod
    def from_json(
        cls,
        rows: Sequence[dict[str, Any]],
        per_size: dict[str, Any] | None = None,
        name: str = "custom",
    ) -> Menu:
        entries = [
            MenuEntry(
                id=r["id"],
                run=r.get("cell", "unknown"),
                idea=r.get("idea", ""),
                search_val=r.get("orig_val"),
                n_params=r.get("n_params"),
                source=r["source"],
                train_config=dict(r["train_config"]),
            )
            for r in rows
        ]
        return cls(entries, per_size, name)


DEFAULT_MENU = "v1c"


def load_menu(name_or_path: str | Path = DEFAULT_MENU) -> Menu:
    """Load a menu by name or from a menu.json produced by ``scripts/program_oct22/build_menu.py``.

    Shipped menus (``menu_<name>.json`` next to this module): ``v1c`` (the default: ``v1`` with the five architectures that were hard-wired to 200 positions rewritten
    length-agnostic, so all 20 build at the 240-base standard input; identical networks at 200 bp), ``v1`` (20 configurations built from runs on five reservoir families
    and a mixed corpus; five of them only build at 200 bp) and ``v0`` (30 usable configurations from genomic-only runs; the starting ensembles in the shared folder were
    trained with it). Each has its own ``per_size_<name>.json`` (``per_size.json`` for ``v0``) with the priority order, how many to train at each size, patience and seeds.
    """
    p = Path(str(name_or_path))
    if not p.exists():
        name = str(name_or_path)
        name = name[5:] if name.startswith("menu_") else name
        p = MENU_DIR / f"menu_{name}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"no menu {name_or_path!r}; shipped menus: {sorted(x.stem[5:] for x in MENU_DIR.glob('menu_*.json'))}"
            )
    version = p.stem[5:] if p.stem.startswith("menu_") else p.stem
    per_size: dict = {}
    if p.parent == MENU_DIR:
        for cand in (
            MENU_DIR / f"per_size_{version}.json",
            MENU_DIR / "per_size.json" if version == "v0" else None,
        ):
            if cand is not None and cand.exists():
                per_size = json.loads(cand.read_text())
                break
    return Menu.from_json(json.loads(p.read_text()), per_size, p.stem)


# ---------------------------------------------------------------------------------------------------------------------- data
def load_labelled(
    path: str | Path, *, allow_unstamped: bool = False
) -> tuple[list[str], np.ndarray]:
    """Sequences and oracle labels from a pool or validation ``.npz``; refuses a set that is not stamped ``oracle_v3``.

    Label provenance has bitten this project before (see docs/STRUCTURE.md): a set labelled by another oracle trains a student
    that looks fine and is quietly wrong, so the stamp is checked here instead of trusted.
    """
    z = np.load(path, allow_pickle=True)
    for key in ("sequences", "oracle_labels"):
        if key not in z.files:
            raise ValueError(f"{path} has no {key!r}")
    oid = str(z["oracle_id"]) if "oracle_id" in z.files else ""
    if oid != ORACLE_ID and not allow_unstamped:
        raise ValueError(
            f"{path} is stamped oracle_id={oid!r}, expected {ORACLE_ID!r} (pass allow_unstamped=True to override knowingly)"
        )
    y = np.asarray(z["oracle_labels"], dtype=np.float32).ravel()
    seqs = [str(s) for s in z["sequences"]]
    if len(seqs) != len(y) or not np.isfinite(y).all():
        raise ValueError(f"{path}: sequence/label mismatch or non-finite labels")
    return seqs, y


TEST_SETS = ("genomic", "ood", "random_32k")


def load_test_sets(
    names: Sequence[str] = TEST_SETS, path: str | Path | None = None
) -> dict[str, tuple[list[str], np.ndarray]]:
    """The study's headline test sets as ``{name: (sequences, oracle_v3 labels)}`` from the ``k562_test_sets`` asset (or ``path``).

    ``genomic`` = held-out chr7/chr13 sequences, ``ood`` = designed out-of-distribution sequences, ``random_32k`` = 32,000 uniformly random
    sequences. These are the sets every number in the ledger is measured on, so a result computed here is comparable with ours.
    """
    from albench.core.paths import resolve

    root = Path(resolve("k562_test_sets", path))
    out = {}
    for name in names:
        z = np.load(root / f"{name}_oracle.npz", allow_pickle=True)
        key = "oracle_mean" if "oracle_mean" in z.files else "oracle_labels"
        out[name] = ([str(x) for x in z["sequences"]], np.asarray(z[key], np.float32).ravel())
    return out


def evaluate(
    model: SequenceModel, names: Sequence[str] = TEST_SETS, path: str | Path | None = None
) -> dict[str, float]:
    """Pearson r of ``model.predict`` on each headline test set.

    The sets are 200-bp insert sequences; a :class:`MenuEnsemble` frames each member's input itself (insert + real plasmid flank for a
    member trained with context, the bare insert for one trained without), so nothing here encodes.
    """
    return {
        name: pearson(model.predict(seqs), y)
        for name, (seqs, y) in load_test_sets(names, path).items()
    }


def one_hot(seqs: Sequence[str], length: int = SEQ_LEN) -> np.ndarray:
    """(N, 4, L) float32; characters outside ACGT (and positions past the end) stay zero.

    This is the bare insert (no plasmid flank): the encoding of members trained without context. Use :func:`encode` for the standard input.
    """
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.zeros((len(seqs), 4, length), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:length].upper()):
            k = idx.get(ch)
            if k is not None:
                x[i, k, j] = 1.0
    return x


def input_length(flank_ctx: int = FLANK_CTX) -> int:
    """Width of a student's input: the insert plus ``flank_ctx`` real plasmid bases on each side (240 for the standard context)."""
    return SEQ_LEN + 2 * int(flank_ctx)


def encode(seqs: Sequence[str], flank_ctx: int = FLANK_CTX) -> np.ndarray:
    """(N, 4, 200 + 2*flank_ctx): each insert with ``flank_ctx`` real plasmid bases per side -- the centred window every member is
    evaluated on (no shift). ``flank_ctx=0`` is the bare insert, exactly :func:`one_hot`."""
    if flank_ctx == 0:
        return one_hot(seqs)
    from albench.arch_search.flank_context import wide_onehot

    return wide_onehot(seqs, int(flank_ctx))


def _train_margin(flank_ctx: int, shift_max: int) -> int:
    """Real bases per side a training array carries: the context plus room for the window shift (a member without context has neither)."""
    return int(flank_ctx) + int(shift_max) if flank_ctx else 0


def encode_train(
    seqs: Sequence[str], flank_ctx: int = FLANK_CTX, shift_max: int = SHIFT_MAX
) -> np.ndarray:
    """(N, 4, 200 + 2*(flank_ctx + shift_max)): the training input, wide enough for the +-``shift_max`` window shift of ``flank_context.wrap``
    (the bare insert when ``flank_ctx`` is 0)."""
    return encode(seqs, _train_margin(flank_ctx, shift_max))


def _as_window(x: np.ndarray | Sequence[str], flank_ctx: int) -> np.ndarray:
    """``x`` framed for ``flank_ctx`` real flank bases per side: sequences are encoded; a one-hot array of any width ``200 + 2a`` keeps its
    central 200 columns (the insert) and gets the plasmid flank put back around them, so a bare ``one_hot`` array handed to a member that
    was trained with context is framed correctly instead of crashing or, worse, being read at the wrong width. An array that already has the
    right width is returned unchanged."""
    if not isinstance(x, np.ndarray):
        return encode(x, flank_ctx)
    want = input_length(flank_ctx)
    w = x.shape[-1]
    if x.ndim != 3 or x.shape[1] != 4 or w < SEQ_LEN or (w - SEQ_LEN) % 2:
        raise ValueError(f"expected a (N, 4, 200 + 2a) one-hot array, got shape {x.shape}")
    if w == want:
        return np.ascontiguousarray(x, dtype=np.float32)
    a = (w - SEQ_LEN) // 2
    insert = x[:, :, a : a + SEQ_LEN]
    if flank_ctx == 0:
        return np.ascontiguousarray(insert, dtype=np.float32)
    from albench.arch_search.plasmid_flank import DOWNSTREAM_FULL, UPSTREAM_FULL

    up = one_hot([UPSTREAM_FULL[-flank_ctx:]], flank_ctx)[0]
    dn = one_hot([DOWNSTREAM_FULL[:flank_ctx]], flank_ctx)[0]
    n = len(x)
    return np.concatenate(
        [
            np.broadcast_to(up, (n, 4, flank_ctx)),
            insert.astype(np.float32, copy=False),
            np.broadcast_to(dn, (n, 4, flank_ctx)),
        ],
        axis=2,
    )


class _Encoded:
    """One list of sequences encoded once per input context: members trained with different contexts can share a batch without
    re-encoding it for each member (the encoding is a Python loop, minutes at a million sequences)."""

    def __init__(self, sequences: Sequence[str]):
        self.sequences = list(sequences)
        self._by_ctx: dict[int, np.ndarray] = {}

    def for_context(self, flank_ctx: int) -> np.ndarray:
        if flank_ctx not in self._by_ctx:
            self._by_ctx[flank_ctx] = encode(self.sequences, flank_ctx)
        return self._by_ctx[flank_ctx]

    def for_member(self, member: TrainedMember) -> np.ndarray:
        return self.for_context(member.input_context)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


# ---------------------------------------------------------------------------------------------------------------------- training
@dataclass
class TrainedMember:
    """One trained configuration. ``state`` holds the best-validation weights of the bare network on the CPU (no ``net.`` prefix of the
    flank-context wrapper). ``input_context`` is the number of real plasmid bases per side the member was trained to see (0 = the bare
    200-bp insert, the default for anything saved before the standard context existed); it fixes how its input is encoded."""

    entry: MenuEntry
    state: dict[str, Any]
    seed: int
    val_pred: np.ndarray
    val_r: float
    epochs: int
    minutes: float
    n_params: int
    info: dict[str, Any] = field(default_factory=dict)
    input_context: int = 0
    _net: Any = field(default=None, repr=False, compare=False)

    @property
    def input_length(self) -> int:
        return input_length(self.input_context)

    @property
    def member_id(self) -> str:
        return f"{self.entry.id}__s{self.seed}"

    def eval(self) -> TrainedMember:
        """No-op: lets code written for a list of torch modules (EnsembleAcquisition calls ``model.eval()``) iterate over members."""
        return self

    def net(self) -> Any:
        """The network with its weights loaded, built once (on the CPU) and cached."""
        if self._net is None:
            net, reason = build_net(
                self.entry, seq_len=self.input_length, max_fwd_ms=1e9
            )  # no timing gate when only loading weights
            if net is None:
                raise RuntimeError(f"cannot rebuild {self.entry.id}: {reason}")
            net.load_state_dict(_bare_state(net, self.state))
            self._net = net.eval()
        return self._net


def _bare_state(net: Any, state: dict[str, Any]) -> dict[str, Any]:
    """``state`` as the bare network's keys. A checkpoint written from the flank-context wrapper (``flank_context.wrap``) has every key under
    ``net.``; those saved by this library, and everything saved before the wrapper existed, do not. Decided by comparing key sets, because a
    model-written network may itself own a child called ``net``."""
    want = set(net.state_dict())
    if set(state) == want:
        return state
    stripped = {k[4:]: v for k, v in state.items() if k.startswith("net.")}
    return stripped if len(stripped) == len(state) and set(stripped) == want else state


# Env var and harness-function overrides are process-global but one ensemble may train its members on several threads (one per GPU): each
# override is reference-counted so the last thread to leave restores it, not the first.
_scope_lock = threading.Lock()
_scope_refs: dict[str, list[Any]] = {}  # key -> [count, original value]


@contextmanager
def _override(key: str, get: Callable[[], Any], put: Callable[[Any], None], value: Any):
    with _scope_lock:
        ref = _scope_refs.setdefault(key, [0, None])
        if ref[0] == 0:
            ref[1] = get()
        ref[0] += 1
        put(value)
    try:
        yield
    finally:
        with _scope_lock:
            ref[0] -= 1
            if ref[0] == 0:
                put(ref[1])
                del _scope_refs[key]


def _env_override(key: str, value: str):
    def put(v: Any) -> None:
        if v is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = v

    return _override(f"env:{key}", lambda: os.environ.get(key), put, value)


def _harness_reads_shift_env(aa: Any) -> bool:
    import inspect

    try:
        return "AL_SHIFT_MAX" in inspect.getsource(aa.train_one)
    except (OSError, TypeError):
        return False


@contextmanager
def _harness_scope(patience: int | None, shift_by_wrapper: bool):
    """Training-loop settings for one fit: the patience (``AL_PATIENCE``) and, for a member whose window shift is done by the flank-context
    wrapper, the harness's own zero-padded shift switched off (``AL_SHIFT_MAX=0``). A harness that predates that variable has the shift
    hard-coded, so its ``shift`` function is replaced by the identity for the duration: otherwise the two shifts would stack and the
    zero padding would reach into the window."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        if patience is not None:
            stack.enter_context(_env_override("AL_PATIENCE", str(int(patience))))
        if shift_by_wrapper:
            stack.enter_context(_env_override("AL_SHIFT_MAX", "0"))
            from albench.arch_search import arch_autoresearch as aa

            if not _harness_reads_shift_env(aa) and hasattr(aa, "shift"):
                stack.enter_context(
                    _override(
                        "fn:arch_autoresearch.shift",
                        lambda: aa.shift,
                        lambda f: setattr(aa, "shift", f),
                        lambda x, m: x,
                    )
                )
        yield


def _cap_minutes(size: int, device: str | None) -> float:
    import torch

    base = MAX_MINUTES_H100[
        min((s for s in MAX_MINUTES_H100 if s >= size), default=max(MAX_MINUTES_H100))
    ]
    name = (
        torch.cuda.get_device_name(torch.device(device))
        if (device or "").startswith("cuda") and torch.cuda.is_available()
        else ""
    )
    return base * (5.0 if "V100" in name else 1.0)


def _require_optimizer(entry: MenuEntry) -> None:
    """Most configurations use Muon. The training loop silently falls back to AdamW when the package is missing, which trains a different model
    from the one the study measured, so refuse instead (set ALBENCH_ALLOW_OPTIMIZER_FALLBACK=1 to accept the fallback knowingly)."""
    if (
        str(entry.train_config.get("optimizer", "")).lower() != "muon"
        or os.environ.get("ALBENCH_ALLOW_OPTIMIZER_FALLBACK") == "1"
    ):
        return
    try:
        from muon import SingleDeviceMuonWithAuxAdam  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"{entry.id} uses the Muon optimizer but the `muon` package is not importable. On the CSHL cluster use the repo .venv (muon.py is "
            "vendored in its site-packages); elsewhere: pip install git+https://github.com/KellerJordan/Muon. "
            "Set ALBENCH_ALLOW_OPTIMIZER_FALLBACK=1 to train with AdamW instead (different models from the study)."
        ) from e


def build_net(
    entry: MenuEntry,
    *,
    seq_len: int = INPUT_LEN,
    max_params: int = 40_000_000,
    max_fwd_ms: float = 8000.0,
):
    """Instantiate an entry's network through the repo's sandboxed builder. Returns ``(net, n_params)`` or ``(None, reason)``.

    The builder includes a CPU forward-time gate (default 8 s) that is node dependent: it drops about 0 to 6 of the 33 entries on a
    given node (architecture 2 always). The experiments ran with this gate, so it is the default; ``max_fwd_ms=1e9`` disables it.

    ``seq_len`` is the width of the network's input: the standard 240 (insert + 20 real bases per side) unless a member trained without
    context is being rebuilt (200). The network is built and probed at that width, so a configuration that only works at 200 fails here.
    """
    from albench.arch_search.validate import build_and_check

    net, verdict = build_and_check(
        entry.source, seq_len=seq_len, max_params=max_params, max_fwd_ms=max_fwd_ms
    )
    if net is None:
        return None, verdict.reason
    return net, int(verdict.n_params)


def train_member(
    entry: MenuEntry,
    x_train: np.ndarray | Sequence[str],
    y_train: np.ndarray,
    x_val: np.ndarray | Sequence[str],
    y_val: np.ndarray,
    *,
    seed: int = 0,
    index: int = 0,
    patience: int | None = None,
    max_minutes: float | None = None,
    device: str | None = None,
    max_fwd_ms: float = 1e9,
    flank_ctx: int = FLANK_CTX,
    shift_max: int = SHIFT_MAX,
    log: Callable[[str], None] = print,
) -> TrainedMember | None:
    """Train one configuration; returns ``None`` (with the reason logged) if it fails to build, trains to a non-finite loss or explodes.

    ``index`` is the entry's position in the menu: together with ``seed`` it fixes the torch / numpy seeds exactly as the experiments
    did (1000 + 100 * seed + index), so a retrain here reproduces a ``menu_retrain.py`` run up to GPU non-determinism.

    Input: the standard context. The network is built for ``200 + 2 * flank_ctx`` positions (240), trained on windows of the insert plus
    ``flank_ctx`` real plasmid bases per side that ``flank_context.wrap`` slides by up to +-``shift_max`` real bases (the harness's own
    zero-padded shift is off), and validated on the centred window. ``x_train`` / ``x_val`` may be 200-bp sequences or one-hot arrays of any
    width ``200 + 2a``: they are re-framed from their central 200 columns (see :func:`encode_train`), so ``one_hot`` arrays still work.
    ``flank_ctx=0`` reproduces the recipe from before the standard: bare 200-bp input and the harness's zero-padded shift (``shift_max`` is
    ignored); it is for reproducing old runs and is not a configuration of the menu. The weights are returned without the wrapper's ``net.``
    key prefix, so a checkpoint is the same kind of file in both cases.
    """
    import torch

    from albench.arch_search.arch_autoresearch import predict_rc, train_one

    flank_ctx, shift_max = int(flank_ctx), int(shift_max)
    if flank_ctx < 0 or shift_max < 0:
        raise ValueError("flank_ctx and shift_max must be >= 0")
    _require_optimizer(entry)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1000 + 100 * seed + index)
    np.random.seed(1000 + 100 * seed + index)
    random.seed(1000 + 100 * seed + index)  # the window shift draws from it
    x_tr = _as_window(x_train, _train_margin(flank_ctx, shift_max))
    x_va = _as_window(x_val, flank_ctx)
    net, n_params = build_net(entry, seq_len=input_length(flank_ctx), max_fwd_ms=max_fwd_ms)
    if net is None:
        log(f"    {entry.id} BUILD-FAIL {str(n_params)[:90]}")
        return None
    if flank_ctx:
        from albench.arch_search.flank_context import wrap

        net = wrap(net, flank_ctx, shift_max)
    cap = max_minutes if max_minutes is not None else _cap_minutes(len(y_train), dev)
    t0 = time.time()
    with _harness_scope(patience, shift_by_wrapper=bool(flank_ctx)):
        model, info, _used = train_one(
            net,
            x_tr,
            np.asarray(y_train, np.float32),
            x_va,
            np.asarray(y_val, np.float32),
            dict(entry.train_config),
            device=dev,
            max_sec=cap * 60,
            log=log,
        )
    if model is None:
        log(f"    {entry.id} TRAIN-FAIL {info.get('error')}")
        return None
    vp = predict_rc(model, x_va)
    if not np.isfinite(vp).all() or float(np.abs(vp).max()) > 1e3:
        log(f"    {entry.id} EXPLODED predictions (|pred| > 1e3); dropped")
        return None
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if flank_ctx:  # the wrapper holds the network as `net`: save the network's own weights
        state = {k[4:]: v for k, v in state.items() if k.startswith("net.")}
    return TrainedMember(
        entry,
        state,
        seed,
        vp,
        pearson(vp, y_val),
        int(info.get("epochs_trained", 0)),
        (time.time() - t0) / 60.0,
        int(n_params),
        {"cap_minutes": cap, "shift_max": shift_max if flank_ctx else None},
        input_context=flank_ctx,
    )


def member_predict(
    member: TrainedMember,
    x: np.ndarray | Sequence[str],
    *,
    device: str | None = None,
    batch: int = 512,
) -> np.ndarray:
    """Reverse-complement averaged predictions of one member on ``x``; the network sits on the GPU only for the call.

    ``x`` is framed for the member's own ``input_context`` (the centred window, no shift): 200-bp sequences are encoded, a one-hot array
    of another width is re-framed from its central 200 columns, one that already fits is used as is.
    """
    import torch

    from albench.arch_search.arch_autoresearch import predict_rc

    x = _as_window(x, member.input_context)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = member.net().to(dev)
    try:
        out = [predict_rc(net, x[i : i + batch]) for i in range(0, len(x), batch)]
    finally:
        net.to("cpu")
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def probe_sequences(seed: int = 0) -> list[str]:
    """Sequences far from the training distribution (uniform random, GC-extreme, repeats) used to screen out members that explode on them.

    One configuration of the menu produced predictions of order 1e5 on shifted evaluation sets in about half of its retrainings while
    looking normal on genomic ones; acquisition scores exactly such sequences, so a member is checked here before it can enter an ensemble.
    """
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for gc, n in ((0.5, 400), (0.8, 300), (0.2, 300)):
        p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T
        out += ["".join(rng.choice(list("ACGT"), SEQ_LEN, p=p)) for _ in range(n)]
    for unit in ("A", "C", "G", "T", "AC", "GC", "CT", "AT"):
        base = (unit * SEQ_LEN)[:SEQ_LEN]
        out.append(base)
        for _ in range(20):
            s = list(base)
            for j in rng.choice(SEQ_LEN, 10, replace=False):
                s[j] = "ACGT"[rng.integers(4)]
            out.append("".join(s))
    return out


def exploded(pred: np.ndarray) -> bool:
    """The study's screen: non-finite predictions, or any magnitude above 1e3, on a set the model is asked to score."""
    return (not bool(np.isfinite(pred).all())) or float(np.abs(pred).max()) > 1e3


# ---------------------------------------------------------------------------------------------------------------------- ensemble
def _common_context(contexts: Any) -> int | None:
    """The input context all members share, or ``None`` if they differ (or there are none)."""
    found = set(contexts)
    return found.pop() if len(found) == 1 else None


class MenuEnsemble(SequenceModel):
    """The menu trained on one labelled set and stacked -- a student for :class:`albench.core.loop.ALLoop`.

    The validation set is fixed at construction (the study uses the 29,449 real chr19/21/X sequences labelled by oracle_v3,
    "V0c"): it drives early stopping of every member and the fit of the stack weights. ``fit(sequences, labels)`` then trains every
    configuration on the labelled data from scratch, as the loop's other students do.

    Acquisition: ``member_predictions`` returns the (members, N) matrix; ``uncertainty`` / ``epistemic_uncertainty`` is the standard
    deviation across the members with positive stack weight; ``models`` / ``_encode_sequences`` / ``_predict_member`` let
    :class:`albench.acquisition.ensemble_acq.EnsembleAcquisition` use the members directly. ``embed`` is not available (the members
    are arbitrary architectures), so embedding-based acquisition must fall back, as the protocol allows.
    """

    separates_uncertainty = False

    def __init__(
        self,
        entries: Sequence[MenuEntry],
        val_sequences: Sequence[str],
        val_labels: np.ndarray,
        *,
        size: int | None = None,
        seeds: int = 1,
        patience: int | None = None,
        max_minutes: float | None = None,
        devices: Sequence[str] | None = None,
        menu: Menu | None = None,
        flank_ctx: int = FLANK_CTX,
        shift_max: int = SHIFT_MAX,
        log: Callable[[str], None] = print,
    ):
        self.menu = menu
        self.flank_ctx = int(
            flank_ctx
        )  # the input context of the members ``fit`` trains; each member remembers its own
        self.shift_max = int(shift_max)
        self._entries = list(entries)
        self.val_sequences = list(val_sequences)
        self.val_labels = np.asarray(val_labels, np.float32)
        self.size_hint = size
        self.seeds = seeds
        self.patience = patience
        self.max_minutes = max_minutes
        self.devices = list(devices) if devices else None
        self.log = log
        self.members: list[TrainedMember] = []
        self.screened_out: list[str] = []
        self.stack: Any = None
        self.train_hash: str | None = None
        self._x_val_cache: tuple[int, np.ndarray] | None = None

    @property
    def _x_val(self) -> np.ndarray:
        """The validation set as the centred window for ``fit`` (built on first use: an ensemble that is only loaded to predict never needs it)."""
        if self._x_val_cache is None or self._x_val_cache[0] != self.flank_ctx:
            self._x_val_cache = (self.flank_ctx, encode(self.val_sequences, self.flank_ctx))
        return self._x_val_cache[1]

    @classmethod
    def for_size(
        cls,
        size: int,
        val_sequences: Sequence[str],
        val_labels: np.ndarray,
        *,
        menu: Menu | str = DEFAULT_MENU,
        **kw: Any,
    ) -> MenuEnsemble:
        """The ensemble recommended for training sets of about ``size`` sequences: the per-size K, seeds and patience of the menu table."""
        m = load_menu(menu) if isinstance(menu, str) else menu
        row = m.recipe(size)
        kw.setdefault("seeds", int(row.get("seeds", 1)))
        kw.setdefault("patience", row.get("patience"))
        return cls(m.for_size(size), val_sequences, val_labels, size=size, menu=m, **kw)

    # -- training ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _hash(seqs: Sequence[str]) -> str:
        h = hashlib.sha1()
        for s in seqs:
            h.update(s.encode())
        return h.hexdigest()[:16]

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        """Train every configuration (x seeds) on the labelled data, then fit the stack. A no-op if this exact training set was
        already fitted or loaded (so the loop's first ``fit`` does not retrain a loaded ensemble)."""
        h = self._hash(sequences)
        if self.members and h == self.train_hash:
            return
        x = encode_train(sequences, self.flank_ctx, self.shift_max)
        y = np.asarray(labels, np.float32)
        jobs = [
            (self.menu.index_of(e.id) if self.menu else i, e, s)
            for s in range(self.seeds)
            for i, e in enumerate(self._entries)
        ]
        size = self.size_hint or len(y)

        def run(job: tuple[int, MenuEntry, int], device: str | None) -> TrainedMember | None:
            i, e, s = job
            return train_member(
                e,
                x,
                y,
                self._x_val,
                self.val_labels,
                seed=s,
                index=i,
                patience=self.patience,
                max_minutes=self.max_minutes,
                device=device,
                flank_ctx=self.flank_ctx,
                shift_max=self.shift_max,
                log=self.log,
            )

        if self.devices and len(self.devices) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                len(self.devices)
            ) as ex:  # one worker per GPU; each takes the next job when it finishes
                results = list(
                    ex.map(
                        lambda ij: run(ij[1], self.devices[ij[0] % len(self.devices)]),
                        enumerate(jobs),
                    )
                )
        else:
            results = [run(j, self.devices[0] if self.devices else None) for j in jobs]
        self.members = [r for r in results if r is not None]
        self.screen()
        if len(self.members) < 2:
            raise RuntimeError(
                f"only {len(self.members)} of {len(jobs)} members trained (size {size}); cannot stack"
            )
        self.train_hash = h
        self.fit_stack()

    def screen(self) -> list[str]:
        """Drop members that explode on the probe set (random, GC-extreme and repetitive sequences); returns their ids. Each member sees the
        probes framed for its own input context."""
        probes = _Encoded(probe_sequences())
        dev = self.devices[0] if self.devices else None
        bad = [
            m for m in self.members if exploded(member_predict(m, probes.for_member(m), device=dev))
        ]
        for m in bad:
            self.log(f"    {m.member_id} explodes on the probe sequences; dropped")
        self.screened_out = [m.member_id for m in bad]
        self.members = [m for m in self.members if m not in bad]
        return self.screened_out

    def fit_stack(self) -> None:
        """Non-negative ElasticNet on the z-scored validation predictions of every member (zero weight = member unused)."""
        from albench.ensemble.stack import fit_zscored_stack

        v = np.stack([m.val_pred for m in self.members])
        self.stack = fit_zscored_stack(v, self.val_labels)

    # -- prediction -------------------------------------------------------------------------------------------------
    @property
    def kept(self) -> list[TrainedMember]:
        if self.stack is None:
            raise RuntimeError("fit or load the ensemble first")
        return [m for m, w in zip(self.members, self.stack.coef) if w > 0]

    def member_predictions(
        self, sequences: Sequence[str], *, only_kept: bool = False
    ) -> np.ndarray:
        """(members, N) reverse-complement averaged predictions (all members, or only those with positive stack weight). ``sequences`` are
        200-bp inserts; each member is fed its own input context (the centred window with the real plasmid flank it was trained with)."""
        enc = _Encoded(sequences)
        ms = self.kept if only_kept else self.members
        dev = self.devices[0] if self.devices else None
        preds = np.stack([member_predict(m, enc.for_member(m), device=dev) for m in ms])
        for j, m in enumerate(
            ms
        ):  # a member that explodes on this batch is masked (NaN) rather than allowed to dominate the ensemble
            if exploded(preds[j]):
                warnings.warn(
                    f"{m.member_id} produced exploding predictions on this batch and is excluded from it",
                    stacklevel=2,
                )
                preds[j] = np.nan
        return preds

    def predict(self, sequences: list[str]) -> np.ndarray:
        p = self.member_predictions(sequences)
        return self.stack.predict(p, active=~np.isnan(p).any(axis=1)).astype(np.float32)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        return self.epistemic_uncertainty(sequences)

    def epistemic_uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Standard deviation of the predictions of the members the stack uses (disagreement; shrinks as labels accumulate)."""
        return np.nanstd(self.member_predictions(sequences, only_kept=True), axis=0)

    def embed(self, sequences: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "MenuEnsemble members are arbitrary architectures and expose no common embedding"
        )

    # -- EnsembleAcquisition hooks ----------------------------------------------------------------------------------
    @property
    def models(self) -> list[TrainedMember]:
        return self.kept

    def _encode_sequences(self, sequences: Sequence[str]) -> np.ndarray | dict[int, np.ndarray]:
        """The sequences as the members' input: one (N, 4, 200 + 2*ctx) array when the members share an input context (all of them, normally),
        else ``{ctx: array}`` with one entry per context present; :meth:`_predict_member` takes either."""
        enc = _Encoded(sequences)
        ctxs = sorted({m.input_context for m in self.members}) or [self.flank_ctx]
        return enc.for_context(ctxs[0]) if len(ctxs) == 1 else {c: enc.for_context(c) for c in ctxs}

    def _predict_member(
        self, member: TrainedMember, x: np.ndarray | dict[int, np.ndarray] | Sequence[str]
    ) -> np.ndarray:
        """One member's predictions on encoded input (EnsembleAcquisition passes the output of :meth:`_encode_sequences`) or on raw sequences
        (BatchBALD passes those). Input is framed for this member's own context whatever it was encoded with."""
        if isinstance(x, dict):
            if member.input_context not in x:
                raise KeyError(f"no encoding for input context {member.input_context}")
            x = x[member.input_context]
        p = member_predict(member, x, device=self.devices[0] if self.devices else None)
        return np.full_like(p, np.nan) if exploded(p) else p

    # -- persistence ------------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write ``ensemble.json`` (members, weights, validation metrics, provenance) and one ``.pt`` of weights per member."""
        import torch

        out = Path(path)
        if (
            out.suffix == ".pt"
        ):  # the AL loop passes <round>/student_checkpoint.pt; use that name as a directory
            out = out.with_suffix("")
        (out / "members").mkdir(parents=True, exist_ok=True)
        rows = []
        for m, w in zip(self.members, self.stack.coef):
            torch.save(m.state, out / "members" / f"{m.member_id}.pt")
            np.save(out / "members" / f"{m.member_id}.val.npy", m.val_pred.astype(np.float32))
            rows.append(
                {
                    "member": m.member_id,
                    "entry": m.entry.id,
                    "source_sha1": m.entry.source_sha1,
                    "seed": m.seed,
                    "weight": float(w),
                    "val_r": m.val_r,
                    "epochs": m.epochs,
                    "minutes": m.minutes,
                    "n_params": m.n_params,
                    "input_context": m.input_context,
                    "shift_max": m.info.get("shift_max"),
                }
            )
        meta = {
            "format": 1,
            "menu": getattr(self.menu, "name", "custom"),
            "train_hash": self.train_hash,
            "size": self.size_hint,
            "oracle_id": ORACLE_ID,
            "patience": self.patience,
            "seeds": self.seeds,
            "input_context": _common_context(m.input_context for m in self.members),
            "stack": self.stack.to_json(),
            "val_r_ensemble": pearson(
                self.stack.predict(np.stack([m.val_pred for m in self.members])), self.val_labels
            ),
            "screened_out": self.screened_out,
            "members": rows,
        }
        (out / "ensemble.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(
        cls,
        path: str | Path,
        val_sequences: Sequence[str],
        val_labels: np.ndarray,
        *,
        menu: Menu | str | None = None,
        devices: Sequence[str] | None = None,
        flank_ctx: int | None = None,
        shift_max: int = SHIFT_MAX,
    ) -> MenuEnsemble:
        """Reload a saved ensemble. The networks are rebuilt from the menu's source (checked against the stored sha1) and the weights loaded.

        Each member comes back with the input context it was trained with (``input_context`` in its row; a row without the key was saved
        before the standard context existed and is a 200-bp member), so an old ensemble predicts exactly as it always did. ``flank_ctx`` is
        the context of the members a later ``fit`` would train: by default the one the loaded members share (a loaded old ensemble keeps
        refitting as an old-style one), the standard if they differ.
        """
        import torch

        from albench.ensemble.stack import ZStack

        d = Path(path)
        meta = json.loads((d / "ensemble.json").read_text())
        m = (
            menu if isinstance(menu, Menu) else load_menu(menu or meta.get("menu", DEFAULT_MENU))
        )  # default: the menu the ensemble was trained with
        members = []
        for r in meta["members"]:
            e = m[r["entry"]]
            if e.source_sha1 != r["source_sha1"]:
                raise ValueError(
                    f"{r['entry']}: menu source changed since this ensemble was saved ({e.source_sha1} != {r['source_sha1']})"
                )
            state = torch.load(d / "members" / f"{r['member']}.pt", map_location="cpu")
            vp = np.load(d / "members" / f"{r['member']}.val.npy")
            members.append(
                TrainedMember(
                    e,
                    state,
                    int(r["seed"]),
                    vp,
                    float(r["val_r"]),
                    int(r["epochs"]),
                    float(r["minutes"]),
                    int(r["n_params"]),
                    {"shift_max": r.get("shift_max")},
                    input_context=int(r.get("input_context", 0)),
                )
            )
        if flank_ctx is None:
            flank_ctx = _common_context(mm.input_context for mm in members)
            flank_ctx = FLANK_CTX if flank_ctx is None else flank_ctx
        ens = cls(
            [mm.entry for mm in members],
            val_sequences,
            val_labels,
            size=meta.get("size"),
            seeds=int(meta.get("seeds", 1)),
            patience=meta.get("patience"),
            devices=devices,
            menu=m,
            flank_ctx=flank_ctx,
            shift_max=shift_max,
        )
        ens.members = members
        ens.stack = ZStack.from_json(meta["stack"])
        ens.train_hash = meta.get("train_hash")
        ens.screened_out = list(meta.get("screened_out", []))
        return ens


def train_menu(
    menu: Menu,
    size: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: str | Path,
    *,
    k: int | None = None,
    seeds: int | None = None,
    patience: int | None = None,
    shard: tuple[int, int] = (0, 1),
    device: str | None = None,
    flank_ctx: int = FLANK_CTX,
    shift_max: int = SHIFT_MAX,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Train the menu's configurations for a dataset of ``size`` sequences into ``out_dir/members`` (resumable, shardable across jobs).

    Takes the first K configurations of the priority order (the entries in ``per_size.json["skip"]`` are not in it), so every shard agrees
    on the same list; a member that fails to train is reported and left out. Each trained member writes ``<id>__s<seed>.pt`` (weights), ``.val.npy``
    (validation predictions) and ``.json`` (metrics). Shard ``(i, n)`` trains the jobs whose position is ``i`` mod ``n``; once all
    shards are done, :func:`stack_dir` fits the stack and writes ``ensemble.json``.

    Members are trained with the standard input context (see :func:`train_member`; ``x_train`` / ``x_val`` may be 200-bp one-hot arrays, they
    are framed once here) and each member's ``.json`` records ``input_context``, which :func:`stack_dir` and ``MenuEnsemble.load`` read back.
    """
    import torch

    x_train = _as_window(
        x_train, _train_margin(flank_ctx, shift_max)
    )  # framed once, not per member
    x_val = _as_window(x_val, flank_ctx)
    members = Path(out_dir) / "members"
    members.mkdir(parents=True, exist_ok=True)
    row = menu.recipe(size)
    seeds = seeds if seeds is not None else int(row.get("seeds", 1))
    patience = patience if patience is not None else row.get("patience")
    want = k if k is not None else int(row.get("k", len(menu)))
    chosen = menu.priority()[:want]
    jobs = [(e, s) for s in range(seeds) for e in chosen]
    done: list[str] = []
    for j, (e, s) in enumerate(jobs):
        if j % shard[1] != shard[0]:
            continue
        mid = f"{e.id}__s{s}"
        if (members / f"{mid}.json").exists():
            done.append(mid)
            continue
        m = train_member(
            e,
            x_train,
            y_train,
            x_val,
            y_val,
            seed=s,
            index=menu.index_of(e.id),
            patience=patience,
            device=device,
            flank_ctx=flank_ctx,
            shift_max=shift_max,
            log=log,
        )
        if m is None:
            continue
        torch.save(m.state, members / f"{mid}.pt")
        np.save(members / f"{mid}.val.npy", m.val_pred.astype(np.float32))
        meta = {
            "member": mid,
            "entry": e.id,
            "source_sha1": e.source_sha1,
            "seed": s,
            "val_r": m.val_r,
            "epochs": m.epochs,
            "minutes": m.minutes,
            "n_params": m.n_params,
            "size": size,
            "patience": patience,
            "input_context": m.input_context,
            "shift_max": m.info.get("shift_max"),
        }
        (members / f"{mid}.json").write_text(
            json.dumps(meta)
        )  # written last: its presence marks the member as complete
        log(f"    {mid} val={m.val_r:.4f} {m.n_params:,}p {m.epochs}ep {m.minutes:.1f}min")
        done.append(mid)
    return done


def stack_dir(
    menu: Menu,
    out_dir: str | Path,
    val_labels: np.ndarray,
    *,
    size: int | None = None,
    train_hash: str | None = None,
    screen: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Combine the members in ``out_dir/members`` (written by :func:`train_menu`, possibly by many shards) into ``ensemble.json``.

    With ``screen`` every member is run on :func:`probe_sequences` first and one that explodes is left out of the stack (listed under
    ``screened_out``); this loads each member's network, so use a GPU node for hundreds of members. A member's ``input_context`` (absent in
    the ``.json`` of members trained before the standard context existed: 0, the bare insert) decides how the probes are framed for it.
    """
    import torch

    from albench.ensemble.stack import fit_zscored_stack

    d = Path(out_dir)
    metas = sorted(
        (json.loads(p.read_text()) for p in (d / "members").glob("*.json")),
        key=lambda r: r["member"],
    )
    screened: list[str] = []
    if screen:
        probes = _Encoded(probe_sequences())
        kept = []
        for r in metas:
            m = TrainedMember(
                menu[r["entry"]],
                torch.load(d / "members" / f"{r['member']}.pt", map_location="cpu"),
                int(r["seed"]),
                np.zeros(0),
                float(r["val_r"]),
                0,
                0.0,
                0,
                input_context=int(r.get("input_context", 0)),
            )
            if exploded(member_predict(m, probes.for_member(m), device=device)):
                screened.append(r["member"])
            else:
                kept.append(r)
        metas = kept
    if len(metas) < 2:
        raise RuntimeError(f"{d}: {len(metas)} usable members; nothing to stack")
    v = np.stack([np.load(d / "members" / f"{r['member']}.val.npy") for r in metas])
    st = fit_zscored_stack(v, np.asarray(val_labels, np.float32))
    rows = [{**r, "weight": float(w)} for r, w in zip(metas, st.coef)]
    meta = {
        "format": 1,
        "menu": menu.name,
        "train_hash": train_hash,
        "size": size if size is not None else metas[0].get("size"),
        "oracle_id": ORACLE_ID,
        "patience": metas[0].get("patience"),
        "seeds": 1 + max(r["seed"] for r in metas),
        "input_context": _common_context(int(r.get("input_context", 0)) for r in metas),
        "stack": st.to_json(),
        "val_r_ensemble": pearson(st.predict(v), val_labels),
        "screened_out": screened,
        "members": rows,
    }
    (d / "ensemble.json").write_text(json.dumps(meta, indent=1))
    return meta
