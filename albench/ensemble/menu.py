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

Torch is imported lazily: listing and exporting configurations needs only numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from albench.model import SequenceModel

MENU_DIR = Path(__file__).resolve().parent / "menus"
ORACLE_ID = "oracle_v3"
SEQ_LEN = 200

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
        if rule is None and stored:
            by_id = self._by_id
            listed = [by_id[i] for i in stored if i in by_id]
            return listed + [
                e
                for e in self.ordered(self.per_size.get("rule", "first2"))
                if e.id not in set(stored)
            ]
        return self.ordered(rule or self.per_size.get("rule", "first2"))

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


def load_menu(name_or_path: str | Path = "v0") -> Menu:
    """Load a menu by name (``v0`` ships with the repo) or from a menu.json produced by ``scripts/program_oct22/build_menu.py``."""
    p = Path(str(name_or_path))
    if not p.exists():
        p = MENU_DIR / f"menu_{name_or_path}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"no menu {name_or_path!r}; shipped menus: {sorted(x.stem[5:] for x in MENU_DIR.glob('menu_*.json'))}"
            )
    per = MENU_DIR / "per_size.json"
    per_size = json.loads(per.read_text()) if per.exists() and p.parent == MENU_DIR else {}
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


def one_hot(seqs: Sequence[str], length: int = SEQ_LEN) -> np.ndarray:
    """(N, 4, L) float32; characters outside ACGT (and positions past the end) stay zero."""
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.zeros((len(seqs), 4, length), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:length].upper()):
            k = idx.get(ch)
            if k is not None:
                x[i, k, j] = 1.0
    return x


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


# ---------------------------------------------------------------------------------------------------------------------- training
@dataclass
class TrainedMember:
    """One trained configuration. ``state`` holds the best-validation weights on the CPU."""

    entry: MenuEntry
    state: dict[str, Any]
    seed: int
    val_pred: np.ndarray
    val_r: float
    epochs: int
    minutes: float
    n_params: int
    info: dict[str, Any] = field(default_factory=dict)
    _net: Any = field(default=None, repr=False, compare=False)

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
                self.entry, max_fwd_ms=1e9
            )  # no timing gate when only loading weights
            if net is None:
                raise RuntimeError(f"cannot rebuild {self.entry.id}: {reason}")
            net.load_state_dict(self.state)
            self._net = net.eval()
        return self._net


@contextmanager
def _patience(patience: int | None):
    """The training loop reads AL_PATIENCE (default 8); set it for the duration of one fit, then restore."""
    if patience is None:
        yield
        return
    old = os.environ.get("AL_PATIENCE")
    os.environ["AL_PATIENCE"] = str(int(patience))
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("AL_PATIENCE", None)
        else:
            os.environ["AL_PATIENCE"] = old


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
    seq_len: int = SEQ_LEN,
    max_params: int = 40_000_000,
    max_fwd_ms: float = 8000.0,
):
    """Instantiate an entry's network through the repo's sandboxed builder. Returns ``(net, n_params)`` or ``(None, reason)``.

    The builder includes a CPU forward-time gate (default 8 s) that is node dependent: it drops about 0 to 6 of the 33 entries on a
    given node (architecture 2 always). The experiments ran with this gate, so it is the default; ``max_fwd_ms=1e9`` disables it.
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
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 0,
    index: int = 0,
    patience: int | None = None,
    max_minutes: float | None = None,
    device: str | None = None,
    max_fwd_ms: float = 8000.0,
    log: Callable[[str], None] = print,
) -> TrainedMember | None:
    """Train one configuration; returns ``None`` (with the reason logged) if it fails to build, trains to a non-finite loss or explodes.

    ``index`` is the entry's position in the menu: together with ``seed`` it fixes the torch / numpy seeds exactly as the experiments
    did (1000 + 100 * seed + index), so a retrain here reproduces a ``menu_retrain.py`` run up to GPU non-determinism.
    """
    import torch

    from albench.arch_search.arch_autoresearch import predict_rc, train_one

    _require_optimizer(entry)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1000 + 100 * seed + index)
    np.random.seed(1000 + 100 * seed + index)
    net, n_params = build_net(entry, max_fwd_ms=max_fwd_ms)
    if net is None:
        log(f"    {entry.id} BUILD-FAIL {str(n_params)[:90]}")
        return None
    cap = max_minutes if max_minutes is not None else _cap_minutes(len(y_train), dev)
    t0 = time.time()
    with _patience(patience):
        model, info, _used = train_one(
            net,
            x_train,
            np.asarray(y_train, np.float32),
            x_val,
            np.asarray(y_val, np.float32),
            dict(entry.train_config),
            device=dev,
            max_sec=cap * 60,
            log=log,
        )
    if model is None:
        log(f"    {entry.id} TRAIN-FAIL {info.get('error')}")
        return None
    vp = predict_rc(model, x_val)
    if not np.isfinite(vp).all() or float(np.abs(vp).max()) > 1e3:
        log(f"    {entry.id} EXPLODED predictions (|pred| > 1e3); dropped")
        return None
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return TrainedMember(
        entry,
        state,
        seed,
        vp,
        pearson(vp, y_val),
        int(info.get("epochs_trained", 0)),
        (time.time() - t0) / 60.0,
        int(n_params),
        {"cap_minutes": cap},
    )


def member_predict(
    member: TrainedMember, x: np.ndarray, *, device: str | None = None, batch: int = 512
) -> np.ndarray:
    """Reverse-complement averaged predictions of one member on one-hot input ``x``; the network sits on the GPU only for the call."""
    import torch

    from albench.arch_search.arch_autoresearch import predict_rc

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = member.net().to(dev)
    try:
        out = [predict_rc(net, x[i : i + batch]) for i in range(0, len(x), batch)]
    finally:
        net.to("cpu")
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ---------------------------------------------------------------------------------------------------------------------- ensemble
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
        log: Callable[[str], None] = print,
    ):
        self.menu = menu
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
        self.stack: Any = None
        self.train_hash: str | None = None
        self._x_val = one_hot(self.val_sequences)

    @classmethod
    def for_size(
        cls,
        size: int,
        val_sequences: Sequence[str],
        val_labels: np.ndarray,
        *,
        menu: Menu | str = "v0",
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
        x = one_hot(sequences)
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
        if len(self.members) < 2:
            raise RuntimeError(
                f"only {len(self.members)} of {len(jobs)} members trained (size {size}); cannot stack"
            )
        self.train_hash = h
        self.fit_stack()

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
        """(members, N) reverse-complement averaged predictions (all members, or only those with positive stack weight)."""
        x = one_hot(sequences)
        ms = self.kept if only_kept else self.members
        dev = self.devices[0] if self.devices else None
        return np.stack([member_predict(m, x, device=dev) for m in ms])

    def predict(self, sequences: list[str]) -> np.ndarray:
        return self.stack.predict(self.member_predictions(sequences)).astype(np.float32)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        return self.epistemic_uncertainty(sequences)

    def epistemic_uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Standard deviation of the predictions of the members the stack uses (disagreement; shrinks as labels accumulate)."""
        return self.member_predictions(sequences, only_kept=True).std(axis=0)

    def embed(self, sequences: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "MenuEnsemble members are arbitrary architectures and expose no common embedding"
        )

    # -- EnsembleAcquisition hooks ----------------------------------------------------------------------------------
    @property
    def models(self) -> list[TrainedMember]:
        return self.kept

    def _encode_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        return one_hot(sequences)

    def _predict_member(self, member: TrainedMember, x: np.ndarray) -> np.ndarray:
        return member_predict(member, x, device=self.devices[0] if self.devices else None)

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
            "stack": self.stack.to_json(),
            "val_r_ensemble": pearson(
                self.stack.predict(np.stack([m.val_pred for m in self.members])), self.val_labels
            ),
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
        menu: Menu | str = "v0",
        devices: Sequence[str] | None = None,
    ) -> MenuEnsemble:
        """Reload a saved ensemble. The networks are rebuilt from the menu's source (checked against the stored sha1) and the weights loaded."""
        import torch

        from albench.ensemble.stack import ZStack

        d = Path(path)
        meta = json.loads((d / "ensemble.json").read_text())
        m = load_menu(menu) if isinstance(menu, str) else menu
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
                )
            )
        ens = cls(
            [mm.entry for mm in members],
            val_sequences,
            val_labels,
            size=meta.get("size"),
            seeds=int(meta.get("seeds", 1)),
            patience=meta.get("patience"),
            devices=devices,
            menu=m,
        )
        ens.members = members
        ens.stack = ZStack.from_json(meta["stack"])
        ens.train_hash = meta.get("train_hash")
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
    log: Callable[[str], None] = print,
) -> list[str]:
    """Train the menu's configurations for a dataset of ``size`` sequences into ``out_dir/members`` (resumable, shardable across jobs).

    Walks the priority order until K configurations have *built*: an entry that fails the build check is skipped and the next one
    takes its place, so every shard agrees on the same K. Each trained member writes ``<id>__s<seed>.pt`` (weights), ``.val.npy``
    (validation predictions) and ``.json`` (metrics). Shard ``(i, n)`` trains the jobs whose position is ``i`` mod ``n``; once all
    shards are done, :func:`stack_dir` fits the stack and writes ``ensemble.json``.
    """
    import torch

    members = Path(out_dir) / "members"
    members.mkdir(parents=True, exist_ok=True)
    row = menu.recipe(size)
    seeds = seeds if seeds is not None else int(row.get("seeds", 1))
    patience = patience if patience is not None else row.get("patience")
    want = k if k is not None else int(row.get("k", len(menu)))
    chosen = []
    for e in menu.priority():
        net, _ = build_net(e)
        if net is not None:
            chosen.append(e)
        if len(chosen) == want:
            break
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
) -> dict[str, Any]:
    """Combine the members in ``out_dir/members`` (written by :func:`train_menu`, possibly by many shards) into ``ensemble.json``."""
    from albench.ensemble.stack import fit_zscored_stack

    d = Path(out_dir)
    metas = sorted(
        (json.loads(p.read_text()) for p in (d / "members").glob("*.json")),
        key=lambda r: r["member"],
    )
    if len(metas) < 2:
        raise RuntimeError(f"{d}: {len(metas)} members; nothing to stack")
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
        "stack": st.to_json(),
        "val_r_ensemble": pearson(st.predict(v), val_labels),
        "members": rows,
    }
    (d / "ensemble.json").write_text(json.dumps(meta, indent=1))
    return meta
