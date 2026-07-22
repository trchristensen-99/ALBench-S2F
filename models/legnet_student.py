"""LegNet student wrapper implementing the SequenceModel interface.

Analogous to DREAMCNNStudent but uses the LegNet architecture
(inverted residual blocks with SE attention and ResidualConcat connections).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from albench.model import SequenceModel
from models.legnet import LegNet, QuickGELU, one_hot_encode_batch
from models.loss_utils import NaNMaskedMSELoss, YeastKLLoss
from models.training import train_model_optimized
from models.training_base import create_optimizer_and_scheduler


def _resolve_activation(name: str):
    """Map an activation name to an nn.Module class (default SiLU = LegNet default)."""
    return {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "quickgelu": QuickGELU,
        "mish": nn.Mish,
        "elu": nn.ELU,
    }.get(str(name).lower(), nn.SiLU)


# Per-batch schedulers step every optimizer step; everything else steps per epoch.
# Anything not handled by name is treated as a per-epoch scheduler.
_PER_BATCH_SCHEDULERS = {"OneCycleLR", "CyclicLR"}


def build_lr_schedulers(optimizer, train_config, steps_per_epoch: int, opt_name: str):
    """Build the LR schedule for a tunable `train_config.lr_schedule` axis.

    Returns ``(scheduler, epoch_scheduler)`` where at most one is non-None:
      * ``scheduler``       — stepped per optimizer step (OneCycle/Cyclic),
      * ``epoch_scheduler`` — stepped per epoch (plateau on val metric, or epoch-based).

    Named menu (LR_SCHEDULE_CHOICES): plateau, onecycle, cosine, cosine_warm, step,
    exponential, constant. The LLM AutoResearch strategy may also pass an OFF-MENU name
    (a torch.optim.lr_scheduler class name) plus extra["lr_schedule_kwargs"]; we build it
    generically and, on any failure, fall back to ReduceLROnPlateau so a run never crashes.
    """
    sched = str(getattr(train_config, "lr_schedule", "plateau") or "plateau")
    lr = train_config.lr
    epochs = train_config.epochs
    L = torch.optim.lr_scheduler

    def _plateau():
        return L.ReduceLROnPlateau(
            optimizer,
            mode="max",  # ranked on val pearson_r
            factor=getattr(train_config, "lr_plateau_factor", 0.5),
            patience=getattr(train_config, "lr_plateau_patience", 5),
            threshold=getattr(train_config, "min_delta", 1e-3),
        )

    if sched == "plateau":
        return None, _plateau()
    if sched == "onecycle":
        return (
            L.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=steps_per_epoch * epochs,
                pct_start=getattr(train_config, "pct_start", 0.3),
                cycle_momentum=(opt_name != "muon"),
            ),
            None,
        )
    if sched == "cosine":
        return None, L.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if sched == "cosine_warm":
        return None, L.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 4))
    if sched == "step":
        return None, L.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    if sched == "exponential":
        return None, L.ExponentialLR(optimizer, gamma=0.95)
    if sched in ("constant", "none"):
        return None, None

    # Off-menu: try to build a torch scheduler class by name with LLM-supplied kwargs.
    kwargs = dict((getattr(train_config, "extra", None) or {}).get("lr_schedule_kwargs", {}))
    cls = getattr(L, sched, None)
    if cls is not None:
        try:
            obj = cls(optimizer, **kwargs)
            if sched in _PER_BATCH_SCHEDULERS:
                return obj, None
            return None, obj
        except Exception as e:  # noqa: BLE001 — never let an off-menu name crash a run
            print(f"⚠ off-menu lr_schedule {sched!r} failed ({e}); falling back to plateau")
    else:
        print(f"⚠ unknown lr_schedule {sched!r}; falling back to plateau")
    return None, _plateau()


@dataclass
class TrainConfig:
    """Training hyperparameters for one ensemble member."""

    batch_size: int = 1024
    epochs: int = 80
    lr: float = 0.005
    weight_decay: float = 0.01
    pct_start: float = 0.3
    early_stopping_patience: int | None = None
    min_delta: float = 0.0
    num_workers: int = 2
    shift_aug: bool = False
    max_shift: int = 15
    evoaug_intensity: str | None = None  # None | "light" | "medium" | "heavy"
    evoaug_prob: float = 0.5  # per-sample apply probability
    optimizer: str = "adamw"  # {"adam", "adamw", "muon"} — see fit() for muon wiring
    use_compile: bool = True  # torch.compile (set False for fast HP search with varied shapes)
    use_reverse_complement: bool = False  # if True, train averages fwd+rc loss and predictions
    loss: str = "mse"  # {"mse", "huber", "smoothl1"} — single-task regression criterion
    huber_delta: float = 1.0  # delta for huber/smoothl1 loss


class _InMemorySequenceDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LegNetStudent(SequenceModel):
    """SequenceModel wrapper around an ensemble of LegNet models."""

    def __init__(
        self,
        in_channels: int = 4,
        sequence_length: int = 200,
        task_mode: str = "k562",
        ensemble_size: int = 3,
        device: str | None = None,
        train_config: TrainConfig | None = None,
        block_sizes: list[int] | None = None,
        ks: int = 5,
        multitask: bool = False,
        dropout: float = 0.0,
        conv_dropout: float | None = None,
        dense_dropout: float = 0.0,
        block_class: str = "eff",
        activation: str = "silu",
        se_reduction: int = 4,
        pool_sizes: list | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.task_mode = task_mode
        self.ensemble_size = ensemble_size
        self.multitask = multitask
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_config = train_config or TrainConfig()

        act_cls = _resolve_activation(activation)
        self.models = [
            LegNet(
                in_channels=in_channels,
                block_sizes=block_sizes,
                ks=ks,
                activation=act_cls,
                se_reduction=se_reduction,
                task_mode=task_mode,
                multitask=multitask,
                dropout=dropout,
                conv_dropout=conv_dropout,
                dense_dropout=dense_dropout,
                block_class=block_class,
                pool_sizes=pool_sizes,
            ).to(self.device)
            for _ in range(ensemble_size)
        ]
        self.histories: list[dict] = []

    def _encode_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        """Encode sequence strings to (N, 4, L) tensor."""
        target_len = self.sequence_length
        standardized: list[str] = []
        for seq in sequences:
            seq = seq.upper()
            if len(seq) < target_len:
                pad = target_len - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > target_len:
                start = (len(seq) - target_len) // 2
                seq = seq[start : start + target_len]
            standardized.append(seq)
        arr = one_hot_encode_batch(standardized, seq_len=target_len)
        return torch.from_numpy(arr)

    def _predict_member(self, model: LegNet, x: torch.Tensor) -> np.ndarray:
        model.eval()
        preds = []
        bs = 512
        with torch.no_grad():
            for i in range(0, len(x), bs):
                batch = x[i : i + bs].to(self.device)
                out = model(batch).detach().cpu().numpy()
                if self.multitask:
                    preds.append(out)  # (bs, 3)
                else:
                    preds.append(out.reshape(-1))
        return np.concatenate(preds)

    def predict(self, sequences: list[str], cell_type_idx: int = 0) -> np.ndarray:
        """Predict as mean across ensemble members.

        Args:
            sequences: Input DNA sequences.
            cell_type_idx: For multitask models, which cell type to return
                (0=K562, 1=HepG2, 2=SknSh). Ignored for single-task.
        """
        x = self._encode_sequences(sequences)
        preds = [self._predict_member(model, x) for model in self.models]
        mean_pred = np.mean(np.stack(preds, axis=0), axis=0)
        if self.multitask:
            return mean_pred[:, cell_type_idx]
        return mean_pred

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        """MC dropout variance (30 passes per member)."""
        x = self._encode_sequences(sequences).to(self.device)
        all_vars: list[np.ndarray] = []
        for model in self.models:
            model.train()
            passes = []
            for _ in range(30):
                with torch.no_grad():
                    passes.append(model(x).detach().cpu().numpy().reshape(-1))
            all_vars.append(np.var(np.stack(passes, axis=0), axis=0))
        return np.mean(np.stack(all_vars, axis=0), axis=0)

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Extract embeddings from the main block output (global avg pooled)."""
        x = self._encode_sequences(sequences)
        embeds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                batch = x.to(self.device)
                h = model.stem_block(batch)
                h = model.main(h)
                # Global average pool -> (B, block_sizes[-1])
                pooled = h.mean(dim=2)
                embeds.append(pooled.cpu().numpy())
        return np.mean(np.stack(embeds, axis=0), axis=0)

    def fit(
        self,
        sequences: list[str],
        labels: np.ndarray,
        val_sequences: list[str] | None = None,
        val_labels: np.ndarray | None = None,
        epoch_callback=None,
    ) -> None:
        """Train all ensemble members.

        If val_sequences/val_labels are provided, they are used for early
        stopping and best-model selection. Otherwise a 10% random split of
        the training data is used.
        """
        x = self._encode_sequences(sequences)
        y = torch.from_numpy(labels.astype(np.float32))

        if val_sequences is not None and val_labels is not None:
            x_val = self._encode_sequences(val_sequences)
            y_val = torch.from_numpy(val_labels.astype(np.float32))
        else:
            # Internal 10% split
            n_val = max(50, int(0.1 * len(x)))
            perm = torch.randperm(len(x))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            x_val, y_val = x[val_idx], y[val_idx]
            x, y = x[train_idx], y[train_idx]

        train_dataset = _InMemorySequenceDataset(x, y)
        val_dataset = _InMemorySequenceDataset(x_val, y_val)
        nw = self.train_config.num_workers
        # Deterministic shuffle + per-worker seeding: torch.initial_seed() reflects
        # the seed set by the caller (train_one_model does torch.manual_seed(hp.seed)),
        # so the shuffle order and any worker-side RNG are reproducible for a given
        # hp.seed and differ across seeds (ensemble diversity).
        base_seed = int(torch.initial_seed() % (2**31 - 1))
        loader_gen = torch.Generator()
        loader_gen.manual_seed(base_seed)

        def _worker_init(worker_id: int) -> None:
            import random as _random

            s = (base_seed + worker_id) % (2**31 - 1)
            np.random.seed(s)
            _random.seed(s)

        loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            persistent_workers=nw > 0,
            drop_last=True,
            generator=loader_gen,
            worker_init_fn=_worker_init if nw > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

        for member_idx, model in enumerate(self.models):
            opt_name = getattr(self.train_config, "optimizer", "adamw").lower()
            if opt_name == "muon":
                from muon import SingleDeviceMuonWithAuxAdam

                params_2d = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
                params_1d = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
                optimizer = SingleDeviceMuonWithAuxAdam(
                    [
                        dict(
                            params=params_2d,
                            use_muon=True,
                            lr=self.train_config.lr,
                            weight_decay=self.train_config.weight_decay,
                            momentum=0.95,
                        ),
                        dict(
                            params=params_1d,
                            use_muon=False,
                            lr=self.train_config.lr,
                            weight_decay=self.train_config.weight_decay,
                            betas=(0.9, 0.95),
                            eps=1e-10,
                        ),
                    ]
                )
            else:
                opt_cls = torch.optim.Adam if opt_name == "adam" else torch.optim.AdamW
                optimizer = opt_cls(
                    model.parameters(),
                    lr=self.train_config.lr,
                    weight_decay=self.train_config.weight_decay,
                )
            # LR schedule is a tunable HP axis (train_config.lr_schedule). Returns at most
            # one of (per-batch scheduler, per-epoch epoch_scheduler).
            scheduler, epoch_scheduler = build_lr_schedulers(
                optimizer, self.train_config, len(loader), opt_name
            )
            if self.task_mode == "yeast":
                criterion: nn.Module = YeastKLLoss()
            elif self.multitask:
                criterion = NaNMaskedMSELoss()
            else:
                loss_name = getattr(self.train_config, "loss", "mse")
                if loss_name in ("huber", "smoothl1"):
                    criterion = nn.HuberLoss(delta=getattr(self.train_config, "huber_delta", 1.0))
                else:
                    criterion = nn.MSELoss()
            # Build optional training-time EvoAug transform
            extra_aug = None
            if getattr(self.train_config, "evoaug_intensity", None):
                from models.evoaug_transform import EvoAugTransform

                # Seed from the run seed (+ member index) so augmentation is
                # reproducible per hp.seed and diverse across ensemble members,
                # instead of a fixed seed that made every run see identical augs.
                extra_aug = EvoAugTransform(
                    intensity=self.train_config.evoaug_intensity,
                    apply_prob=self.train_config.evoaug_prob,
                    seed=(base_seed + member_idx) % (2**31 - 1),
                    target_length=self.sequence_length,
                )

            history = train_model_optimized(
                model=model,
                train_loader=loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=self.train_config.epochs,
                device=self.device,
                scheduler=scheduler,
                epoch_scheduler=epoch_scheduler,
                checkpoint_dir=None,
                use_reverse_complement=bool(
                    getattr(self.train_config, "use_reverse_complement", False)
                ),
                early_stopping_patience=self.train_config.early_stopping_patience,
                min_delta=self.train_config.min_delta,
                metric_for_best="pearson_r",
                use_amp=True,
                use_compile=self.train_config.use_compile,
                shift_aug=self.train_config.shift_aug,
                max_shift=self.train_config.max_shift,
                multitask=self.multitask,
                epoch_callback=epoch_callback,
                extra_augment=extra_aug,
            )
            self.histories.append(history)
