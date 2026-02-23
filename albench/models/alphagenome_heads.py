"""Configurable AlphaGenome heads for short-sequence S2F tasks.

This module provides reusable head registration helpers so experiments can
switch head architectures (e.g. ``mlp-512-512`` vs ``pool-flatten``) while
keeping the AlphaGenome encoder frozen.
"""

from __future__ import annotations

from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
from alphagenome.models import dna_output
from alphagenome_ft import (
    CustomHead,
    CustomHeadConfig,
    CustomHeadType,
    register_custom_head,
    templates,
)

HeadArch = Literal[
    "mlp-512-512",
    "pool-flatten",
    "boda-flatten-512-512",
    "boda-sum-512-512",
    "boda-mean-512-512",
    "boda-max-512-512",
    "boda-center-512-512",
]
TaskMode = Literal["yeast", "human"]


class _BaseLossHead(templates.EncoderOnlyHead):
    """Encoder-only base head with task-specific loss helpers."""

    def _task_loss(self, predictions: jnp.ndarray, batch: dict) -> dict[str, jnp.ndarray]:
        targets = batch.get("targets")
        if targets is None:
            return {"loss": jnp.array(0.0, dtype=jnp.float32)}

        task_mode = str(self._metadata.get("task_mode", "yeast"))
        if task_mode == "yeast":
            # Yeast setup uses 18-bin targets and KL/CrossEntropy on logits.
            clipped = jnp.clip(targets, 0.0, 17.0)
            bins = jnp.round(clipped).astype(jnp.int32)
            target_probs = jax.nn.one_hot(bins, self._num_tracks)
            log_probs = jax.nn.log_softmax(predictions, axis=-1)
            return {"loss": -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))}

        # Human/K562 setup uses scalar regression with MSE.
        if predictions.ndim > 1:
            pred = jnp.squeeze(predictions, axis=-1)
        else:
            pred = predictions
        target = jnp.asarray(targets, dtype=jnp.float32)
        return {"loss": jnp.mean((pred - target) ** 2)}


class MLP512512Head(_BaseLossHead):
    """DeepSets architecture: LayerNorm, then per-token 512->512 MLP, then mean-pool."""

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        if not hasattr(embeddings, "encoder_output") or embeddings.encoder_output is None:
            raise ValueError(
                "MLP512512Head requires encoder_output. "
                "Use create_model_with_heads(..., use_encoder_output=True)."
            )

        x = embeddings.encoder_output  # (B, T, 1536)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)
        x = hk.Linear(512, name="hidden1")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512, name="hidden2")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)

        # Pool logits over the sequence dimension (T)
        predictions = jnp.mean(x, axis=1)
        return predictions

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


class PoolFlattenHead(_BaseLossHead):
    """LayerNorm + Pool+flatten encoder representation, then project to outputs.

    Concatenates mean-pooled and max-pooled token features with flattened token
    features to preserve both global and position-aware signals on short inputs.
    """

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        if not hasattr(embeddings, "encoder_output") or embeddings.encoder_output is None:
            raise ValueError(
                "PoolFlattenHead requires encoder_output. "
                "Use create_model_with_heads(..., use_encoder_output=True)."
            )

        x = embeddings.encoder_output  # (B, T, 1536)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)

        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        # Avoid exploding parameter counts on very long inputs, but for T=3 it's fine.
        flat = jnp.reshape(x, (x.shape[0], -1))
        z = jnp.concatenate([mean_pool, max_pool, flat], axis=-1)
        z = hk.Linear(512, name="hidden1")(z)
        z = jax.nn.relu(z)
        z = hk.Linear(256, name="hidden2")(z)
        z = jax.nn.relu(z)
        return hk.Linear(self._num_tracks, name="output")(z)

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


class BodaFlattenHead(_BaseLossHead):
    """Exactly replicates boda 'flatten' pooling: Flattens Spatial -> MLP[512, 512]."""

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        x = embeddings.encoder_output  # (B, T, 1536)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)

        # Boda 'flatten':
        x = jnp.reshape(x, (x.shape[0], -1))

        x = hk.Linear(512, name="hidden1")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512, name="hidden2")(x)
        x = jax.nn.relu(x)

        return hk.Linear(self._num_tracks, name="output")(x)

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


class BodaSumHead(_BaseLossHead):
    """Exactly replicates boda 'sum' pooling: per-position MLP[512, 512] -> Sum Spatial."""

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        x = embeddings.encoder_output  # (B, T, 1536)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)

        x = hk.Linear(512, name="hidden1")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512, name="hidden2")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)

        # Boda 'sum' pools post-MLP:
        return jnp.sum(x, axis=1)

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


class BodaMeanHead(_BaseLossHead):
    """Exactly replicates boda 'mean' pooling: per-position MLP[512, 512] -> Mean Spatial."""

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        x = embeddings.encoder_output
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)
        x = hk.Linear(512, name="hidden1")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512, name="hidden2")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)
        return jnp.mean(x, axis=1)

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


class BodaMaxHead(_BaseLossHead):
    """Exactly replicates boda 'max' pooling: per-position MLP[512, 512] -> Max Spatial."""

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        x = embeddings.encoder_output
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)
        x = hk.Linear(512, name="hidden1")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512, name="hidden2")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)
        return jnp.max(x, axis=1)

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


class BodaCenterHead(_BaseLossHead):
    """Exactly replicates boda 'center' pooling: per-position MLP[512, 512] -> Slice Center."""

    def predict(self, embeddings, organism_index, **kwargs):  # type: ignore[override]
        x = embeddings.encoder_output
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm")(x)
        x = hk.Linear(512, name="hidden1")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512, name="hidden2")(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_tracks, name="output")(x)

        # Boda 'center' extracting raw center-most slice:
        center_idx = x.shape[1] // 2
        return x[:, center_idx, :]

    def loss(self, predictions, batch):  # type: ignore[override]
        return self._task_loss(predictions, batch)


def get_head_class(arch: HeadArch) -> type[CustomHead]:
    """Return a head class for the requested architecture.

    Args:
        arch: Architecture identifier.

    Returns:
        Custom head class compatible with ``alphagenome_ft.register_custom_head``.

    Raises:
        ValueError: If architecture is unsupported.
    """
    if arch == "mlp-512-512":
        return MLP512512Head
    if arch == "pool-flatten":
        return PoolFlattenHead
    if arch == "boda-flatten-512-512":
        return BodaFlattenHead
    if arch == "boda-sum-512-512":
        return BodaSumHead
    if arch == "boda-mean-512-512":
        return BodaMeanHead
    if arch == "boda-max-512-512":
        return BodaMaxHead
    if arch == "boda-center-512-512":
        return BodaCenterHead
    raise ValueError(f"Unsupported AlphaGenome head architecture: {arch}")


def register_s2f_head(
    *,
    head_name: str,
    arch: HeadArch,
    task_mode: TaskMode,
    num_tracks: int,
    output_type: dna_output.OutputType = dna_output.OutputType.RNA_SEQ,
) -> None:
    """Register an ALBench S2F AlphaGenome head with frozen-encoder training.

    Args:
        head_name: Registry name used by ``create_model_with_heads``.
        arch: Head architecture key.
        task_mode: ``yeast`` for binned objective, ``human`` for regression.
        num_tracks: Number of output channels/logits.
        output_type: AlphaGenome output type enum.
    """
    head_class = get_head_class(arch)
    head_config = CustomHeadConfig(
        type=CustomHeadType.GENOME_TRACKS,
        output_type=output_type,
        num_tracks=num_tracks,
        metadata={"task_mode": task_mode},
    )
    register_custom_head(head_name, head_class, head_config)
