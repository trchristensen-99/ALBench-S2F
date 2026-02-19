"""AlphaGenome wrapper scaffold for integration with ALBench model stack."""

from __future__ import annotations

import torch
from torch import nn


class AlphaGenomeWrapper(nn.Module):
    """PyTorch module wrapper placeholder for AlphaGenome encoder/head usage.

    This class exists to stabilize import paths while AlphaGenome integration
    is finalized in JAX/Haiku training scripts.
    """

    def __init__(self, output_dim: int = 1) -> None:
        """Initialize wrapper metadata.

        Args:
            output_dim: Number of output targets to predict.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Raises:
            NotImplementedError: Always, until integration is implemented.
        """
        raise NotImplementedError(
            "AlphaGenomeWrapper forward pass is not implemented yet. "
            "Use experiments/train_oracle_alphagenome.py for current AlphaGenome training."
        )
