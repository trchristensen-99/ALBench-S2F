"""
Utility functions for loss computation, especially for yeast bin-based classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_to_bin_probabilities(
    continuous_values: torch.Tensor, num_bins: int = 18, min_val: float = 0.0, max_val: float = 17.0
) -> torch.Tensor:
    """
    Convert continuous expression values to bin probability distributions.

    Uses soft assignment: probability is split between the two adjacent bins
    proportional to proximity. For exact integer values this reduces to one-hot.
    For non-integer values (e.g. oracle pseudolabels), this preserves inter-bin
    information that hard rounding would destroy.

    Args:
        continuous_values: Continuous expression values of shape (batch_size,)
        num_bins: Number of bins (18 for yeast)
        min_val: Minimum expression value (0 for yeast)
        max_val: Maximum expression value (17 for yeast)

    Returns:
        Bin probability distributions of shape (batch_size, num_bins)
    """
    batch_size = continuous_values.shape[0]
    device = continuous_values.device

    # Clamp values to valid range
    continuous_values = torch.clamp(continuous_values, min_val, max_val)

    floor_vals = torch.floor(continuous_values).long()
    ceil_vals = torch.clamp(floor_vals + 1, max=num_bins - 1)
    frac = (continuous_values - floor_vals.float()).unsqueeze(1)

    bin_probs = torch.zeros(batch_size, num_bins, device=device)
    bin_probs.scatter_add_(1, floor_vals.unsqueeze(1), 1.0 - frac)
    bin_probs.scatter_add_(1, ceil_vals.unsqueeze(1), frac)

    return bin_probs


class NaNMaskedMSELoss(nn.Module):
    """MSE loss that ignores NaN values in targets.

    Designed for multi-task training where some cell-type labels may be
    missing (NaN) for certain sequences. Computes MSE only over valid
    (non-NaN) entries, averaged across all valid entries.

    Expects predictions and targets of shape (batch, n_tasks).
    """

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute NaN-masked MSE loss.

        Args:
            predictions: Model outputs of shape (batch, n_tasks).
            targets: Labels of shape (batch, n_tasks), may contain NaN.

        Returns:
            Scalar loss averaged over non-NaN entries.
        """
        mask = ~torch.isnan(targets)
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        return F.mse_loss(predictions[mask], targets[mask])


class YeastKLLoss(nn.Module):
    """
    KL divergence loss for yeast 18-bin classification.

    This loss:
    1. Gets logits from the model (before SoftMax)
    2. Converts continuous labels to bin probabilities
    3. Applies log_softmax to logits
    4. Computes KL divergence
    """

    def __init__(self, reduction: str = "batchmean"):
        super().__init__()
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)

    def forward(self, model_logits: torch.Tensor, continuous_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            model_logits: Raw logits from model of shape (batch_size, 18)
            continuous_targets: Continuous expression values of shape (batch_size,)

        Returns:
            KL divergence loss
        """
        # Convert continuous targets to bin probabilities
        target_probs = continuous_to_bin_probabilities(
            continuous_targets, num_bins=18, min_val=0.0, max_val=17.0
        )

        # Apply log_softmax to logits (required for KL divergence)
        log_probs = F.log_softmax(model_logits, dim=1)

        # Compute KL divergence: KL(target || predicted)
        loss = self.kl_div(log_probs, target_probs)

        return loss
