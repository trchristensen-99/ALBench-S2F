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


class L1KLMixedLoss(nn.Module):
    """L1 + KL divergence mixed loss (matching Malinois paper / boda2).

    Combines L1 regression loss with KL divergence on log-softmax-normalized
    predictions. The KL term is computed across the cell-type output dimension
    (dim=-1), treating the multi-task outputs as a probability distribution.
    This encourages the model to match the relative ranking across cell types,
    not just absolute values.

    For single-output models (n_outputs=1), the KL term is meaningless
    (softmax of a scalar = 1.0). Use multitask=True only with n_outputs >= 2.

    The boda2 paper uses alpha=1.0, beta=5.0 with reduction='mean' (not 'batchmean').

    Args:
        alpha: Weight for L1 component (default 1.0).
        beta: Weight for KL component (default 5.0).
        multitask: If True, compute KL across dim=-1 (cell-type outputs).
            If False, falls back to flattened KL (legacy behavior).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 5.0, multitask: bool = False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.multitask = multitask

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Mask NaN targets for multi-task training
        if self.multitask and targets.dim() == 2:
            nan_mask = torch.isnan(targets)
            if nan_mask.any():
                # For L1: zero out NaN positions
                safe_targets = targets.clone()
                safe_targets[nan_mask] = predictions[nan_mask].detach()
            else:
                safe_targets = targets
        else:
            safe_targets = targets

        # L1 component
        l1_loss = F.l1_loss(predictions, safe_targets)

        # KL divergence component
        if self.multitask and predictions.dim() == 2 and predictions.shape[-1] > 1:
            # boda2 style: log_softmax across cell-type dim (dim=-1)
            pred_log_probs = F.log_softmax(predictions, dim=-1)
            target_log_probs = F.log_softmax(safe_targets, dim=-1)
            # KL(target || pred) with log_target=True
            kl_loss = F.kl_div(pred_log_probs, target_log_probs, reduction="mean", log_target=True)
        else:
            # Legacy: flatten and normalize across all values
            pred_log_probs = F.log_softmax(predictions.view(-1), dim=0).view_as(predictions)
            target_probs = F.softmax(safe_targets.view(-1), dim=0).view_as(safe_targets)
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

        return (self.alpha * l1_loss + self.beta * kl_loss) / (self.alpha + self.beta)


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
