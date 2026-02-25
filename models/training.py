"""
Optimized training utilities with mixed precision and torch.compile support.

This module extends the base training.py with:
- Automatic Mixed Precision (AMP) for 2x speedup
- torch.compile() for 10-30% additional speedup
- Backward compatible with existing code
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss_utils import YeastKLLoss
from .training_base import compute_metrics, evaluate


def train_epoch_optimized(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
    use_reverse_complement: bool = False,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch with optional mixed precision.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Optional learning rate scheduler (called after each batch)
        use_reverse_complement: Whether to average predictions with reverse complement
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    # Initialize scaler for mixed precision if on CUDA
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (sequences, targets) in enumerate(pbar):
        sequences = sequences.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward and backward pass with optional mixed precision
        if scaler.is_enabled():
            with autocast("cuda"):
                # For yeast: get logits and use KL divergence
                # For K562: get predictions and use MSE
                if hasattr(model, "task_mode") and model.task_mode == "yeast":
                    logits = model.get_logits(sequences)
                    predictions = model(sequences)  # Weighted average for metrics
                    loss = criterion(logits, targets)
                else:
                    predictions = model(sequences)
                    loss = criterion(predictions, targets)

            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_ran = scaler.get_scale() >= scale_before
        else:
            # For yeast: get logits and use KL divergence
            # For K562: get predictions and use MSE
            if hasattr(model, "task_mode") and model.task_mode == "yeast":
                logits = model.get_logits(sequences)
                predictions = model(sequences)  # Weighted average for metrics
                loss = criterion(logits, targets)
            else:
                predictions = model(sequences)
                loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            optimizer_step_ran = True

        # Update scheduler if provided
        if scheduler is not None and optimizer_step_ran:
            scheduler.step()

        # Track metrics
        total_loss += loss.item()
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    metrics["loss"] = avg_loss

    return metrics


def train_model_optimized(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
    use_reverse_complement: bool = True,
    early_stopping_patience: Optional[int] = None,
    metric_for_best: str = "pearson_r",
    use_amp: bool = True,
    use_compile: bool = True,
) -> Dict[str, Any]:
    """
    Train a model with validation, checkpointing, and optimizations.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of epochs to train
        device: Device to train on
        scheduler: Optional learning rate scheduler
        checkpoint_dir: Directory to save checkpoints
        use_reverse_complement: Whether to average predictions with reverse complement
        early_stopping_patience: Stop if no improvement for this many epochs
        metric_for_best: Metric to use for saving best model ('pearson_r', 'spearman_r', or 'loss')
        use_amp: Whether to use automatic mixed precision (FP16) training
        use_compile: Whether to use torch.compile() for optimization

    Returns:
        Dictionary with training history
    """
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Compile model if requested and supported
    original_model = model
    if use_compile and hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="max-autotune")
            print("✓ Model compiled successfully!")
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}")
            print("Continuing without compilation...")
            model = original_model
    elif use_compile and device.type != "cuda":
        print("Note: torch.compile() skipped (not on CUDA)")
    elif use_compile:
        print("Warning: torch.compile() requested but not available (requires PyTorch 2.0+)")

    if use_amp and device.type == "cuda":
        print("✓ Using automatic mixed precision (FP16)")
    elif use_amp:
        print("Note: AMP skipped (not on CUDA)")

    history = {
        "train_loss": [],
        "train_pearson_r": [],
        "train_spearman_r": [],
        "val_loss": [],
        "val_pearson_r": [],
        "val_spearman_r": [],
        "learning_rates": [],
        "use_amp": use_amp and device.type == "cuda",
        "use_compile": use_compile and hasattr(torch, "compile"),
    }

    best_metric = -float("inf") if metric_for_best != "loss" else float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"Training on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("-" * 60)

    # Resume from checkpoint if exists
    start_epoch = 0
    if checkpoint_dir is not None and (checkpoint_dir / "last_model.pt").exists():
        last_ckpt_path = checkpoint_dir / "last_model.pt"
        print(f"Resuming from checkpoint: {last_ckpt_path}")
        try:
            # Determine if we should load into original model or compiled model
            # Base class load_checkpoint handles state_dict loading
            load_model = original_model if use_compile else model

            # Use load_checkpoint from SequenceModel base class (or manual load)
            if hasattr(load_model, "load_checkpoint"):
                ckpt = load_model.load_checkpoint(str(last_ckpt_path))
            else:
                ckpt = torch.load(str(last_ckpt_path), map_location=device)
                load_model.load_state_dict(ckpt["model_state_dict"])

            # Restore training state
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            # Restore metrics and loop state
            start_epoch = ckpt.get("epoch", -1) + 1
            if "history" in ckpt:
                history = ckpt["history"]
            if "best_metric" in ckpt:
                best_metric = ckpt["best_metric"]
            if "best_epoch" in ckpt:
                best_epoch = ckpt["best_epoch"]
            if "patience_counter" in ckpt:
                patience_counter = ckpt["patience_counter"]

            print(f"✓ Resumed successfully from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠ Failed to resume from checkpoint: {e}")
            print("Starting from scratch...")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        # Train with optimizations
        train_metrics = train_epoch_optimized(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            use_reverse_complement=use_reverse_complement,
            use_amp=use_amp and device.type == "cuda",
        )

        # Validate (use standard evaluate function)
        val_metrics = evaluate(
            model, val_loader, criterion, device, use_reverse_complement=use_reverse_complement
        )

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_pearson_r"].append(train_metrics["pearson_r"])
        history["train_spearman_r"].append(train_metrics["spearman_r"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_pearson_r"].append(val_metrics["pearson_r"])
        history["val_spearman_r"].append(val_metrics["spearman_r"])

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/pearson_r": train_metrics["pearson_r"],
                    "train/spearman_r": train_metrics["spearman_r"],
                    "val/loss": val_metrics["loss"],
                    "val/pearson_r": val_metrics["pearson_r"],
                    "val/spearman_r": val_metrics["spearman_r"],
                    "lr": current_lr,
                }
            )

        epoch_time = time.time() - epoch_start_time

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  LR: {current_lr:.6f}")
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Pearson R: {train_metrics['pearson_r']:.4f}, "
            f"Spearman R: {train_metrics['spearman_r']:.4f}"
        )
        print(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Pearson R: {val_metrics['pearson_r']:.4f}, "
            f"Spearman R: {val_metrics['spearman_r']:.4f}"
        )

        # Check if this is the best model
        if metric_for_best == "loss":
            current_metric = val_metrics["loss"]
            is_best = current_metric < best_metric
        else:
            current_metric = val_metrics[metric_for_best]
            is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            if checkpoint_dir is not None:
                best_path = checkpoint_dir / "best_model.pt"
                # Save the original model (not compiled version)
                save_model = original_model if use_compile else model
                save_model.save_checkpoint(
                    str(best_path),
                    epoch=epoch,
                    optimizer_state_dict=optimizer.state_dict(),
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
                print(f"  → Saved best model (val {metric_for_best}: {best_metric:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best {metric_for_best}: {best_metric:.4f} at epoch {best_epoch + 1}")
            break

        # Save last model state for resuming
        if checkpoint_dir is not None:
            last_path = checkpoint_dir / "last_model.pt"
            save_model = original_model if use_compile else model

            # Prepare extra state to save
            extra_state = {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
            }
            if scheduler is not None:
                extra_state["scheduler_state_dict"] = scheduler.state_dict()

            # Use model's save_checkpoint method
            if hasattr(save_model, "save_checkpoint"):
                save_model.save_checkpoint(str(last_path), **extra_state)
            else:
                # Fallback manual save
                torch.save(
                    {"model_state_dict": save_model.state_dict(), **extra_state}, str(last_path)
                )

        print("-" * 60)

    # Save final model
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "final_model.pt"
        save_model = original_model if use_compile else model
        save_model.save_checkpoint(
            str(final_path), epoch=num_epochs - 1, optimizer_state_dict=optimizer.state_dict()
        )

    return history
