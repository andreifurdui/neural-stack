"""Base Trainer with callback system and method override hooks.

The Trainer receives pre-built components (model, optimizer, dataloaders)
and orchestrates the training loop. It is extensible via:

- **Callbacks**: For infrastructure concerns (logging, checkpointing, early stopping)
- **Method overrides**: For training recipes (distillation, fine-tuning, semi-supervision)

Example:
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=[
            PrintCallback(),
            CheckpointCallback("./checkpoints"),
        ],
        num_epochs=25,
    )
    results = trainer.fit()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_stack.training.callbacks import Callback, CallbackList, TrainState


class Trainer:
    """Base trainer with callback system and override hooks.

    Receives pre-built components and orchestrates the training loop.
    The training loop includes clear hook points for callbacks and
    virtual methods that can be overridden in subclasses for custom
    training recipes.

    Args:
        model: PyTorch model to train.
        optimizer: Configured optimizer.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        criterion: Loss function (defaults to CrossEntropyLoss).
        lr_scheduler: Optional learning rate scheduler.
        callbacks: List of callback instances.
        device: Device to train on ("cuda", "cpu", "mps").
        num_epochs: Number of training epochs.

    Attributes:
        model: The model being trained.
        optimizer: The optimizer.
        train_loader: Training data loader.
        val_loader: Validation data loader (may be None).
        criterion: Loss function.
        lr_scheduler: LR scheduler (may be None).
        device: Training device.
        num_epochs: Total epochs to train.
        state: Current TrainState instance.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = "cuda",
        num_epochs: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = torch.device(device)
        self.num_epochs = num_epochs

        # Move model to device
        self.model.to(self.device)

        # Initialize callback list
        self.callbacks = CallbackList(callbacks)
        self.callbacks.set_trainer(self)

        # Training state
        self.state = TrainState()

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def fit(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary containing:
                - best_metric: Best value of monitored metric
                - best_epoch: Epoch when best metric was achieved
                - final_epoch: Last completed epoch
        """
        self.callbacks.on_train_begin(self.state)

        try:
            for epoch in range(self.num_epochs):
                self.state.epoch = epoch
                self.state.metrics = {}  # Reset epoch metrics

                if self.state.should_stop:
                    break

                self._run_epoch()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        self.callbacks.on_train_end(self.state)

        return {
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "final_epoch": self.state.epoch,
        }

    def _run_epoch(self) -> None:
        """Run a single training epoch."""
        self.callbacks.on_epoch_begin(self.state)

        # Training phase
        train_metrics = self._train_epoch()
        self.state.metrics.update({f"train/{k}": v for k, v in train_metrics.items()})

        # Validation phase
        if self.val_loader is not None:
            val_metrics = self._validate()
            self.state.metrics.update({f"val/{k}": v for k, v in val_metrics.items()})

        # LR scheduler step (epoch-level)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.callbacks.on_epoch_end(self.state)

    # =========================================================================
    # Override Points for Training Recipes
    # =========================================================================

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Override this method for custom epoch-level training logic.
        The default implementation iterates over batches and calls
        train_step() for each.

        Returns:
            Dictionary of training metrics for this epoch.
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.state.batch_idx = batch_idx
            self.callbacks.on_batch_begin(self.state)

            # Core training step (overridable)
            loss, batch_metrics = self.train_step(batch)

            self.state.loss = loss.item()
            self.state.global_step += 1

            # Accumulate metrics
            total_loss += loss.item()
            if "correct" in batch_metrics:
                total_correct += batch_metrics["correct"]
                total_samples += batch_metrics["total"]

            self.callbacks.on_batch_end(self.state)

        # Compute epoch metrics
        metrics = {"loss": total_loss / len(self.train_loader)}
        if total_samples > 0:
            metrics["accuracy"] = total_correct / total_samples

        return metrics

    def train_step(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single training step.

        Override this method for custom forward/backward logic.
        This is the primary extension point for training recipes
        like distillation or semi-supervised learning.

        Args:
            batch: A batch from the dataloader.

        Returns:
            Tuple of (loss tensor, metrics dict).
            The metrics dict should contain at minimum:
                - "correct": Number of correct predictions
                - "total": Total samples in batch
        """
        # Unpack batch
        inputs, targets = self.unpack_batch(batch)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute batch metrics
        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            correct = (predictions == targets).sum().item()

        return loss, {"correct": correct, "total": len(targets)}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Override for custom forward logic (e.g., multi-input models).

        Args:
            inputs: Input tensor.

        Returns:
            Model outputs.
        """
        return self.model(inputs)

    def unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch into inputs and targets.

        Override for custom data formats.

        Args:
            batch: A batch from the dataloader.

        Returns:
            Tuple of (inputs, targets).
        """
        # Default: assume tuple/list of (inputs, targets)
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        # Dict format
        if isinstance(batch, dict):
            return batch["inputs"], batch["targets"]
        raise ValueError(f"Cannot unpack batch of type {type(batch)}")

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss.

        Override for custom loss computation (e.g., distillation loss,
        multi-task loss, regularization).

        Args:
            outputs: Model outputs.
            targets: Ground truth targets.

        Returns:
            Loss tensor (scalar).
        """
        return self.criterion(outputs, targets)

    def _validate(self) -> Dict[str, float]:
        """Run validation.

        Override for custom validation logic.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self.unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == targets).sum().item()
                total_samples += len(targets)

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": total_correct / total_samples,
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint and restore model/optimizer state.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary with metadata.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)
        self.state.best_metric = checkpoint.get("best_metric", 0.0)
        self.state.best_epoch = checkpoint.get("best_epoch", 0)

        return checkpoint

    def get_lr(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate from the first param group.
        """
        return self.optimizer.param_groups[0]["lr"]
