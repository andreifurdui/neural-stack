"""Callback system for training loop extensibility.

Callbacks provide hooks into the training loop for infrastructure concerns
like logging, checkpointing, and early stopping. They are composable and
each callback handles a single responsibility.

Example:
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        callbacks=[
            PrintCallback(),
            CheckpointCallback("./checkpoints", save_best=True),
            EarlyStoppingCallback(patience=5),
        ],
    )
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import re
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from neural_stack.training.trainer import Trainer


# =============================================================================
# Training State
# =============================================================================

@dataclass
class TrainState:
    """Mutable training state passed to callbacks.

    This dataclass holds the current state of training, including
    epoch/step counters, metrics, and control flags.

    Attributes:
        epoch: Current epoch (0-indexed).
        global_step: Total training steps completed.
        batch_idx: Current batch index within epoch.
        loss: Loss from the most recent batch.
        grad_norm: Gradient norm from the most recent batch.
        metrics: Dictionary of metrics from current epoch.
        best_metric: Best value of monitored metric so far.
        best_epoch: Epoch when best metric was achieved.
        should_stop: Flag to signal early stopping.
    """
    epoch: int = 0
    global_step: int = 0
    batch_idx: int = 0

    loss: float = 0.0
    grad_norm: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    best_metric: float = 0.0
    best_epoch: int = 0

    should_stop: bool = False

    best_checkpoint: Optional[Path] = None

# =============================================================================
# Base Callback
# =============================================================================

class Callback(ABC):
    """Base callback class with lifecycle hooks.

    Override any of the hook methods to customize behavior at
    different points in the training loop.

    Attributes:
        trainer: Reference to the Trainer instance (set automatically).
    """

    trainer: Optional[Trainer] = None

    def set_trainer(self, trainer: Trainer) -> None:
        """Called when callback is attached to trainer.

        Args:
            trainer: The Trainer instance this callback is attached to.
        """
        self.trainer = trainer

    def on_train_begin(self, state: TrainState) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, state: TrainState) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, state: TrainState) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, state: TrainState) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, state: TrainState) -> None:
        """Called before each training batch."""
        pass

    def on_batch_end(self, state: TrainState) -> None:
        """Called after each training batch."""
        pass


# =============================================================================
# Callback List Container
# =============================================================================

class CallbackList:
    """Container for managing multiple callbacks.

    Dispatches lifecycle events to all registered callbacks in order.

    Args:
        callbacks: List of Callback instances.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def set_trainer(self, trainer: Trainer) -> None:
        """Set trainer reference on all callbacks."""
        for cb in self.callbacks:
            cb.set_trainer(trainer)

    def on_train_begin(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_epoch_begin(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(state)

    def on_epoch_end(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(state)

    def on_batch_begin(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(state)

    def on_batch_end(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(state)


# =============================================================================
# Built-in Callbacks
# =============================================================================

class PrintCallback(Callback):
    """Simple console logging callback.

    Prints epoch metrics to stdout at the end of each epoch.
    """

    def on_epoch_end(self, state: TrainState) -> None:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(state.metrics.items())
        )
        print(f"Epoch {state.epoch + 1}: {metrics_str}")


class CheckpointCallback(Callback):
    """Save model checkpoints during training.

    Supports saving:
    - Best model (based on monitored metric)
    - Periodic checkpoints every N epochs

    Args:
        checkpoint_dir: Directory to save checkpoints.
        save_best: Whether to save best model based on monitored metric.
        monitor: Metric name to monitor for best model (e.g., "val/accuracy").
        mode: "max" if higher is better, "min" if lower is better.
        save_every: Save checkpoint every N epochs (None to disable).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_best: bool = True,
        monitor: str = "val/accuracy",
        mode: str = "max",
        save_every: Optional[int] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir, "checkpoints")
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every

        self.best_value = float("-inf") if mode == "max" else float("inf")

    def on_train_begin(self, state: TrainState) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state.best_checkpoint = self.checkpoint_dir / "best.pt"

    def on_epoch_end(self, state: TrainState) -> None:
        # Save periodic checkpoint
        if self.save_every and (state.epoch + 1) % self.save_every == 0:
            self._save_checkpoint(state, f"epoch_{state.epoch + 1}.pt")

        # Save best checkpoint
        if self.save_best and self.monitor in state.metrics:
            current = state.metrics[self.monitor]
            is_best = (
                (self.mode == "max" and current > self.best_value) or
                (self.mode == "min" and current < self.best_value)
            )
            if is_best:
                self.best_value = current
                state.best_metric = current
                state.best_epoch = state.epoch
                self._save_checkpoint(state, "best.pt")
                print(f"  -> New best {self.monitor}: {current:.4f}")

    def on_train_end(self, state: TrainState) -> None:
        # Save final checkpoint
        self._save_checkpoint(state, "last.pt")

    def _save_checkpoint(self, state: TrainState, filename: str) -> None:
        """Save a checkpoint to disk."""
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": self.trainer.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "metrics": state.metrics,
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
        }

        if self.trainer.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.trainer.lr_scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to monitor (e.g., "val/loss").
        mode: "min" if lower is better, "max" if higher is better.
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def on_epoch_end(self, state: TrainState) -> None:
        if self.monitor not in state.metrics:
            return

        current = state.metrics[self.monitor]

        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered at epoch {state.epoch + 1}")
                state.should_stop = True


class WandbCallback(Callback):
    """Weights & Biases logging callback.

    Logs training metrics, validation metrics, and learning rate to W&B.
    Requires wandb to be installed.

    Args:
        project: W&B project name.
        name: Run name (optional).
        group: Run group (optional).
        notes: Run notes (optional).
        tags: List of tags for the run (optional).
        config: Configuration dict to log (optional).
        dir: Directory for W&B files (optional).
        log_every: Log batch metrics every N steps (1 = every step).
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        log_every: int = 1,
        log_best_checkpoint: bool = True
    ):
        self.project = project
        self.name = name
        self.group = group
        self.notes = notes
        self.tags = tags
        self.config = config
        self.dir = dir
        self.log_every = log_every
        self.log_best_checkpoint = log_best_checkpoint
        self.run = None

    def on_train_begin(self, state: TrainState) -> None:
        try:
            import wandb
            self.run = wandb.init(
                project=self.project,
                name=self.name,
                group=self.group,
                notes=self.notes,
                tags=self.tags,
                config=self.config,
                dir=self.dir,
            )
        except ImportError:
            print("Warning: wandb not installed. WandbCallback disabled.")
            self.run = None

    def on_batch_end(self, state: TrainState) -> None:
        if self.run is None:
            return

        if state.global_step % self.log_every == 0:
            self.run.log({
                "train/loss": state.loss,
                "train/lr": self.trainer.optimizer.param_groups[0]["lr"],
                "train/grad_norm": state.grad_norm
            }, step=state.global_step)

    def on_epoch_end(self, state: TrainState) -> None:
        if self.run is None:
            return

        self.run.log(state.metrics, step=state.global_step)

    def on_train_end(self, state: TrainState) -> None:
        if self.run is None:
            return
        
        if self.log_best_checkpoint:
            self.run.save(state.best_checkpoint)
        
        self.run.finish()


class LRSchedulerCallback(Callback):
    """Learning rate scheduler callback.

    Steps the scheduler at the appropriate time (epoch or batch level).

    Args:
        step_on_batch: If True, step scheduler after each batch.
                      If False, step after each epoch (default).
    """

    def __init__(self, step_on_batch: bool = False):
        self.step_on_batch = step_on_batch

    def on_batch_end(self, state: TrainState) -> None:
        if self.step_on_batch and self.trainer.lr_scheduler is not None:
            self.trainer.lr_scheduler.step()

    def on_epoch_end(self, state: TrainState) -> None:
        if not self.step_on_batch and self.trainer.lr_scheduler is not None:
            self.trainer.lr_scheduler.step()

class LRSchedulerIterPatchCallback(Callback):
    def on_train_begin(self, state: TrainState) -> None:
        num_iters_per_epoch = len(self.trainer.train_loader)
        try:
            self.trainer.lr_scheduler.set_iters_per_epoch(num_iters_per_epoch)
            print(f"Patched LR scheduler {self.trainer.lr_scheduler.__class__.__name__} with {num_iters_per_epoch} iters/epoch.")
        except AttributeError:
            print(f"Warning: LR scheduler {self.trainer.lr_scheduler.__class__.__name__} does not support set_iters_per_epoch().")
            pass

class ProgressCallback(Callback):
    """Progress bar callback using tqdm.

    Shows training progress with current loss. This is an alternative
    to the progress bar built into the Trainer.

    Args:
        show_epoch_pbar: Show progress bar for epochs.
        show_batch_pbar: Show progress bar for batches.
    """

    def __init__(self, show_epoch_pbar: bool = True, show_batch_pbar: bool = True):
        self.show_epoch_pbar = show_epoch_pbar
        self.show_batch_pbar = show_batch_pbar
        self._epoch_pbar = None
        self._batch_pbar = None

    def on_train_begin(self, state: TrainState) -> None:
        if self.show_epoch_pbar:
            try:
                from tqdm import tqdm
                self._epoch_pbar = tqdm(
                    total=self.trainer.num_epochs,
                    desc="Training",
                    unit="epoch",
                )
            except ImportError:
                pass

    def on_epoch_begin(self, state: TrainState) -> None:
        if self.show_batch_pbar:
            try:
                from tqdm import tqdm
                self._batch_pbar = tqdm(
                    total=len(self.trainer.train_loader),
                    desc=f"Epoch {state.epoch + 1}",
                    unit="batch",
                    leave=False,
                )
            except ImportError:
                pass

    def on_batch_end(self, state: TrainState) -> None:
        if self._batch_pbar is not None:
            self._batch_pbar.update(1)
            self._batch_pbar.set_postfix(loss=f"{state.loss:.4f}")

    def on_epoch_end(self, state: TrainState) -> None:
        if self._batch_pbar is not None:
            self._batch_pbar.close()
            self._batch_pbar = None

        if self._epoch_pbar is not None:
            self._epoch_pbar.update(1)

    def on_train_end(self, state: TrainState) -> None:
        if self._epoch_pbar is not None:
            self._epoch_pbar.close()

class FreezeCallback(Callback):
    """Unified callback for freezing model layers with multiple strategies.

    Supports depth-based, pattern-based, and type-based freezing.
    Automatically recreates optimizer to reflect trainable parameter changes.

    Args:
        freeze_first_n: Freeze first N modules in ModuleList structures.
        freeze_last_n: Freeze last N modules in ModuleList structures.
        freeze_embeddings: Freeze all embedding layers.
        freeze_head: Freeze classification head.
        freeze_patterns: List of regex patterns for layers to freeze.
        unfreeze_patterns: List of regex patterns for layers to unfreeze (takes precedence).
        freeze_types: List of module types to freeze (e.g., [nn.Dropout, nn.LayerNorm]).
        keep_batchnorm_trainable: Keep BatchNorm layers trainable even if frozen.
        verbose: Print detailed summary of frozen parameters.

    Example:
        # Simple: Freeze first 3 transformer blocks
        FreezeCallback(freeze_first_n=3)

        # Transfer learning: Freeze backbone, train head
        FreezeCallback(
            freeze_embeddings=True,
            freeze_first_n=12,
            unfreeze_patterns=[r'.*head.*']
        )
    """

    def __init__(
        self,
        freeze_first_n: Optional[int] = None,
        freeze_last_n: Optional[int] = None,
        freeze_embeddings: bool = False,
        freeze_head: bool = False,
        freeze_patterns: Optional[List[str]] = None,
        unfreeze_patterns: Optional[List[str]] = None,
        freeze_types: Optional[List] = None,
        keep_batchnorm_trainable: bool = True,
        verbose: bool = True,
    ):
        self.freeze_first_n = freeze_first_n
        self.freeze_last_n = freeze_last_n
        self.freeze_embeddings = freeze_embeddings
        self.freeze_head = freeze_head
        self.freeze_patterns = freeze_patterns or []
        self.unfreeze_patterns = unfreeze_patterns or []
        self.freeze_types = freeze_types or []
        self.keep_batchnorm_trainable = keep_batchnorm_trainable
        self.verbose = verbose

    def on_train_begin(self, state: TrainState) -> None:
        """Apply freezing at the start of training."""
        model = self.trainer.model

        # Apply freezing strategies in order
        self._freeze_by_depth(model)
        self._freeze_by_pattern(model, re)
        self._freeze_by_type(model)

        # Special handling
        if self.keep_batchnorm_trainable:
            self._handle_batchnorm(model, nn)

        # Recreate optimizer with new trainable params
        self._recreate_optimizer(self.trainer)

        # Report
        if self.verbose:
            self._report_frozen_params(model)

    def _freeze_by_depth(self, model) -> None:
        """Freeze layers by depth (for ModuleList structures)."""
        import torch.nn as nn

        # Find ModuleList containers (e.g., transformer_stack)
        for name, module in model.named_modules():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                # Freeze first N
                if self.freeze_first_n is not None:
                    for i in range(min(self.freeze_first_n, len(module))):
                        for param in module[i].parameters():
                            param.requires_grad = False

                # Freeze last N
                if self.freeze_last_n is not None:
                    for i in range(max(0, len(module) - self.freeze_last_n), len(module)):
                        for param in module[i].parameters():
                            param.requires_grad = False

        # Freeze embeddings (common pattern in transformers)
        if self.freeze_embeddings:
            for name, module in model.named_modules():
                if 'embed' in name.lower():
                    for param in module.parameters():
                        param.requires_grad = False

        # Freeze head (classification layer)
        if self.freeze_head:
            for name, module in model.named_modules():
                if 'head' in name.lower() or 'classifier' in name.lower():
                    for param in module.parameters():
                        param.requires_grad = False

    def _freeze_by_pattern(self, model, re) -> None:
        """Freeze/unfreeze using regex patterns."""
        freeze_regexes = [re.compile(p) for p in self.freeze_patterns]
        unfreeze_regexes = [re.compile(p) for p in self.unfreeze_patterns]

        for name, param in model.named_parameters():
            # Check freeze patterns
            if any(regex.match(name) for regex in freeze_regexes):
                param.requires_grad = False

            # Unfreeze patterns take precedence
            if any(regex.match(name) for regex in unfreeze_regexes):
                param.requires_grad = True

    def _freeze_by_type(self, model) -> None:
        """Freeze all parameters in modules of specified types."""
        if not self.freeze_types:
            return

        for module in model.modules():
            if any(isinstance(module, freeze_type) for freeze_type in self.freeze_types):
                for param in module.parameters():
                    param.requires_grad = False

    def _handle_batchnorm(self, model, nn) -> None:
        """Keep BatchNorm layers trainable even if frozen."""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters():
                    param.requires_grad = True

    def _recreate_optimizer(self, trainer) -> None:
        """Recreate optimizer with current trainable parameters.

        Critical: Optimizers capture trainable params at creation time.
        When freezing/unfreezing, we MUST recreate the optimizer.
        """
        # Extract optimizer config
        OptimClass = type(trainer.optimizer)
        state_dict = trainer.optimizer.state_dict()

        # Get current optimizer hyperparams (only valid constructor args)
        old_param_group = trainer.optimizer.param_groups[0]

        # Whitelist of common optimizer hyperparameters
        # These are the typical constructor args for PyTorch optimizers
        valid_hyperparams = {
            'lr', 'weight_decay', 'momentum', 'dampening', 'nesterov',  # SGD
            'betas', 'eps', 'amsgrad',  # Adam/AdamW
            'alpha', 'centered',  # RMSprop
            'rho',  # Adadelta
        }

        optim_kwargs = {
            k: v for k, v in old_param_group.items()
            if k in valid_hyperparams
        }

        # Get only trainable params
        trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]

        # Recreate optimizer
        trainer.optimizer = OptimClass(trainable_params, **optim_kwargs)

        # Patch scheduler's optimizer reference to point to new optimizer
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.optimizer = trainer.optimizer

        # Try to restore optimizer state for params that are still being trained
        # (This will partially fail if params changed, which is expected)
        try:
            trainer.optimizer.load_state_dict(state_dict)
        except (KeyError, ValueError):
            # Expected when trainable params change
            if self.verbose:
                print("  Note: Optimizer state reset due to parameter changes")

    def _report_frozen_params(self, model) -> None:
        """Print detailed summary of frozen vs trainable parameters."""
        frozen_params = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        frozen_count = sum(p.numel() for _, p in frozen_params)
        trainable_count = sum(p.numel() for _, p in trainable_params)
        total_count = frozen_count + trainable_count

        print("\n" + "="*60)
        print("Model Freezing Summary")
        print("="*60)
        print(f"Frozen:     {frozen_count:>12,} params ({frozen_count/total_count*100:>5.1f}%)")
        print(f"Trainable:  {trainable_count:>12,} params ({trainable_count/total_count*100:>5.1f}%)")
        print(f"Total:      {total_count:>12,} params")

        if frozen_params:
            print(f"\nFrozen layers ({len(frozen_params)}):")
            for name, param in frozen_params:
                print(f"  - {name:<50} {param.numel():>10,} params")

        print("="*60 + "\n")