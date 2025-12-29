"""Training framework for neural-stack.

This module provides a lightweight, modular training framework with:

- **Type-safe configuration** via dataclasses with YAML serialization
- **Extensible Trainer** with callbacks and method override hooks
- **Component factories** for models, optimizers, and schedulers
- **Data utilities** for common datasets

Quick Start:
    from neural_stack.training import (
        TrainConfig, load_config,
        Trainer, build_from_config, build_dataloaders,
        PrintCallback, CheckpointCallback,
    )

    # Load configuration
    config = load_config("configs/vit_cifar10.yaml")

    # Build components
    components = build_from_config(config)
    train_loader, val_loader = build_dataloaders(config.data)

    # Create trainer with callbacks
    trainer = Trainer(
        model=components["model"],
        optimizer=components["optimizer"],
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=components["criterion"],
        lr_scheduler=components["scheduler"],
        callbacks=[
            PrintCallback(),
            CheckpointCallback(config.checkpoint_dir),
        ],
        device=config.device,
        num_epochs=config.num_epochs,
    )

    # Train
    results = trainer.fit()

For custom training recipes, subclass Trainer and override methods:
    - train_step(): Custom forward/backward logic
    - compute_loss(): Custom loss computation
    - unpack_batch(): Custom data format handling
    - _validate(): Custom validation logic
"""

# Configuration
from neural_stack.training.config import (
    TrainConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataConfig,
    load_config,
    save_config,
    config_to_dict,
)

# Trainer
from neural_stack.training.trainer import Trainer

# Callbacks
from neural_stack.training.callbacks import (
    TrainState,
    Callback,
    CallbackList,
    PrintCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    WandbCallback,
    LRSchedulerCallback,
    LRSchedulerIterPatchCallback,
    ProgressCallback,
)

# Factories
from neural_stack.training.factory import (
    build_model,
    build_optimizer,
    build_scheduler,
    build_criterion,
    build_from_config,
    set_seed,
)

# Data
from neural_stack.training.data import (
    build_datasets,
    build_dataloaders,
    get_dataset_info,
)

__all__ = [
    # Config
    "TrainConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "DataConfig",
    "load_config",
    "save_config",
    "config_to_dict",
    # Trainer
    "Trainer",
    "TrainState",
    # Callbacks
    "Callback",
    "CallbackList",
    "PrintCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "WandbCallback",
    "LRSchedulerCallback",
    "LRSchedulerIterPatchCallback",
    "ProgressCallback",
    # Factories
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_criterion",
    "build_from_config",
    "set_seed",
    # Data
    "build_datasets",
    "build_dataloaders",
    "get_dataset_info",
]
