"""Factory functions for building training components from configuration.

This module provides explicit factory functions for constructing models,
optimizers, schedulers, and other training components from configuration
dataclasses. Uses direct if/else logic rather than registry patterns
for explicitness and debuggability.

Example:
    from neural_stack.training import TrainConfig, build_from_config

    config = TrainConfig()
    components = build_from_config(config)

    model = components["model"]
    optimizer = components["optimizer"]
"""

from typing import Dict, Any, Optional

import timm
import torch
import torch.nn as nn

from neural_stack.training.config import (
    TrainConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)


# =============================================================================
# Model Factory
# =============================================================================

def build_model(config: ModelConfig) -> nn.Module:
    """Build a model from configuration.

    Args:
        config: Model configuration specifying architecture and parameters.

    Returns:
        Constructed PyTorch model.

    Raises:
        ValueError: If model name is not recognized.

    Supported models:
        - "vit": Vision Transformer (neural_stack.models.vision_transformer)
    """
    if config.source == "custom":
        if config.name == "vit":
            from neural_stack.models.vision_transformer import VisionTransformer

            return VisionTransformer(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_channels=config.in_channels,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                num_classes=config.num_classes,
                positional_embedding=config.positional_embedding,
                use_cls_token=config.use_cls_token,
            )
        else:
            raise ValueError(
                f"Unknown model: '{config.name}'. "
                f"Supported models: 'vit'"
            )
    elif config.source == "timm":
        try:
            model = timm.create_model(
                config.name,
                pretrained=config.pretrained,
                num_classes=config.num_classes,
            )
            return model
        except Exception as e:
            raise ValueError(
                f"Error creating timm model '{config.name}': {e}"
            ) from e


# =============================================================================
# Optimizer Factory
# =============================================================================

def build_optimizer(
    config: OptimizerConfig,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Build an optimizer from configuration.

    Args:
        config: Optimizer configuration.
        model: Model whose parameters will be optimized.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer name is not recognized.

    Supported optimizers:
        - "adamw": AdamW with weight decay
        - "adam": Adam
        - "sgd": SGD with momentum
    """
    if config.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
        )
    elif config.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: '{config.name}'. "
            f"Supported optimizers: 'adamw', 'adam', 'sgd'"
        )


# =============================================================================
# Scheduler Factory
# =============================================================================

def build_scheduler(
    config: SchedulerConfig,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build a learning rate scheduler from configuration.

    Args:
        config: Scheduler configuration.
        optimizer: Optimizer to schedule.
        num_epochs: Total number of training epochs (used for cosine scheduler).

    Returns:
        Configured scheduler, or None if config.name is "none".

    Raises:
        ValueError: If scheduler name is not recognized.

    Supported schedulers:
        - "step": StepLR - decay by gamma every step_size epochs
        - "cosine": CosineAnnealingLR - cosine annealing to eta_min
        - "cosine_warmup": CosineAnnealingWarmupLR - cosine annealing with warmup
        - "none": No scheduler (returns None)
    """
    if config.name == "none":
        return None
    elif config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    elif config.name == "cosine":
        T_max = config.T_max if config.T_max is not None else num_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.eta_min,
        )
    elif config.name == "cosine_warmup":
        from neural_stack.training.schedulers import CosineAnnealingWarmupLR

        T_max = config.T_max if config.T_max is not None else num_epochs
        return CosineAnnealingWarmupLR(
            optimizer,
            T_max=T_max,
            warmup_epochs=config.warmup_epochs,
            eta_min=config.eta_min,
        )
    else:
        raise ValueError(
            f"Unknown scheduler: '{config.name}'. "
            f"Supported schedulers: 'step', 'cosine', 'cosine_warmup', 'none'"
        )


# =============================================================================
# Criterion Factory
# =============================================================================

def build_criterion(name: str = "cross_entropy") -> nn.Module:
    """Build a loss function.

    Args:
        name: Name of the loss function.

    Returns:
        Loss module.

    Raises:
        ValueError: If criterion name is not recognized.

    Supported criteria:
        - "cross_entropy": CrossEntropyLoss
        - "mse": MSELoss
        - "bce": BCEWithLogitsLoss
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(
            f"Unknown criterion: '{name}'. "
            f"Supported criteria: 'cross_entropy', 'mse', 'bce'"
        )


# =============================================================================
# High-Level Builder
# =============================================================================

def build_from_config(config: TrainConfig) -> Dict[str, Any]:
    """Build all training components from a TrainConfig.

    This is a convenience function that builds model, optimizer,
    scheduler, and criterion in one call.

    Args:
        config: Complete training configuration.

    Returns:
        Dictionary containing:
            - model: The constructed model
            - optimizer: Configured optimizer
            - lr_scheduler: LR scheduler (or None)
            - criterion: Loss function

    Example:
        config = load_config("configs/vit_cifar10.yaml")
        components = build_from_config(config)

        trainer = Trainer(
            model=components["model"],
            optimizer=components["optimizer"],
            criterion=components["criterion"],
            lr_scheduler=components["lr_scheduler"],
            ...
        )
    """
    model = build_model(config.model)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.num_epochs)
    criterion = build_criterion()

    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "criterion": criterion,
    }


# =============================================================================
# Seeding Utility
# =============================================================================

def set_seed(seed: Optional[int]) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch.

    Args:
        seed: Random seed. If None, no seeding is performed.
    """
    if seed is None:
        return

    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
