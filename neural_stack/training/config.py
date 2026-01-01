"""Configuration system using dataclasses with YAML serialization.

This module provides type-safe configuration for the training framework.
Configs are defined as nested dataclasses and can be loaded from/saved to YAML files.

Example:
    # Create config programmatically
    config = TrainConfig()
    config.model.embed_dim = 512

    # Load from YAML
    config = load_config("configs/experiment.yaml")

    # Save to YAML
    save_config(config, "configs/experiment.yaml")
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Any, Dict, Tuple, get_type_hints, get_origin, get_args
from pathlib import Path

import yaml


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Attributes:
        name: Model type identifier (e.g., "vit", "resnet").
        img_size: Input image dimensions as (height, width).
        patch_size: Size of each patch for ViT models.
        in_channels: Number of input channels.
        embed_dim: Embedding/hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer/residual blocks.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout probability.
        num_classes: Number of output classes.
        positional_embedding: Type of positional embedding ("learned-1d", "learned-2d", "none").
        use_cls_token: Whether to use a CLS token for classification.
    """
    name: str = "vit"
    img_size: Tuple[int, int] = (32, 32)
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    num_classes: int = 10
    positional_embedding: str = "learned-1d"
    use_cls_token: bool = True


# =============================================================================
# Optimizer Configuration
# =============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for optimizer.

    Attributes:
        name: Optimizer type ("adamw", "adam", "sgd").
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        betas: Adam beta parameters.
        momentum: SGD momentum.
    """
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


# =============================================================================
# Scheduler Configuration
# =============================================================================

@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler.

    Attributes:
        name: Scheduler type ("step", "cosine", "none").
        step_size: Epochs between LR decay (for StepLR).
        gamma: LR decay factor (for StepLR).
        T_max: Maximum iterations for cosine annealing (defaults to num_epochs).
        eta_min: Minimum LR for cosine annealing and warmup init.
        warmup_epochs: Number of epochs for linear warmup.
    """
    name: str = "step"
    step_size: int = 12
    gamma: float = 0.1
    T_max: Optional[int] = None
    eta_min: float = 1e-9
    warmup_epochs: int = 3


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        dataset: Dataset name ("cifar10", "cifar100", "mnist").
        data_dir: Directory for dataset storage.
        batch_size: Training batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        random_horizontal_flip: Apply random horizontal flip augmentation.
        random_crop: Apply random crop augmentation.
        crop_scale: Scale range for random resized crop.
        crop_ratio: Aspect ratio range for random resized crop.
    """
    dataset: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    random_horizontal_flip: bool = True
    random_crop: bool = True
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    crop_ratio: Tuple[float, float] = (0.9, 1.1)


# =============================================================================
# Top-Level Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Top-level training configuration.

    Combines all sub-configurations and training hyperparameters.

    Attributes:
        model: Model architecture configuration.
        optimizer: Optimizer configuration.
        scheduler: LR scheduler configuration.
        data: Data loading configuration.
        num_epochs: Number of training epochs.
        device: Device to train on ("cuda", "cpu", "mps").
        seed: Random seed for reproducibility (None for no seeding).
        checkpoint_dir: Directory for saving checkpoints.
        save_best: Whether to save best model checkpoint.
        save_every: Save checkpoint every N epochs (None to disable).
        log_every: Log metrics every N batches.
        experiment_name: Name for experiment tracking.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    use_amp: bool = False 
    grad_clip_norm: Optional[float] = None

    num_epochs: int = 25
    device: str = "cuda"
    seed: Optional[int] = 42

    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    save_every: Optional[int] = None
    log_every: int = 50
    experiment_name: Optional[str] = None


# =============================================================================
# YAML Serialization Utilities
# =============================================================================

def _is_dataclass_type(cls: type) -> bool:
    """Check if a type is a dataclass."""
    return hasattr(cls, '__dataclass_fields__')


def _get_field_type(cls: type, field_name: str) -> type:
    """Get the type of a dataclass field, handling Optional types."""
    hints = get_type_hints(cls)
    field_type = hints.get(field_name)

    # Handle Optional[X] -> X
    if get_origin(field_type) is type(None) or field_type is type(None):
        return type(None)

    # Handle Optional (Union with None)
    origin = get_origin(field_type)
    if origin is not None:
        args = get_args(field_type)
        # Check if it's Optional (Union[X, None])
        if type(None) in args:
            # Return the non-None type
            for arg in args:
                if arg is not type(None):
                    return arg

    return field_type


def _dataclass_from_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively construct a dataclass from a dictionary.

    Args:
        cls: The dataclass type to construct.
        data: Dictionary with field values.

    Returns:
        Instance of the dataclass.
    """
    if not _is_dataclass_type(cls):
        return data

    if data is None:
        return cls()

    init_kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]
        field_type = _get_field_type(cls, f.name)

        # Handle nested dataclasses
        if _is_dataclass_type(field_type):
            if isinstance(value, dict):
                init_kwargs[f.name] = _dataclass_from_dict(field_type, value)
            else:
                init_kwargs[f.name] = value
        # Handle tuples (YAML loads as lists)
        elif get_origin(field_type) is tuple or field_type is tuple:
            if isinstance(value, list):
                init_kwargs[f.name] = tuple(value)
            else:
                init_kwargs[f.name] = value
        else:
            init_kwargs[f.name] = value

    return cls(**init_kwargs)


def load_config(path: str | Path) -> TrainConfig:
    """Load training configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        TrainConfig instance with values from the file.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if data is None:
        return TrainConfig()

    return _dataclass_from_dict(TrainConfig, data)


def save_config(config: TrainConfig, path: str | Path) -> None:
    """Save training configuration to a YAML file.

    Args:
        config: TrainConfig instance to save.
        path: Output path for the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    data = asdict(config)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: TrainConfig) -> Dict[str, Any]:
    """Convert a TrainConfig to a flat dictionary for logging.

    Flattens nested configs with "/" separators for compatibility
    with logging systems like W&B.

    Args:
        config: TrainConfig instance.

    Returns:
        Flattened dictionary with string keys.
    """
    def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                result.update(flatten(value, full_key))
            else:
                result[full_key] = value
        return result

    return flatten(asdict(config))
