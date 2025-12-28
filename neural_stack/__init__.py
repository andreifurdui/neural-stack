"""Neural Stack - Transformer fundamentals and neural network implementations."""

# Import submodules to make them accessible via neural_stack.submodule
from neural_stack.models import attention
from neural_stack.models import vision_transformer
from neural_stack import visualization
from neural_stack import training

__version__ = "0.1.0"

__all__ = [
    # Submodules
    "attention",
    "vision_transformer",
    "visualization",
    "training"
]
