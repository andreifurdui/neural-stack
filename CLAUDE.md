# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**neural-stack** is a learning project for the **Advanced AI Engineering Program**—a 58-week self-directed curriculum covering modern AI systems from transformers to production deployment. The repository contains from-scratch implementations of foundational architectures with a modular training framework.

**Current Focus:** Module 1 - Vision Transformers & Modern Architectures
**Program Structure:** 8 modules (7 core + 1 optional) covering transformers → diffusion → distributed training → MLOps → agentic AI → RL → privacy ML → 3D vision

See `course_docs/00_programme_overview.md` for complete curriculum details.

## Repository Organization

```
neural_stack/              # Main Python package (pip install -e .)
├── models/               # Model implementations
│   ├── attention.py      # ScaledDotProductAttention, MultiHeadAttention
│   └── vision_transformer.py  # ViT with configurable positional embeddings
├── training/             # Training framework
│   ├── config.py         # Dataclass configs + YAML utilities
│   ├── trainer.py        # Base Trainer with callbacks + override hooks
│   ├── callbacks.py      # Callback system (checkpointing, logging, etc.)
│   ├── factory.py        # Component builders (model, optimizer, scheduler)
│   └── data.py           # Dataset/dataloader utilities
├── utils.py              # Model analysis (params, FLOPs)
└── visualization.py      # Attention heatmap visualization

configs/                  # YAML configuration files
notebooks/                # Jupyter notebooks for experimentation
course_docs/              # Curriculum documentation
data/                     # Datasets, checkpoints (gitignored)

.claude/
├── agents/              # Custom Claude Code agents
└── skills/              # Custom skills
```

## Development Workflow

### Installation
```bash
pip install -e .          # Development mode
pip install -e ".[dev]"   # With Jupyter
```

### Training Models

The training framework uses YAML configs and a callback-based Trainer:

```python
from neural_stack.training import (
    load_config, TrainConfig,
    Trainer, build_from_config, build_dataloaders,
    PrintCallback, CheckpointCallback, EarlyStoppingCallback,
)

# Load from YAML or create programmatically
config = load_config("configs/vit_cifar10.yaml")
# OR: config = TrainConfig()

# Build components
components = build_from_config(config)
train_loader, val_loader = build_dataloaders(config.data)

# Train with callbacks
trainer = Trainer(
    model=components["model"],
    optimizer=components["optimizer"],
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=components["criterion"],
    lr_scheduler=components["scheduler"],
    callbacks=[
        PrintCallback(),
        CheckpointCallback(config.checkpoint_dir, save_best=True),
        EarlyStoppingCallback(patience=5),
    ],
    device=config.device,
    num_epochs=config.num_epochs,
)
results = trainer.fit()
```

### Configuration System

Configs are nested dataclasses with YAML serialization:

```python
from neural_stack.training import TrainConfig, save_config, load_config

config = TrainConfig()
config.model.embed_dim = 512
config.optimizer.lr = 1e-4

save_config(config, "configs/my_experiment.yaml")
loaded = load_config("configs/my_experiment.yaml")
```

Config hierarchy: `TrainConfig` → `ModelConfig`, `OptimizerConfig`, `SchedulerConfig`, `DataConfig`

### Running Experiments
```bash
jupyter notebook
```

Notebooks in `notebooks/` are for experimentation and analysis.

## Training Framework

### Extending via Callbacks

Callbacks hook into the training loop for infrastructure concerns:

```python
from neural_stack.training import Callback, TrainState

class MyCallback(Callback):
    def on_epoch_end(self, state: TrainState) -> None:
        print(f"Epoch {state.epoch}: {state.metrics}")
```

Available hooks: `on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`

Built-in callbacks:
- `PrintCallback` - Console logging
- `CheckpointCallback` - Save best/periodic checkpoints
- `EarlyStoppingCallback` - Stop on metric plateau
- `WandbCallback` - W&B integration
- `ProgressCallback` - tqdm progress bars

### Extending via Subclassing

For custom training recipes (distillation, fine-tuning), subclass `Trainer`:

```python
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=4.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model

    def compute_loss(self, outputs, targets):
        # Custom distillation loss
        hard_loss = self.criterion(outputs, targets)
        soft_loss = kl_div(outputs, self.teacher(inputs))
        return 0.5 * hard_loss + 0.5 * soft_loss
```

Override points: `train_step()`, `compute_loss()`, `unpack_batch()`, `forward()`, `_validate()`

### Factory Functions

Build components from config:

```python
from neural_stack.training import build_model, build_optimizer, build_scheduler

model = build_model(config.model)        # Supports: "vit"
optimizer = build_optimizer(config.optimizer, model)  # Supports: "adamw", "adam", "sgd"
scheduler = build_scheduler(config.scheduler, optimizer, num_epochs)  # Supports: "step", "cosine", "none"
```

### Data Loading

```python
from neural_stack.training import build_dataloaders, get_dataset_info

train_loader, val_loader = build_dataloaders(config.data)
info = get_dataset_info("cifar10")  # Returns num_classes, img_size, etc.
```

Supported datasets: `cifar10`, `cifar100`, `mnist`, `fashion_mnist`

## Model Architecture Patterns

### Factory Pattern for Variants
```python
pos_emb = PositionalEmbedding.create(
    positional_embedding_type='learned-2d',  # or 'learned-1d', 'none'
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=embed_dim,
    use_cls_token=use_cls_token
)
```

### Attention Returns Scores
```python
output, attention_scores = multi_head_attention(query, key, value)
output, attention_list = vision_transformer(images, return_attention=True)
```

Use `visualization.plot_attention_heatmap(attention_scores)` for visualization.

### Model Analysis
```python
from neural_stack.utils import model_summary
num_params, num_flops, num_activations, table = model_summary(model, sample_input)
```

## Dependencies

Core (in `pyproject.toml`):
- `torch>=2.0.0`, `torchvision>=0.15.0` - Deep learning
- `pyyaml>=6.0` - Config serialization
- `tqdm>=4.65.0` - Progress bars
- `plotly>=5.0.0` - Visualization
- `fvcore>=0.1.5` - Model analysis
- `pandas>=2.0.0` - Data manipulation

Optional:
- `wandb` - Experiment tracking (for WandbCallback)
- `jupyter` - Notebook environment

## Program Context

### Learning Structure
- **Weekly practicals** tracked as GitHub Issues
- **Module milestones** for the 8 curriculum modules
- **Portfolio projects** at module completion

Branch naming: `practical/{topic}` or `module-{N}/{feature}`

### Completion Criteria
1. Explain concepts without references
2. Implement algorithms from scratch
3. Compare to 3+ recent papers
4. Apply to novel problems
5. Deliver portfolio artifact

### Custom Agents
- **neural-stack-mentor**: Technical mentorship (Socratic approach)
- **module-planning-agent**: Curriculum design

## Maintenance

### Adding Models
1. Create `neural_stack/models/{name}.py`
2. Add to factory in `neural_stack/training/factory.py`
3. Create experiment notebook

### Adding Datasets
1. Add to `neural_stack/training/data.py`
2. Include normalization stats in `DATASET_STATS`

### Updating CLAUDE.md
Update when:
- Repository structure changes significantly
- New training patterns are established
- Dependencies change

## Philosophy

This is a **learning repository**:
- **Deep understanding** through from-scratch implementation
- **Experimentation** via systematic comparisons
- **Interpretability** through visualizations
- **Clarity** over optimization

Code is educational first, production-ready second.
