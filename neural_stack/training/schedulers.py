import math
from typing import override
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class CosineAnnealingWarmupLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        # Iters per batch will be set during training
        self.iters_per_batch = None

        super().__init__(optimizer, last_epoch)

    def set_iters_per_epoch(self, iters_per_epoch: int):
        """Set the number of iterations per epoch."""
        if self.iters_per_batch is None:
            self.iters_per_batch = iters_per_epoch
            self.warmup_steps = self.warmup_epochs * iters_per_epoch
            self.T_max = (self.T_max * iters_per_epoch) - self.warmup_steps

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate of each parameter group."""
        if self._is_initial or self.last_epoch <= 0:
            return [
                self.eta_min for _ in self.optimizer.param_groups
            ]
        
        if self.iters_per_batch is None:
            raise ValueError("iters_per_batch must be set before calling get_lr()")

        if self.last_epoch < self.warmup_steps:
            return [
                group["lr"] + (base_lr - self.eta_min) / self.warmup_steps
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        
        if self.last_epoch == self.warmup_steps:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.T_max))
                * 0.5
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        
        if self.last_epoch > self.warmup_steps:
            return [
                self.eta_min
                + (group["lr"] - self.eta_min)
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.T_max))
                / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps - 1) / self.T_max))
                for group in self.optimizer.param_groups
            ]
    
    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.last_epoch - self.warmup_steps / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]