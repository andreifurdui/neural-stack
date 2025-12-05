import os
import torch
import torch.nn as nn

from typing import Tuple
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def count_model_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def model_summary(model: nn.Module, inputs: torch.Tensor | Tuple[torch.Tensor, ...]):
    flops = FlopCountAnalysis(model, inputs)
    acts = ActivationCountAnalysis(model, inputs)

    num_params = count_model_params(model)
    num_flops = flops.total()
    num_activations = acts.total()

    summary = flop_count_table(flops, activations=acts)

    return num_params, num_flops, num_activations, summary