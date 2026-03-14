"""Self-contained LoRA (Low-Rank Adaptation) module for SAM.

Replaces the manual bottleneck adapters (adapter_v, adapter_tf) with
standard LoRA on Linear layers: W' = W + (B @ A) * scaling.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation.

    Freezes the original weight W and learns ΔW = B @ A (rank r).
    Output: x @ (W + B @ A * scaling)^T + bias

    Args:
        original_linear: The nn.Linear to wrap.
        r: LoRA rank.
        alpha: LoRA scaling factor (scaling = alpha / r).
    """

    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.scaling = alpha / r

        # Freeze original weight and bias
        self.linear = original_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA low-rank matrices (on same device/dtype as original weight)
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))

        # Kaiming uniform init for A, zero init for B (so ΔW = 0 at start)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.linear(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        result = result + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, scaling={self.scaling}"
        )


def apply_lora_to_linear(
    module: nn.Module,
    target_names: List[str],
    r: int = 16,
    alpha: float = 16.0,
) -> int:
    """Replace specified nn.Linear layers in a module with LoRALinear.

    Args:
        module: The parent module to modify (in-place).
        target_names: Attribute names of nn.Linear layers to wrap.
        r: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Number of layers replaced.
    """
    count = 0
    for name in target_names:
        if not hasattr(module, name):
            continue
        original = getattr(module, name)
        if not isinstance(original, nn.Linear):
            continue
        setattr(module, name, LoRALinear(original, r=r, alpha=alpha))
        count += 1
    return count
