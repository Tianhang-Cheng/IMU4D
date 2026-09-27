"""Low-rank adaptation of the Show-o backbone.

`LoRALinear` subclasses `nn.Linear` and reuses the original weight / bias
Parameters, so the module keeps its state-dict keys (`...q_proj.weight`) and
only adds `...q_proj.lora_A` / `...q_proj.lora_B`.  Two consequences matter for
this repository:

* a full-parameter checkpoint loads into a LoRA model unchanged (the adapter
  keys are simply missing, and `lora_B` is zero-initialised, so the adapted
  model starts out identical to its initialisation), and
* `merged_state_dict` folds `alpha / r * B @ A` back into `weight`, so the
  `unwrapped_model/pytorch_model.bin` written by `save_checkpoint` is an
  ordinary full checkpoint that every evaluation / fine-tuning path reads
  without knowing that LoRA was used.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2")


class LoRALinear(nn.Linear):
    """`nn.Linear` with an additive low-rank branch; the base weight is frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        # Build the shell on the meta device and adopt the original Parameters:
        # no second copy of the base weight is allocated.
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device="meta",
        )
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(
            torch.empty(self.lora_r, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.lora_r, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + update * self.lora_scaling

    def delta_weight(self) -> torch.Tensor:
        """`alpha / r * B @ A`, in the base weight's dtype."""
        delta = self.lora_B.detach().float() @ self.lora_A.detach().float()
        return (delta * self.lora_scaling).to(self.weight.dtype)

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{super().extra_repr()}, lora_r={self.lora_r}, "
            f"lora_alpha={self.lora_alpha}"
        )


def _named_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def has_lora(model: nn.Module) -> bool:
    return any(True for _ in _named_lora_modules(model))


def apply_lora(
    model: nn.Module,
    *,
    scope: str = "showo",
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    train_norms: bool = True,
    train_biases: bool = False,
) -> dict:
    """Freeze `scope` and add a LoRA branch to each targeted `nn.Linear` in it.

    Everything outside `scope` (the IMU aggregators, motion / pose heads,
    identity head, ...) keeps the trainability the caller gave it, so the
    existing `train_only` / `finetune_on_specific_dataset` rules still decide
    what else is optimized.
    """
    root = model.get_submodule(scope) if scope else model
    for param in root.parameters():
        param.requires_grad = False

    targets = tuple(target_modules)
    replaced: list[str] = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(f"{scope}.{name}" if scope else name)

    if not replaced:
        raise ValueError(
            f"LoRA target modules {targets} matched no nn.Linear under '{scope}'"
        )

    unfrozen: list[str] = []
    if train_norms:
        for name, param in root.named_parameters():
            if "norm" in name.lower():
                param.requires_grad = True
                unfrozen.append(f"{scope}.{name}" if scope else name)
    if train_biases:
        for name, param in root.named_parameters():
            if name.endswith(".bias") and "lora" not in name:
                param.requires_grad = True
                unfrozen.append(f"{scope}.{name}" if scope else name)

    lora_params = sum(
        module.lora_A.numel() + module.lora_B.numel()
        for _, module in _named_lora_modules(model)
    )
    return {
        "replaced": replaced,
        "unfrozen": unfrozen,
        "lora_parameters": lora_params,
        "rank": r,
        "alpha": alpha,
        "dropout": dropout,
        "scope": scope,
        "target_modules": list(targets),
    }


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors (plus any other trainable, non-base tensor)."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if key.endswith((".lora_A", ".lora_B"))
    }


def merged_state_dict(model: nn.Module) -> dict:
    """Full state dict with every LoRA branch folded into its base weight."""
    state_dict = model.state_dict()
    merged = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith((".lora_A", ".lora_B"))
    }
    for name, module in _named_lora_modules(model):
        weight_key = f"{name}.weight" if name else "weight"
        if weight_key not in merged:
            raise KeyError(f"LoRA module {name} has no base weight in the state dict")
        base = merged[weight_key]
        merged[weight_key] = base + module.delta_weight().to(base.device)
    return merged
