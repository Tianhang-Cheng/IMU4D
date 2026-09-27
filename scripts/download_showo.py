#!/usr/bin/env python3
"""Download the public Show-o initialization weights for stage 1 training."""

from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
target = hf_hub_download(
    "showlab/show-o", "pytorch_model.safetensors",
    local_dir=ROOT / "pretrained_weight" / "showo",
)
print(target)
