#!/usr/bin/env python3
"""Download IMU4D release weights from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from infer import MODELS


REPO_ID = "TianhangCheng7/IMU4d"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=sorted(MODELS), help="Default: all eight models")
    args = parser.parse_args()
    for model in args.models or list(MODELS):
        folder = MODELS[model][0]
        filename = f"checkpoints/{folder}/pytorch_model.bin"
        target = hf_hub_download(REPO_ID, filename, local_dir=ROOT)
        print(f"{model}: {target}")


if __name__ == "__main__":
    main()
