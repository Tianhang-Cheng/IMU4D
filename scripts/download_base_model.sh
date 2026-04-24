#!/bin/bash

# Create target directory
mkdir -p exp/exp_train/checkpoint-446000/unwrapped_model

# Download the model file from HuggingFace
wget -O exp/exp_train/checkpoint-446000/unwrapped_model/pytorch_model.bin \
  "https://huggingface.co/TianhangCheng7/IMU4d/resolve/main/pytorch_model.bin"

echo "Download complete!"