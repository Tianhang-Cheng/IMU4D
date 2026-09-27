---
library_name: pytorch
tags:
- imu
- human-motion
- human-object-interaction
- smpl-x
---

# IMU4D model weights

Eight IMU4D generator weights for wearable-IMU motion, text, and scene
inference. These are **internal training versions**. The public training
scripts in the [code repository](https://github.com/Tianhang-Cheng/IMU4D)
provide a training recipe; they do not describe the exact training history of
these weight files.

| Model name | File | Task |
|---|---|---|
| `pretrain` | `checkpoints/showo_pretrain_full/pytorch_model.bin` | General motion and text |
| `noise` | `checkpoints/showo_pretrain_full_noise/pytorch_model.bin` | Noise augmented IMU |
| `imuposer` | `checkpoints/showo_finetune_imuposer/pytorch_model.bin` | IMUPoser |
| `dipimu` | `checkpoints/showo_finetune_dipimu/pytorch_model.bin` | DIP-IMU |
| `ncsa` | `checkpoints/showo_finetune_ncsa_fix/pytorch_model.bin` | NCSA capture |
| `hiphi` | `checkpoints/showo_finetune_hiphi/pytorch_model.bin` | HiPHI scenes |
| `omomo` | `checkpoints/showo_finetune_omomo/pytorch_model.bin` | OMOMO scenes |
| `humoto` | `checkpoints/showo_finetune_humoto/pytorch_model.bin` | HUMOTO scenes |

Install the code and follow its [README](https://github.com/Tianhang-Cheng/IMU4D#readme)
for dependencies, the included motion tokenizer, input format, and inference.
For example:

```bash
python scripts/download_checkpoints.py pretrain
python scripts/infer.py --model pretrain \
  --input dataset_process/sample_data/LINGO_17992.pkl \
  --output exp/inference/lingo_example
```

The weights contain model parameters for inference. They do not include
optimizer states or intermediate training snapshots. Processed evaluation
data is in [IMU4DData](https://huggingface.co/datasets/TianhangCheng7/IMU4DData).
