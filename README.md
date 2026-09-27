# IMU4D: 4D Human-Object Understanding from Wearable IMUs

[Paper](https://arxiv.org/abs/2604.21926) · [Project page](https://tianhang-cheng.github.io/IMU4D/) · [Model weights](https://huggingface.co/TianhangCheng7/IMU4d) · [Processed data](https://huggingface.co/datasets/TianhangCheng7/IMU4DData)

This repository contains the IMU4D inference and training code. It predicts
SMPL-X motion and text from wearable IMU sequences; the scene models also
predict objects. The eight released generator weights are **internal training
versions**. The training scripts below are the public training recipe and do
not claim to reproduce those exact weight files.

## Installation

Create a Python 3.11 environment. The example below uses CUDA 12.8 PyTorch;
choose the wheel that matches your CUDA setup.

```bash
conda create -n imu4d python=3.11 -y
conda activate imu4d
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install diffusers==0.36.0 transformers==4.57.3 accelerate==1.1.1
pip install omegaconf webdataset huggingface_hub safetensors wandb jaxtyping evo tqdm smplx scipy matplotlib trimesh
pip install 'human-body-prior@git+https://github.com/nghorbani/human_body_prior@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc'
pip install -e .
```

The motion tokenizer is included at
`motion_tokenizer/pretrained_weight/4_final.pth`. Obtain
`SMPLX_NEUTRAL.npz` from [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php)
under its license and place it at `data/models/smplx/SMPLX_NEUTRAL.npz`, or
set `IMU4D_SMPLX_PATH` to its full path.

## Released models

Each Hugging Face path contains a `pytorch_model.bin` file. Download only the
model you need, or omit the model arguments to download all eight (about
48.3 GB total).

```bash
python scripts/download_checkpoints.py pretrain
# All eight:
python scripts/download_checkpoints.py
```

| `--model` | Hugging Face directory | Intended input |
|---|---|---|
| `pretrain` | `checkpoints/showo_pretrain_full` | Synthetic IMU, general motion and text |
| `noise` | `checkpoints/showo_pretrain_full_noise` | Noise augmented synthetic IMU |
| `imuposer` | `checkpoints/showo_finetune_imuposer` | IMUPoser |
| `dipimu` | `checkpoints/showo_finetune_dipimu` | DIP-IMU |
| `ncsa` | `checkpoints/showo_finetune_ncsa_fix` | NCSA capture |
| `hiphi` | `checkpoints/showo_finetune_hiphi` | HiPHI human-object scenes |
| `omomo` | `checkpoints/showo_finetune_omomo` | OMOMO human-object scenes |
| `humoto` | `checkpoints/showo_finetune_humoto` | HUMOTO human-object scenes |

These are inference weights. Optimizer state and intermediate training
snapshots are not part of the model release. The NCSA download directory has
the `ncsa_fix` suffix; its inference profile is `configs/showo_finetune_ncsa.yaml`.

## Inference on one IMU sample

The repository includes a packed example. No training dataset download is
needed for this command:

```bash
python scripts/infer.py \
  --model pretrain \
  --input dataset_process/sample_data/LINGO_17992.pkl \
  --frames 60 --gpu 0 \
  --output exp/inference/lingo_example
```

`scripts/infer.py` accepts all eight names in the table. Supply a packed
`.pkl` with the IMU4D sample fields and choose the corresponding model. The
script loads the downloaded weight without copying the multi-GB file and
writes motion and text predictions under `--output`. Use `--weights PATH` to
load an existing local `pytorch_model.bin`, `--dataset NAME` to override the
input's dataset label, or `--invalid-imu-ids '[3]'` to mask sensor slots.
The default mask matches the selected model; slots listed as missing in a
measured IMU sample are also masked. Sensor indices are 0: left hip, 1: right
hip, 2: left ear, 3: right
ear, 4: left elbow, 5: right elbow.

The sample format is the same as the processed dataset: synthetic samples
contain `imu_traj`; measured IMU samples contain `imu_acc` and `imu_ori`.
Inference uses one GPU and does not require WebDataset shards. The scene
weights use the model architecture without object geometry conditioning.

To evaluate a processed test split, pass its WebDataset root instead of
`--input`. This runs up to 50 test samples by default and writes the metrics
under `--output`:

```bash
python scripts/infer.py --model hiphi \
  --dataset-root data/processed/hiphi/v1 \
  --max-samples 50 --frames 60 \
  --output exp/evaluation/hiphi_50
```

## Data for training

Processed WebDataset shards are published in
[IMU4DData](https://huggingface.co/datasets/TianhangCheng7/IMU4DData).
The training configs use `data/` at the repository root by default.

```bash
export IMU4D_DATA_ROOT="$PWD/data"
hf download TianhangCheng7/IMU4DData --repo-type dataset \
  --local-dir "$IMU4D_DATA_ROOT/processed"
```

For a single scene dataset, pass `--include 'hiphi/**'` (or `omomo/**`,
`humoto/**`). DIP-IMU is subject to its own license and is not in the public
dataset repository. Prepare its processed `dipimu/v2` shards separately.
Training also uses rewritten caption sidecars included in the public
processed-data release.

## Training

Stage 1 starts from the [public Show-o model](https://huggingface.co/showlab/show-o),
trains an adapter warm-up, then trains the full IMU4D model to **500,000
optimizer steps on one GPU** by default. Download Show-o separately from the
eight IMU4D inference weights:

```bash
python scripts/download_showo.py
TRAIN_GPU_ID=0 bash scripts/fast_train/stage1_pretrain.sh
```

`ADAPTER_WARMUP_STEPS` defaults to 1,000 and `FULL_STEPS` defaults to 500,000.
`FULL_STEPS` is an absolute target. To resume a full checkpoint with its
optimizer state:

```bash
TRAIN_GPU_ID=0 RESUME_FULL=1 bash scripts/fast_train/stage1_pretrain.sh
```

The later stages use the completed stage 1 checkpoint. Stage 2 adds IMU
noise and supplies the starting point for real-world fine-tuning; scene
fine-tuning branches from stage 1.

```bash
TRAIN_GPU_ID=0 bash scripts/fast_train/stage2_noise_aug.sh
TRAIN_GPU_ID=0 DATASET=imuposer bash scripts/fast_train/stage3_realworld.sh
TRAIN_GPU_ID=0 DATASET=dipimu   bash scripts/fast_train/stage3_realworld.sh
TRAIN_GPU_ID=0 DATASET=ncsa     bash scripts/fast_train/stage3_realworld.sh
TRAIN_GPU_ID=0 DATASET=hiphi    bash scripts/fast_train/stage3_scene.sh
TRAIN_GPU_ID=0 DATASET=omomo    bash scripts/fast_train/stage3_scene.sh
TRAIN_GPU_ID=0 DATASET=humoto   bash scripts/fast_train/stage3_scene.sh
```

Each script supports `RESUME_FULL=1`, `FULL_DIR=...`, `FULL_STEPS=...`, and
`DRY_RUN=1`; read its header for stage-specific options.
Scene training defaults to no object geometry embedding, matching the released
scene models. Set `OBJ_GEOM=1` to train a different geometry-conditioned model.
Training checkpoints live under `exp/` and are separate from the flat
inference weights under `checkpoints/`.
