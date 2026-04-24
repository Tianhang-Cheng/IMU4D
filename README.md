# Seeing Without Eyes: 4D Human-Scene Understanding from Wearable IMUs

<p align="center">
  <a href="https://arxiv.org/abs/2604.21926" title="arXiv">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv paper" />
  </a>
  <a href="https://tianhang-cheng.github.io/IMU4D/" title="Project page (replace URL when available)">
    <img src="https://img.shields.io/badge/Project-Page-4285F4?style=flat-square&logo=google-chrome&logoColor=white" alt="Project page" />
  </a>
</p>

## Environment

### 1) Create the conda environment

```bash
conda create -n imu4d python=3.11 -y
conda activate imu4d
```

### 2) Install Python dependencies

CUDA 12.8 PyTorch wheels are shown below. If your CUDA version differs, adjust `--index-url`.

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install jaxtyping evo tqdm omegaconf wandb
pip install human-body-prior@git+https://github.com/nghorbani/human_body_prior@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc
pip install diffusers==0.36.0 transformers==4.57.3 accelerate==1.1.1
```

### 3) Install this repository in editable mode

Run this from the repository root:

```bash
pip install -e .
```

## Checkpoints and Weights

- Obtain pretrained **Generator** weights using following commands. It will download base`showo_imu/checkpoint-446000/unwapped_model`. Other finetuned models are coming soon.

```bash
bash scripts/download_base_model.sh
```
- **VQVAE** is already included in `motion_vqvae/pretrained_weight`.
- **SMPLX** model requires downloading `SMPLX_NEUTRAL.npz` from [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php), put in `dataset_process_root`. Or modify `dataset_process/custom_path.py` to set the path.

## Datasets

Coming soon.

## Inference

### IMU to SMPL-X Motion and Text Description

(1) Evaluate on selected dataset (the data is coming soon)

```bash
export CUDA_VISIBLE_DEVICES=0
python run.py \
  config=configs/train.yaml \
  experiment.ckpt_dir=exp/exp_train \
  experiment.eval_selected_dataset=LINGO \
  experiment.mode=test \
  experiment.name=exp_test_lingo_5pt \
  experiment.output_dir=exp/exp_test_lingo_5pt \
  experiment.resume_from_checkpoint=True \
  experiment.strict_resume=True \
  experiment.max_eval_imu_len=60 \
  experiment.eval_invalid_imu_id=[3] \
  experiment.save_test_sample=True
```

(2) Evaluate on select single IMU input

```bash
export CUDA_VISIBLE_DEVICES=0
python run.py \
  config=configs/train.yaml \
  experiment.ckpt_dir=exp/exp_train \
  experiment.eval_selected_imu_seq=dataset_process/sample_data/LINGO_17992.pkl \
  experiment.mode=test \
  experiment.name=exp_test_lingo_5pt \
  experiment.output_dir=exp/test_sample_data_5pt \
  experiment.resume_from_checkpoint=True \
  experiment.strict_resume=True \
  experiment.max_eval_imu_len=60 \
  experiment.eval_invalid_imu_id=[3] \
  experiment.save_test_sample=True
```

result will be like:

```text
Saved sample 0 at step 0
Saved metrics to exp/test_sample_data_5pt/viz_test_generate_number_shifted_0/id_0_step_0.txt
mean_traj_error: 0.0095, mean_orient_error: 0.0344, mean_pose_error: 0.0527
mean_status_top1_acc: 29.63%, mean_status_top5_acc: 70.12%, mean_status_ce: 2.2361
mean_text_top1_acc: 0.00%, mean_text_top5_acc: 14.29%, mean_text_ce: 11.3125
mean_mpjpe: 26.67 mm
```

<details>
<summary>Inference options (click to expand)</summary>

| Option | Role |
|--------|------|
| `experiment.ckpt_dir` | Folder containing checkpoints to load when resuming. |
| `experiment.eval_selected_dataset` | Benchmark split / dataset name (for example `LINGO`). |
| `experiment.eval_selected_imu_seq` | Path to a single `.pkl` (same format as split samples). If set, evaluation runs on this file only (optional `eval_selected_dataset` for label handling). |
| `experiment.mode=test` | Test/evaluation pass (no training). |
| `experiment.name` | Run name for logging and metadata. |
| `experiment.output_dir` | Logs, metrics, and saved test artifacts. |
| `experiment.resume_from_checkpoint` | Load the latest (or specified) checkpoint from `ckpt_dir`. |
| `experiment.strict_resume` | If `True`, checkpoint keys must match the model exactly. |
| `experiment.max_eval_imu_len` | Max IMU sequence length (timesteps) generated during evaluation. |
| `experiment.eval_invalid_imu_id` | 0-based IMU slot indices treated as missing (masked). |
| `experiment.save_test_sample` | Save per-sample predictions (needed for step 3). |

IMU slots (order matches `IMU_device_names` in `train.py`):  
`left_hip`, `right_hip`, `left_ear`, `right_ear`, `left_elbow`, `right_elbow` (indices `0` to `5`).

Example: `[3]` marks `right_ear` as invalid; the model applies attention masking on that slot.

The following settings are trained in the checkpoints:

| `eval_invalid_imu_id` | Active IMUs |
|------------------------|-------------|
| `[3]` | left_hip, right_hip, left_ear, left_elbow, right_elbow |
| `[1,3,5]` | left_hip, left_ear, left_elbow |
| `[1,3,4]` | left_hip, left_ear, right_elbow |
| `[2,3,5]` | left_hip, right_hip, left_elbow |
| `[2,3,4]` | left_hip, right_hip, right_elbow |

</details>

## Visualization

Motion rollouts are visualized in [Rerun](https://www.rerun.io/) with SMPL-X bodies and scene objects. The script reads paths from `custom_path.py`: `smplx_model_dir` (directory containing `SMPLX_NEUTRAL.npz`) and `humoto_objects_folder` (Humoto object mesh root).

Install dependencies (from the repository root, after `pip install -e .`):

```bash
pip install rerun trimesh smplx scipy matplotlib
```

Run with the bundled example rollout (same layout as `experiment.save_test_sample` outputs):

```bash
python visualize/viz_motion.py --predict_path visualize/example_result/id_0_step_0.npy
```

Optional: `--baseline-path PATH` overlays a second `.npy` for comparison; `--show-pose-axis-plot` opens a matplotlib pelvis axis-angle plot before Rerun.

Rerun opens a local viewer when a display is available (`spawn=True`). On a headless machine, install Rerun on a workstation with a monitor and use its documented workflow for viewing recordings, or run with display forwarding.

## Evaluation

### Motion metrics

If `experiment.save_test_sample=True`, per-sample predictions are saved as `*.npy` under a subfolder of `experiment.output_dir` whose name starts with `viz_test_generate_number` (for example `viz_test_generate_number_shifted_0`).

Run from the repository root:

```bash
python evaluation/get_motion_metric.py \
  --result-folder exp/exp_test_lingo_5pt/viz_test_generate_number_shifted_0 \
  --model Ours \
  --dataset LINGO \
  --eval-frame-length 60
```

By default, metrics are written to `<result-folder>/evaluation`. To use a custom output directory, set `--log-dir`

<details>
<summary>Metric script options (click to expand)</summary>

| Flag | Meaning |
|------|--------|
| `--result-folder` | Directory containing `*.npy` samples (sorted by index in filename). Must start with `viz_test_generate_number`. |
| `--log-dir` | Optional metric output directory. Default: `<result-folder>/evaluation`. |
| `--model` | Label used in output filenames (default: `Ours`). |
| `--dataset` | Label used in output filenames (default: `LINGO`). |
| `--eval-frame-length` | Evaluate only the first `N` frames per clip (default: `60`). |
| `--apply-shifted-window-avg` | Optional. If `viz_test_generate_number_shifted_2` exists, average overlapping predictions with `shifted_0` to smooth window boundaries. |

</details>

#### Metrics computed (per sample, then mean/std)

- **MPJPE**: Mean per-joint position error (mm), using `compute_mpjpe` in `utils.metrics`.
- **PA MPJPE**: Procrustes-aligned MPJPE (mm).
- **MJPRE**: Mean joint pose reconstruction error (degrees), computed from axis-angle body pose differences.
- **MPJVE**: Mean per-vertex Euclidean error (mm) between predicted and GT meshes.
- **MTE**: Global motion reconstruction error from `eval_recon` on concatenated orientation and translation.

#### Metric outputs

In the metric output directory (see `--log-dir`):

- `*raw_motion_metric.txt` and `*raw_motion_metric.npy`, or
- `*avg_motion_metric.*` when `--apply-shifted-window-avg` is enabled.

The `.npy` includes `mean`, `std`, and per-sample `raw` arrays for each metric.

### Text Metric

coming soon

### Scene Metric

coming soon

## Training

### Step 1: Motion-Text Pretraining

(1) Train from scratch. Need set `model.showo.load_from_showo=True`
```bash
TRAIN_GPU_ID=7 accelerate launch --config_file accelerate_configs/1_gpus.yaml \
  --main_process_port=1644 run.py \
  config=configs/train.yaml \
  experiment.name=exp_train \
  experiment.project=imu4d \
  model.showo.load_from_showo=True \
  experiment.output_dir=exp/exp_train \
  experiment.max_eval_sample_num=50 \
  experiment.max_eval_imu_len=60 \
  experiment.min_train_imu_len=50 \
  experiment.max_train_imu_len=200 \
  experiment.eval_every=30 \
  experiment.eval_selected_dataset=LINGO \
  experiment.train_invalid_imu_id='random' \
  experiment.eval_invalid_imu_id=[3] \
  training.gradient_accumulation_steps=2 \
  training.batch_size_imu=16 \
  training.seed=1234
```

(2) Train from given checkpoint. 

The code will load `experiment.output_dir/checkpoint-xxx/pytorch_model` requires exact the same number of GPU. `experiment.output_dir/checkpoint-xxx/unwrapped_model` works for any GPU， but Optimizer, LR scheduler, and Accelerate RNG state are not restored.

```bash
TRAIN_GPU_ID=7 accelerate launch --config_file accelerate_configs/1_gpus.yaml \
  --main_process_port=1644 run.py \
  config=configs/train.yaml \
  experiment.name=exp_train \
  experiment.project=imu4d \
  experiment.output_dir=exp/exp_train \
  experiment.ckpt_dir=exp/exp_train \
  experiment.resume_from_checkpoint=True \
  experiment.load_without_optimizer=True \
  experiment.strict_resume=True \
  experiment.max_eval_sample_num=50 \
  experiment.max_eval_imu_len=60 \
  experiment.min_train_imu_len=50 \
  experiment.max_train_imu_len=200 \
  experiment.eval_every=30 \
  experiment.eval_selected_dataset=LINGO \
  experiment.train_invalid_imu_id='random' \
  experiment.eval_invalid_imu_id=[3] \
  experiment.eval_every=1000 \
  experiment.max_eval_sample_num=30 \
  training.gradient_accumulation_steps=1 \
  training.batch_size_imu=32 \
  training.seed=1234
```

Choose the matching `accelerate_configs/*.yaml` and set `TRAIN_GPU_ID` to a comma-separated list:
```bash
TRAIN_GPU_ID=4,7 accelerate launch --config_file accelerate_configs/2_gpus.yaml ...
```
 
### Step 2: Finetune on specific dataset (3D scene / realworld)

Load checkpoint from step 1 as initialization.

+ IMUPoser dataset (Realworld). 
```bash
TRAIN_GPU_ID=4 accelerate launch --config_file accelerate_configs/1_gpus.yaml \
  --main_process_port=1644 run.py \
  config=configs/train.yaml \
  experiment.name=exp_train \
  experiment.project=imu4d \
  experiment.output_dir=exp/exp_finetune_imuposer \
  experiment.ckpt_dir=exp/exp_train \
  experiment.resume_from_checkpoint=True \
  experiment.load_without_optimizer=True \
  experiment.strict_resume=False \
  experiment.max_eval_sample_num=50 \
  experiment.max_eval_imu_len=200 \
  experiment.min_train_imu_len=100 \
  experiment.max_train_imu_len=200 \
  experiment.eval_every=30 \
  experiment.save_every=100 \
  experiment.is_start_eval=True \
  experiment.is_start_sample=True \
  experiment.train_selected_dataset=imuposer \
  experiment.eval_selected_dataset=imuposer \
  experiment.train_invalid_imu_id='random' \
  experiment.eval_invalid_imu_id=[3] \
  training.gradient_accumulation_steps=3 \
  training.batch_size_imu=24 \
  training.seed=1234 \
  training.num_train_epochs=420 \
  training.finetune_on_specific_dataset=True
```

<details>
<summary>DIPIMU dataset (Realworld) (click to expand)</summary>

```bash
TRAIN_GPU_ID=4 accelerate launch --config_file accelerate_configs/1_gpus.yaml \
    --main_process_port=1644 run.py \
    config=configs/train.yaml \
    experiment.name=exp_train \
    experiment.project=imu4d \
    experiment.output_dir=exp/exp_finetune_dipimu \
    experiment.ckpt_dir=exp/exp_train \
    experiment.resume_from_checkpoint=True \
    experiment.load_without_optimizer=True \
    experiment.strict_resume=False \
    experiment.max_eval_sample_num=50 \
    experiment.max_eval_imu_len=200 \
    experiment.min_train_imu_len=100 \
    experiment.max_train_imu_len=200 \
    experiment.eval_every=50 \
    experiment.save_every=100 \
    experiment.is_start_eval=True \
    experiment.is_start_sample=True \
    experiment.train_selected_dataset=dipimu \
    experiment.eval_selected_dataset=dipimu \
    experiment.train_invalid_imu_id='random' \
    experiment.eval_invalid_imu_id=[3] \
    training.gradient_accumulation_steps=2 \
    training.batch_size_imu=32 \
    training.seed=1234 \
    training.num_train_epochs=600 \
    training.finetune_on_specific_dataset=True
```

</details>

<details>
<summary>Humoto dataset (Synthetic 3D scene dataset) (click to expand)</summary>

```bash
TRAIN_GPU_ID=4 accelerate launch --config_file accelerate_configs/1_gpus.yaml \
    --main_process_port=1644 run.py \
    config=configs/train.yaml \
    experiment.name=exp_train \
    experiment.project=imu4d \
    experiment.output_dir=exp/exp_finetune_humoto \
    experiment.ckpt_dir=exp/exp_train \
    experiment.resume_from_checkpoint=True \
    experiment.load_without_optimizer=False \
    experiment.strict_resume=True \
    experiment.max_eval_sample_num=50 \
    experiment.max_eval_imu_len=200 \
    experiment.min_train_imu_len=60 \
    experiment.max_train_imu_len=200 \
    experiment.eval_every=30 \
    experiment.save_every=100 \
    experiment.is_start_eval=True \
    experiment.is_start_sample=True \
    experiment.train_selected_dataset=HUMOTO \
    experiment.eval_selected_dataset=HUMOTO \
    experiment.train_invalid_imu_id='random' \
    experiment.eval_invalid_imu_id=[3] \
    training.gradient_accumulation_steps=3 \
    training.batch_size_imu=24 \
    training.seed=1234 \
    training.max_grad_norm=2.0 \
    training.num_train_epochs=500 \
    training.finetune_on_specific_dataset=True \
```
</details>

<details>
<summary>Training options (click to expand)</summary>

See `configs/train.yaml` and `run.py` for the complete list.

| Option | Role |
|--------|------|
| `TRAIN_GPU_ID` | GPU index(es) for this run (environment variable read by launch script). |
| `--main_process_port` | Free port for the main process in distributed launch. |
| `experiment.output_dir` | Logs and generated outputs. |
| `experiment.ckpt_dir` | Checkpoint save/load directory (often same as `output_dir`). |
| `experiment.resume_from_checkpoint` | Resume from latest checkpoint if present. |
| `experiment.load_without_optimizer` | If `True`, load only `checkpoint-*/unwrapped_model/pytorch_model.bin` (see table above). Optimizer, scheduler, and Accelerate RNG are not loaded. If `False` in training, uses full `accelerator.load_state` (same GPU/process count as when saved). |
| `experiment.strict_resume` | Applies when loading unwrapped weights. If `True`, `load_state_dict(..., strict=True)` — every key and shape must match. If `False`, loads overlapping keys; mismatched shapes get a partial copy into the current parameter tensors (useful when the model definition changed slightly). |
| `experiment.max_eval_sample_num` | Max evaluation samples per evaluation pass. |
| `experiment.max_eval_imu_len` | Max IMU length during evaluation. |
| `experiment.min_train_imu_len` / `experiment.max_train_imu_len` | IMU sequence length range used in training. |
| `experiment.eval_every` | Run evaluation every `N` training steps. |
| `experiment.eval_selected_dataset` | Dataset/split used for evaluation (for example `LINGO`). |
| `experiment.train_invalid_imu_id` | Invalid IMU slot sampling strategy. 'random' mode will randomly choose valid combinations |
| `experiment.eval_invalid_imu_id` | Fixed invalid IMU slot indices during evaluation. |
| `training.gradient_accumulation_steps` | Gradient accumulation steps before optimizer update. |
| `training.batch_size_imu` | IMU batch size. |
| `training.seed` | Random seed. |
| `finetune_on_specific_dataset`| Will freeze MLP decoder from previous stage. |

</details>

### Optional: Time Series Tokenizer (TOTEM)

The tokenizer supports `compression_rate` **4** or **8**. The default setup uses **4**.

Train it from the TOTEM checkout (for example under `motion_vqvae`):

```bash
cd motion_vqvae
python train.py --compression_rate 4
```
