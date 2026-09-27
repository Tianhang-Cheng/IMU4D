#!/usr/bin/env bash
# Real-world fine-tuning of the Show-o full model on IMUPoser, DIP-IMU or the NCSA capture.
# Initializes from the latest complete checkpoint of exp/showo_pretrain_full_noise
# (the run started by scripts/fast_train/stage2_noise_aug.sh) and streams the real-IMU
# shards under data/processed/{imuposer,dipimu,ncsa}/v*/wds (see
# dataset_process/realworld/README.md and dataset_process/ncsa/README_ncsa_imu.md
# to build them).
#
# New:    TRAIN_GPU_ID=7 DATASET=imuposer bash scripts/fast_train/stage3_realworld.sh
#         TRAIN_GPU_ID=7 DATASET=dipimu   bash scripts/fast_train/stage3_realworld.sh
#         TRAIN_GPU_ID=7 DATASET=ncsa     bash scripts/fast_train/stage3_realworld.sh
# Zero-shot metrics of the source checkpoint (no training):
#         FULL_DIR=exp/showo_eval_ncsa_pretrained DATASET=ncsa \
#         bash scripts/fast_train/stage3_realworld.sh -- experiment.full_eval_only=True
# Resume: TRAIN_GPU_ID=7 DATASET=imuposer RESUME_FULL=1 bash scripts/fast_train/stage3_realworld.sh
# Other source checkpoint:
#         SHOWO_INIT_CHECKPOINT=exp/showo_pretrain_full/checkpoint-40000 DATASET=dipimu bash scripts/fast_train/stage3_realworld.sh
# LoRA is the default (rank 32, alpha 64, LR 2e-4; see configs/README.md);
# the saved checkpoint has the adapters merged in and is read like any other one:
#         LORA_R=64 DATASET=ncsa bash scripts/fast_train/stage3_realworld.sh
#         LORA=0 FT_LR=5e-5 DATASET=ncsa bash scripts/fast_train/stage3_realworld.sh
# Overrides:
#         EXTRA_OVERRIDES="training.max_train_steps=20000" DATASET=imuposer bash scripts/fast_train/stage3_realworld.sh
# Output: FULL_DIR (default exp/showo_finetune_<dataset>)
# Validate only:
#         DRY_RUN=1 DATASET=imuposer bash scripts/fast_train/stage3_realworld.sh

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../_imu4d_env.bash"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
dataset="${DATASET:-imuposer}"
case "$dataset" in
    imuposer|dipimu|ncsa) ;;
    *) echo "DATASET must be imuposer, dipimu or ncsa, got '$dataset'." >&2; exit 2 ;;
esac
config="${CONFIG:-configs/showo_finetune_${dataset}.yaml}"
default_steps="$(python - "$dataset" <<'PYEOF'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/train_iters.yaml")
print(OmegaConf.select(cfg, f"joint.finetune.real.{sys.argv[1]}"))
PYEOF
)"
full_steps="${FULL_STEPS:-$default_steps}"
resume_full="${RESUME_FULL:-0}"
showo_init_source="${SHOWO_INIT_CHECKPOINT:-exp/showo_pretrain_full_noise}"
dry_run=()
[[ "${DRY_RUN:-0}" == "1" ]] && dry_run=(--dry-run)

[[ "$full_steps" =~ ^[1-9][0-9]*$ ]] || {
    echo "FULL_STEPS must be a positive integer." >&2
    exit 2
}

latest_complete_checkpoint() {
    local root="$1" checkpoint step best_step=-1 best_path=""
    [[ "$root" = /* ]] || root="$repo_root/$root"
    shopt -s nullglob
    for checkpoint in "$root"/checkpoint-*; do
        [[ -f "$checkpoint/unwrapped_model/pytorch_model.bin" ]] || continue
        step="${checkpoint##*-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if (( step > best_step )); then
            best_step="$step"
            best_path="$checkpoint"
        fi
    done
    shopt -u nullglob
    [[ -n "$best_path" ]] && printf '%s\n' "$best_path"
}

if [[ "$resume_full" == "1" ]]; then
    # Full optimizer-state resume from the fine-tuning output directory.
    exec env FULL_STEPS="$full_steps" RESUME_FULL=1 \
        python scripts/launch_pretrain.py "${dry_run[@]}" "$config" "$@"
fi

[[ "$showo_init_source" = /* ]] || showo_init_source="$repo_root/$showo_init_source"
if [[ -d "$showo_init_source" && "$(basename "$showo_init_source")" != checkpoint-* ]]; then
    showo_init_checkpoint="$(latest_complete_checkpoint "$showo_init_source" || true)"
    [[ -n "$showo_init_checkpoint" ]] || {
        echo "No complete checkpoint found under Show-o initialization directory: $showo_init_source" >&2
        exit 1
    }
else
    showo_init_checkpoint="$showo_init_source"
fi
[[ -f "$showo_init_checkpoint/unwrapped_model/pytorch_model.bin" ]] || {
    echo "Initial checkpoint weights do not exist: $showo_init_checkpoint/unwrapped_model/pytorch_model.bin" >&2
    exit 1
}

echo "Starting Show-o real-world ($dataset) fine-tuning from: $showo_init_checkpoint"
exec env SHOWO_INIT_CHECKPOINT="$showo_init_checkpoint" FULL_STEPS="$full_steps" RESUME_FULL=0 \
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}" \
    python scripts/launch_pretrain.py "${dry_run[@]}" "$config" "$@"
