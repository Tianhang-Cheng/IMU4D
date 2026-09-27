#!/usr/bin/env bash
# Continued Show-o full pretraining with synthetic-IMU device-realism
# augmentation (configs/showo_pretrain_full_noise.yaml).
# Initializes from the latest complete checkpoint of exp/showo_pretrain_full.
#
# New:    TRAIN_GPU_ID=5 bash scripts/fast_train/stage2_noise_aug.sh
# Resume: TRAIN_GPU_ID=5 RESUME_FULL=1 bash scripts/fast_train/stage2_noise_aug.sh
#         (FULL_STEPS is an ABSOLUTE step target and must exceed the resumed
#          checkpoint's step; default 24000)
# Other source checkpoint:
#         SHOWO_INIT_CHECKPOINT=exp/showo_pretrain_full/checkpoint-40000 bash scripts/fast_train/stage2_noise_aug.sh
# Overrides:
#         EXTRA_OVERRIDES="training.imu_noise.acc_white_std=0.3" bash scripts/fast_train/stage2_noise_aug.sh
# Output: FULL_DIR (default exp/showo_pretrain_full_noise)
# Validate only:
#         DRY_RUN=1 bash scripts/fast_train/stage2_noise_aug.sh

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../_imu4d_env.bash"

# Random world-yaw augmentation: stage 2 canonicalizes each training window to a
# uniformly random horizontal heading instead of the GT-pelvis one (every
# world-frame field -- IMU channels, root orient/transl, objects -- rotated by the
# same Ry). Stage 1 leaves it off (training.imu_dataset defaults the flag to 0),
# so the heading invariance is learned together with the device-realism noise.
export IMU4D_YAW_AUG="${IMU4D_YAW_AUG:-1}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
config="${CONFIG:-configs/showo_pretrain_full_noise.yaml}"
full_steps="${FULL_STEPS:-24000}"
resume_full="${RESUME_FULL:-0}"
showo_init_source="${SHOWO_INIT_CHECKPOINT:-exp/showo_pretrain_full}"
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

echo "Starting Show-o noise-augmented pretraining from: $showo_init_checkpoint"
exec env SHOWO_INIT_CHECKPOINT="$showo_init_checkpoint" FULL_STEPS="$full_steps" RESUME_FULL=0 \
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}" \
    python scripts/launch_pretrain.py "${dry_run[@]}" "$config" "$@"
