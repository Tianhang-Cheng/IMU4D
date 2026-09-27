#!/usr/bin/env bash
# Stage 1: public Show-o weights -> adapter warm-up -> IMU4D full pretraining.
# The default target is 500,000 optimizer steps on one GPU.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../_imu4d_env.bash"

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/1_gpus.yaml}"
warmup_steps="${ADAPTER_WARMUP_STEPS:-1000}"
full_steps="${FULL_STEPS:-500000}"
warmup_dir="${ADAPTER_WARMUP_DIR:-exp/showo_adapter_warmup}"
full_dir="${FULL_DIR:-exp/showo_pretrain_full}"
dry_run=()
[[ "${DRY_RUN:-0}" == "1" ]] && dry_run=(--dry-run)

for value in "$warmup_steps" "$full_steps"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Step targets must be positive integers." >&2; exit 2; }
done
[[ -f pretrained_weight/showo/pytorch_model.safetensors ]] || {
    echo "Missing public Show-o weights: pretrained_weight/showo/pytorch_model.safetensors" >&2
    exit 1
}

latest_complete_checkpoint() {
    local root="$1" checkpoint step best=-1 result=""
    shopt -s nullglob
    for checkpoint in "$root"/checkpoint-*; do
        [[ -f "$checkpoint/unwrapped_model/pytorch_model.bin" ]] || continue
        step="${checkpoint##*-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if (( step > best )); then best="$step"; result="$checkpoint"; fi
    done
    shopt -u nullglob
    [[ -n "$result" ]] && printf '%s\n' "$result"
}

if [[ "${RESUME_FULL:-0}" == "1" ]]; then
    exec env FULL_DIR="$full_dir" FULL_STEPS="$full_steps" RESUME_FULL=1 \
        python scripts/launch_pretrain.py "${dry_run[@]}" configs/showo_pretrain_full.yaml "$@"
fi

warmup_checkpoint="$(latest_complete_checkpoint "$warmup_dir" || true)"
warmup_step="${warmup_checkpoint##*-}"
if [[ -z "$warmup_checkpoint" || "$warmup_step" -lt "$warmup_steps" ]]; then
    warmup_resume=0
    [[ -n "$warmup_checkpoint" ]] && warmup_resume=1
    env ADAPTER_WARMUP_DIR="$warmup_dir" FULL_STEPS="$warmup_steps" RESUME_FULL="$warmup_resume" \
        python scripts/launch_pretrain.py "${dry_run[@]}" configs/showo_adapter_warmup.yaml
    [[ "${DRY_RUN:-0}" == "1" ]] && exit 0
    warmup_checkpoint="$(latest_complete_checkpoint "$warmup_dir" || true)"
fi
[[ -n "$warmup_checkpoint" ]] || {
    echo "No complete adapter warm-up checkpoint under $warmup_dir" >&2
    exit 1
}

exec env SHOWO_INIT_CHECKPOINT="$warmup_checkpoint" FULL_DIR="$full_dir" \
    FULL_STEPS="$full_steps" RESUME_FULL=0 \
    EXTRA_OVERRIDES="experiment.strict_resume=True experiment.reset_object_modules_on_weight_load=False ${EXTRA_OVERRIDES:-}" \
    python scripts/launch_pretrain.py "${dry_run[@]}" configs/showo_pretrain_full.yaml "$@"
