#!/usr/bin/env bash
# Scene fine-tuning (HiPHI / OMOMO / HUMOTO) on the fast_train lineage, i.e.
# from exp/showo_pretrain_full -- the own-checkpoint pretraining trunk.
#
# Branches off PRETRAIN, not the noise-augmented trunk: noise augmentation
# exists to bridge to measured IMU, and these sets are clean synthetic IMU.
# See configs/train_iters.yaml (joint.finetune.scene.init_from: pretrain).
#
# Object supervision stays on -- that is the point of this stage; the configs
# keep training.finetune_on_specific_dataset false so the object heads are
# actually optimized (see configs/showo_finetune_scene_base.yaml).
#
#
# Full eval (every 2500 steps) dumps records-*.jsonl + tracks/*.npz, including
# the dataset's object motion/valid masks, so the dynamic-mesh metrics are a
# post-hoc step and never need another model pass:
#   python -m evaluation.score_scene_metrics <output_dir>/evaluation/full/step-N
# New:    TRAIN_GPU_ID=6 DATASET=hiphi bash scripts/fast_train/stage3_scene.sh
# Resume: TRAIN_GPU_ID=6 DATASET=hiphi RESUME_FULL=1 bash scripts/fast_train/stage3_scene.sh
# Output: FULL_DIR (default exp/showo_finetune_<dataset>)
# Geometry: disabled by default to match the released scene weights.
#         Set OBJ_GEOM=1 to opt in to ULIP-2 features for a new run.
#         Pass the same OBJ_GEOM* settings on resume.
# Validate only:
#         DRY_RUN=1 DATASET=hiphi bash scripts/fast_train/stage3_scene.sh

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../_imu4d_env.bash"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
dataset="${DATASET:-hiphi}"
case "$dataset" in
    hiphi|omomo|humoto) ;;
    *) echo "DATASET must be hiphi, omomo or humoto, got '$dataset'." >&2; exit 2 ;;
esac
config="${CONFIG:-configs/showo_finetune_${dataset}.yaml}"
# Object geometry conditioning is off by default.
# OBJ_GEOM=1 turns it on; OBJ_GEOM_FEATURES=<npz> picks other features (e.g.
# dataset_process/asset_geometry_bps.npz). OBJ_GEOM_TAG=<tag> gives the run its
# own output dir / name, exp/showo_finetune_<dataset>_<tag>.
if [[ "${OBJ_GEOM:-0}" == "0" ]]; then
    export OBJ_GEOM_FEATURES=null
else
    export OBJ_GEOM_FEATURES="${OBJ_GEOM_FEATURES:-dataset_process/asset_geometry_ulip.npz}"
    [[ -f "$OBJ_GEOM_FEATURES" ]] || {
        echo "Geometry features not found: $OBJ_GEOM_FEATURES (dataset_process/asset_geometry_{ulip,features}.py)" >&2
        exit 1
    }
fi
if [[ -n "${OBJ_GEOM_TAG:-}" ]]; then
    export FULL_DIR="${FULL_DIR:-exp/showo_finetune_${dataset}_${OBJ_GEOM_TAG}}"
    export EXTRA_OVERRIDES="experiment.name=showo_finetune_${dataset}_${OBJ_GEOM_TAG} ${EXTRA_OVERRIDES:-}"
fi
# Default step target comes from configs/train_iters.yaml so every caller agrees.
default_steps="$(python - "$dataset" <<'PYEOF'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/train_iters.yaml")
print(OmegaConf.select(cfg, f"joint.finetune.scene.{sys.argv[1]}"))
PYEOF
)"
full_steps="${FULL_STEPS:-$default_steps}"
resume_full="${RESUME_FULL:-0}"
showo_init_source="${SHOWO_INIT_CHECKPOINT:-exp/showo_pretrain_full}"
dry_run=()
[[ "${DRY_RUN:-0}" == "1" ]] && dry_run=(--dry-run)

[[ "$full_steps" =~ ^[1-9][0-9]*$ ]] || {
    echo "FULL_STEPS must be a positive integer, got '$full_steps'." >&2
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

echo "Starting Show-o scene ($dataset) fine-tuning from: $showo_init_checkpoint (target $full_steps steps)"
exec env SHOWO_INIT_CHECKPOINT="$showo_init_checkpoint" FULL_STEPS="$full_steps" RESUME_FULL=0 \
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}" \
    python scripts/launch_pretrain.py "${dry_run[@]}" "$config" "$@"
