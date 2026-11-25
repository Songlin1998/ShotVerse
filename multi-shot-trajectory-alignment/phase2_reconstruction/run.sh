#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG_FILE="${PROJECT_ROOT}/configs/pipeline_config.yaml"
config_vars=$(python3 "$PROJECT_ROOT/tools/parse_config.py" "$CONFIG_FILE")
eval "$config_vars"

ENV_SAM2="${CFG_ENVIRONMENTS_SAM2:-gd-sam2}"
ENV_PI3="${CFG_ENVIRONMENTS_PI3:-pi3}"
INPUT_ROOT=""
OUTPUT_ROOT="${CFG_PATHS_OUTPUT_ROOT:-./data/output_scenes}"
SAM2_REPO="${CFG_PATHS_SAM2_REPO:-./external/Grounded-SAM-2}"
SAM2_CHECKPOINT="${CFG_PATHS_SAM2_CHECKPOINT}"
SAM2_CONFIG="${CFG_SAM2_MODEL_CONFIG:-configs/sam2.1/sam2.1_hiera_l}"
GROUNDING_MODEL="${CFG_SAM2_GROUNDING_MODEL:-IDEA-Research/grounding-dino-tiny}"
TEXT_PROMPT="${CFG_SAM2_TEXT_PROMPT:-person.}"
CONF_THRESHOLD="${CFG_PI3_CONF_THRESHOLD:-0.1}"
PI3_CHECKPOINT="${CFG_PI3_CHECKPOINT}"
NUM_FRAMES="${CFG_SAMPLING_NUM_FRAMES:-5}"
SKIP_FIRST="${CFG_SAMPLING_SKIP_FIRST_FRAMES:-10}"
FORCE="${CFG_RUNTIME_FORCE_RERUN:-false}"
DEVICE="${CFG_GPU_DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CFG_GPU_CUDA_VISIBLE_DEVICES:-0}"

SKIP_STAGES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)      DEVICE="$2"; shift 2 ;;
        --input)       INPUT_ROOT="$2"; shift 2 ;;
        --output)      OUTPUT_ROOT="$2"; shift 2 ;;
        --skip_stage)  SKIP_STAGES="${SKIP_STAGES},$2"; shift 2 ;;
        --force)       FORCE="true"; shift ;;
        -h|--help)
            echo "Usage: bash phase2_reconstruction/run.sh [options]"
            echo "  --input DIR        Input directory"
            echo "  --output DIR       Output directory"
            echo "  --skip_stage N     Skip stage N (1-4)"
            echo "  --force            Force rerun all stages"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Phase 1 writes the split clips and full-frame pose JSONs under output_root.
INPUT_ROOT=$(cd "${INPUT_ROOT:-$OUTPUT_ROOT}" && pwd)
mkdir -p "$OUTPUT_ROOT"
# Keep the directory lock on the shell's open descriptor across subprocesses.
exec 9>>"$OUTPUT_ROOT/.alignment.lock"
python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX)'
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
shopt -s nullglob
scenes=("$INPUT_ROOT"/scene_*/)
(( ${#scenes[@]} )) || { echo "No scene directories in $INPUT_ROOT" >&2; exit 1; }

should_skip() { echo "$SKIP_STAGES" | grep -q ",$1"; }

eval "$(conda shell.bash hook)"

# Build each stage on the destination filesystem. A retry can recover a swap
# interrupted after the previous directory was moved aside.
publish_stage() {
    local next=$1 target=$2 previous="$SCENE_OUT/.previous-${2##*/}"
    touch "$next/.complete"
    if [[ -d "$target" ]]; then mv "$target" "$previous"; fi
    mv "$next" "$target"
    rm -rf "$previous"
}

run_dir=""
cleanup_run() { if [[ -n "$run_dir" ]]; then rm -rf "$run_dir"; fi; }
trap cleanup_run EXIT

failed=0
for scene_path in "${scenes[@]}"; do
    [ ! -d "$scene_path" ] && continue
    scene_name=$(basename "$scene_path")
    echo "Processing: $scene_name"

    SCENE_OUT="${OUTPUT_ROOT}/${scene_name}"
    S1="${SCENE_OUT}/stage1_sam2"
    S2="${SCENE_OUT}/stage2_pi3"
    S3="${SCENE_OUT}/stage3_aligned"
    S4="${SCENE_OUT}/stage4_merged"
    mkdir -p "$SCENE_OUT"
    cleanup_run
    run_dir=$(mktemp -d "$SCENE_OUT/.alignment-run.XXXXXXXX")
    for target in "$S1" "$S2" "$S3" "$S4"; do
        previous="$SCENE_OUT/.previous-${target##*/}"
        if [[ -d "$previous" ]]; then
            if [[ -d "$target" ]]; then rm -rf "$previous"; else mv "$previous" "$target"; fi
        fi
    done

    # ---- Stage 1: SAM2 Segmentation ----
    if ! should_skip 1; then
        if [[ "$FORCE" = "true" || ! -f "$S1/.complete" ]]; then
            rm -f "$S1/.complete" "$S2/.complete" "$S3/.complete" "$S4/.complete"
            conda activate "$ENV_SAM2"
            export PYTHONPATH="$SAM2_REPO${PYTHONPATH:+:$PYTHONPATH}"
            sam_device_args=()
            [[ "$DEVICE" != cpu ]] || sam_device_args+=(--force-cpu)
            python "${PROJECT_ROOT}/phase2_reconstruction/stage1_sam2_segmentation/sam2_segmentation.py" \
                --grounding-model "$GROUNDING_MODEL" \
                --text-prompt "$TEXT_PROMPT" \
                --video-dir "$scene_path" \
                --output-base-dir "$run_dir/s1" \
                --num-frames "$NUM_FRAMES" \
                --skip-first-frames "$SKIP_FIRST" \
                --sam2-checkpoint "$SAM2_CHECKPOINT" \
                --sam2-model-config "$SAM2_CONFIG" "${sam_device_args[@]}" \
            || { echo "Stage 1 failed: $scene_name" >&2; failed=1; continue; }
            publish_stage "$run_dir/s1" "$S1"
        fi
    fi

    # ---- Stage 2-4: pi3 env ----
    conda activate "$ENV_PI3"

    # ---- Stage 2: Pi3 Reconstruction ----
    if ! should_skip 2; then
        if [[ "$FORCE" = "true" || ! -f "$S2/.complete" ]]; then
            rm -f "$S2/.complete" "$S3/.complete" "$S4/.complete"
            PI3_INPUT="$run_dir/pi3_input"
            mkdir -p "$PI3_INPUT/$scene_name" "$run_dir/s2"
            for shot_dir in "$S1"/*/; do
                [ ! -d "$shot_dir" ] && continue
                shot_name=$(basename "$shot_dir")
                clip_dir="$PI3_INPUT/$scene_name/$shot_name"
                mkdir -p "$clip_dir"
                find "$shot_dir" -name "*_black_mask.jpg" -exec ln -sf {} "$clip_dir/" \;
            done

            python "${PROJECT_ROOT}/phase2_reconstruction/stage2_pi3_reconstruction/pi3_reconstruction.py" \
                --root-dir "$PI3_INPUT" \
                --device "$DEVICE" \
                --conf-threshold "$CONF_THRESHOLD" \
                --ckpt "$PI3_CHECKPOINT" \
            || { echo "Stage 2 failed: $scene_name" >&2; failed=1; continue; }

            shots=("$S1"/*/)
            poses=("$PI3_INPUT/$scene_name"/*/*_poses.json)
            if (( ${#shots[@]} == 0 || ${#shots[@]} != ${#poses[@]} )) || [[ ! -s "$PI3_INPUT/$scene_name/$scene_name.glb" ]]; then
                echo "Incomplete Pi3 reconstruction for $scene_name" >&2
                failed=1
                continue
            fi
            mv "$PI3_INPUT/$scene_name/$scene_name.glb" "${poses[@]}" "$run_dir/s2/"
            publish_stage "$run_dir/s2" "$S2"
        fi
    fi

    # ---- Stage 3: Pose Alignment ----
    if ! should_skip 3; then
        if [[ "$FORCE" = "true" || ! -f "$S3/.complete" ]]; then
            rm -f "$S3/.complete" "$S4/.complete"
            python "${PROJECT_ROOT}/phase2_reconstruction/stage3_pose_alignment/align_all_poses.py" \
                --pi3-root "$S2" \
                --full-pose-root "$scene_path" \
                --output-root "$run_dir/s3" \
            || { echo "Stage 3 failed: $scene_name" >&2; failed=1; continue; }
            publish_stage "$run_dir/s3" "$S3"
        fi
    fi

    # ---- Stage 4: Trajectory Merging ----
    if ! should_skip 4; then
        if [[ "$FORCE" = "true" || ! -f "$S4/.complete" ]]; then
            rm -f "$S4/.complete"
            manifest_args=()
            if [[ -f "$scene_path/trajectory.json" ]]; then
                manifest_args=(--shot-manifest "$scene_path/trajectory.json")
            fi
            python "${PROJECT_ROOT}/phase2_reconstruction/stage4_trajectory_merging/merge_trajectories.py" \
                --input-dir "$S3" \
                --output-dir "$run_dir/s4" "${manifest_args[@]}" \
            || { echo "Stage 4 failed: $scene_name" >&2; failed=1; continue; }
            publish_stage "$run_dir/s4" "$S4"
        fi
    fi
done

(( failed == 0 )) || exit 1
echo "Phase 2 done. Results: $OUTPUT_ROOT"
