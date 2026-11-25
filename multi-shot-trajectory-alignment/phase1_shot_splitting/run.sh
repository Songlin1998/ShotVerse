#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load config
CONFIG_FILE="${PROJECT_ROOT}/configs/pipeline_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found. Run: cp configs/pipeline_config.yaml.example configs/pipeline_config.yaml"
    exit 1
fi
config_vars=$(python3 "$PROJECT_ROOT/tools/parse_config.py" "$CONFIG_FILE")
eval "$config_vars"

ENV_NAME="${CFG_ENVIRONMENTS_PI3:-pi3}"
INPUT_DIR="${CFG_PATHS_INPUT_ROOT:-./data/input_scenes}"
OUTPUT_DIR="${CFG_PATHS_OUTPUT_ROOT:-./data/output_scenes}"
DEVICE="${CFG_GPU_DEVICE:-cuda}"

# CLI overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)  INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash phase1_shot_splitting/run.sh [--input DIR] [--output DIR] [--device DEV]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[ ! -d "$INPUT_DIR" ] && { echo "Error: input dir not found: $INPUT_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR"
# Keep the directory lock on the shell's open descriptor across subprocesses.
exec 9>>"$OUTPUT_DIR/.alignment.lock"
python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX)'

# Activate env
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CFG_GPU_CUDA_VISIBLE_DEVICES:-0}"
TOTAL=0; SUCCESS=0; FAILED=0

for scene_dir in "$INPUT_DIR"/scene_*; do
    [ ! -d "$scene_dir" ] && continue

    scene_name=$(basename "$scene_dir")
    video_path="$scene_dir/camera_1_recon.mp4"
    trajectory_path="$scene_dir/trajectory.json"
    output_scene="$OUTPUT_DIR/$scene_name"

    TOTAL=$((TOTAL + 1))

    if [[ ! -f "$video_path" || ! -f "$trajectory_path" ]]; then
        echo "Missing video or trajectory.json in $scene_dir" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    if python phase1_shot_splitting/pipeline_shot_trajectory.py \
        --video_path "$video_path" \
        --trajectory_path "$trajectory_path" \
        --output_root "$output_scene" \
        --device "$DEVICE" --resume; then
        manifest_tmp=$(mktemp "$output_scene/.manifest.XXXXXXXX")
        cp "$trajectory_path" "$manifest_tmp"
        mv -f "$manifest_tmp" "$output_scene/trajectory.json"
        [ -f "$scene_dir/global_prompt.txt" ] && cp "$scene_dir/global_prompt.txt" "$output_scene/"
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo "Phase 1 done: total=$TOTAL success=$SUCCESS failed=$FAILED"
(( TOTAL > 0 && FAILED == 0 ))
