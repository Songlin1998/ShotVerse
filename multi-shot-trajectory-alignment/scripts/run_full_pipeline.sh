#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SKIP_PHASE1=false
SKIP_PHASE2=false
INPUT_OVERRIDE=""
PHASE1_ARGS=()
PHASE2_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_phase1) SKIP_PHASE1=true; shift ;;
        --skip_phase2) SKIP_PHASE2=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/run_full_pipeline.sh [options]"
            echo "  --skip_phase1    Skip Phase 1 (shot splitting)"
            echo "  --skip_phase2    Skip Phase 2 (reconstruction + evaluation)"
            exit 0 ;;
        --input) INPUT_OVERRIDE="${2:?missing input path}"; shift 2 ;;
        --device) PHASE1_ARGS+=("$1" "${2:?missing device}"); PHASE2_ARGS+=("$1" "$2"); shift 2 ;;
        --output) PHASE1_ARGS+=("$1" "${2:?missing output path}"); PHASE2_ARGS+=("$1" "$2"); shift 2 ;;
        --skip_stage) PHASE2_ARGS+=("$1" "${2:?missing stage number}"); shift 2 ;;
        --force) PHASE2_ARGS+=("$1"); shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$INPUT_OVERRIDE" ]]; then
    if [[ "$SKIP_PHASE1" = true ]]; then
        PHASE2_ARGS+=(--input "$INPUT_OVERRIDE")
    else
        PHASE1_ARGS+=(--input "$INPUT_OVERRIDE")
    fi
fi

START_TIME=$(date +%s)

if [ "$SKIP_PHASE1" = false ]; then
    bash phase1_shot_splitting/run.sh "${PHASE1_ARGS[@]}"
fi

if [ "$SKIP_PHASE2" = false ]; then
    bash phase2_reconstruction/run.sh "${PHASE2_ARGS[@]}"
fi

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "All done. Elapsed: $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
