#!/bin/bash
# GPU launcher for exporting dense SONATA features from one ScanNet-style scene.
#
# Usage examples:
#   INPUT_PATH=/path/to/scene.ply OUTPUT_PATH=results/features/scene0000_00.npz \
#     bash scripts/run_extract_scannet_features.sh
#
#   SAMPLE_NAME=sample1 OUTPUT_PATH=results/features/sample1.npz \
#     bash scripts/run_extract_scannet_features.sh
#
# For Juliet submissions where you specifically want an A100, prefer overriding
# resources at submission time, for example:
#   sbatch --gres=gpu:a100:1 scripts/run_extract_scannet_features.sh

#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=extract_sonata_feat
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT" || exit 1

INPUT_PATH="${INPUT_PATH:-}"
SAMPLE_NAME="${SAMPLE_NAME:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-sonata}"
REPO_ID="${REPO_ID:-facebook/sonata}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
MAX_POINTS="${MAX_POINTS:-0}"
SEED="${SEED:-53124}"
ESTIMATE_NORMALS="${ESTIMATE_NORMALS:-0}"
FORCE_DISABLE_FLASH="${FORCE_DISABLE_FLASH:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VENV_PATH="/home/rubifrah/sonata/venv"

SCRIPT_PATH="$REPO_ROOT/experiments/axis1/extract_scannet_features.py"

mkdir -p "$REPO_ROOT/results/logs"

if [ -z "$OUTPUT_PATH" ]; then
    echo "OUTPUT_PATH is required."
    echo "Example:"
    echo "  INPUT_PATH=/path/to/scene.ply OUTPUT_PATH=results/features/scene.npz bash scripts/run_extract_scannet_features.sh"
    exit 1
fi

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using active conda environment: $CONDA_DEFAULT_ENV"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
elif [ -d "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    echo "Activated virtual environment: $VENV_PATH"
else
    echo "No active Python environment detected."
    echo "Activate one first, or set VENV_PATH to the environment you want to use."
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/sonata-article:${PYTHONPATH:-}"

CMD=(
    python "$SCRIPT_PATH"
    --output-path "$OUTPUT_PATH"
    --model-name "$MODEL_NAME"
    --repo-id "$REPO_ID"
    --device "$DEVICE"
    --max-points "$MAX_POINTS"
    --seed "$SEED"
)

if [ -n "$INPUT_PATH" ] && [ -n "$SAMPLE_NAME" ]; then
    echo "Provide only one of INPUT_PATH or SAMPLE_NAME."
    exit 1
fi

if [ -n "$INPUT_PATH" ]; then
    CMD+=(--input-path "$INPUT_PATH")
elif [ -n "$SAMPLE_NAME" ]; then
    CMD+=(--sample-name "$SAMPLE_NAME")
else
    echo "Provide one of INPUT_PATH or SAMPLE_NAME."
    exit 1
fi

if [ -n "$DOWNLOAD_ROOT" ]; then
    CMD+=(--download-root "$DOWNLOAD_ROOT")
fi

if [ "$ESTIMATE_NORMALS" = "1" ]; then
    CMD+=(--estimate-normals)
fi

if [ "$FORCE_DISABLE_FLASH" = "1" ]; then
    CMD+=(--force-disable-flash)
fi

if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Running SONATA feature extraction"
echo "Script: $SCRIPT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Device: $DEVICE"
if [ -n "$INPUT_PATH" ]; then
    echo "Input path: $INPUT_PATH"
else
    echo "Sample name: $SAMPLE_NAME"
fi

"${CMD[@]}"
