#!/bin/bash
# Dedicated launcher for the Axis 1 shortcut experiment.
#
# This wrapper keeps the invocation stable for both local runs and SLURM runs.
# By default it executes the synthetic sanity check, because that path should be
# validated before real scene-level feature exports are analyzed.

#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --account=m25146
#SBATCH --job-name=shortcut_test
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODE="${MODE:-synthetic}"
SCENE_LIMIT="${SCENE_LIMIT:-0}"
RESULTS_TAG="${RESULTS_TAG:-$(date +%Y%m%d_%H%M%S)}"
INPUT_GLOB="${INPUT_GLOB:-}"
FEATURE_KEY="${FEATURE_KEY:-feat}"
COORD_KEY="${COORD_KEY:-coord}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.7}"
SEED="${SEED:-7}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/venv}"

SCRIPT_PATH="$REPO_ROOT/experiments/axis1/shortcut_test.py"
RESULTS_DIR="$REPO_ROOT/results/shortcut_test/${RESULTS_TAG}"

mkdir -p "$REPO_ROOT/results/logs" "$RESULTS_DIR"

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

# The upstream SONATA package sits in sonata-article/, so include it in
# PYTHONPATH now even though the current script only requires NumPy.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/sonata-article:${PYTHONPATH:-}"

echo "Running shortcut analysis"
echo "Mode: $MODE"
echo "Script: $SCRIPT_PATH"
echo "Results: $RESULTS_DIR"
echo "Scene limit: $SCENE_LIMIT"

if [ "$MODE" = "npz" ] && [ -z "$INPUT_GLOB" ]; then
    echo "MODE=npz requires INPUT_GLOB, for example:"
    echo "  INPUT_GLOB='results/features/*.npz' bash scripts/run_shortcut_test.sh"
    exit 1
fi

CMD=(
    python "$SCRIPT_PATH"
    --results-dir "$RESULTS_DIR"
    --mode "$MODE"
    --scene-limit "$SCENE_LIMIT"
    --feature-key "$FEATURE_KEY"
    --coord-key "$COORD_KEY"
    --train-fraction "$TRAIN_FRACTION"
    --seed "$SEED"
)

if [ -n "$INPUT_GLOB" ]; then
    CMD+=(--input-glob "$INPUT_GLOB")
fi

if [ -n "$EXTRA_ARGS" ]; then
    # EXTRA_ARGS is kept as a simple escape hatch for manual experimentation.
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

"${CMD[@]}"
