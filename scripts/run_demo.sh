#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=sonata_pca
#SBATCH --output=results/logs/pca_%j.out

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Running PCA Demo..."
cd "$REPO_ROOT" || exit 1
mkdir -p results/logs
source "$REPO_ROOT/venv/bin/activate"

cd "$REPO_ROOT/sonata-article" || exit 1

# Mandatory for Sonata imports to work inside the upstream repo
export PYTHONPATH=./

# Run the upstream PCA demo
python demo/0_pca.py
