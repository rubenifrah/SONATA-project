#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=sonata_pca
#SBATCH --output=logs/pca_%j.out

echo "Running PCA Demo..."
source venv/bin/activate

# Mandatory for Sonata imports to work
export PYTHONPATH=./

# Run the script (Correct path for a fresh clone)
python demo/0_pca.py