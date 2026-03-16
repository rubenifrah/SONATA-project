#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --account=m25146
#SBATCH --job-name=install_env
#SBATCH --output=logs/install_%j.out

echo "--- STARTING INSTALLATION ---"
mkdir -p logs

# 1. Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip ninja

# 2. Install PyTorch (2.5.0 for CUDA 12.4)
echo "Installing PyTorch..."
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install Sonata Core (Spconv & Flash Attention)
echo "Installing Core Libs..."
pip install spconv-cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install flash-attn --no-build-isolation

# 4. Install Demo Dependencies (Open3D, etc.)
echo "Installing Demo Libs..."
# Note: numpy<2.0 is critical for Open3D
pip install open3d fast_pytorch_kmeans psutil "numpy<2.0" timm addict scipy

echo "--- INSTALLATION COMPLETE ---"
python -c "import torch; import spconv.pytorch; print('SUCCESS: Environment is ready.')"