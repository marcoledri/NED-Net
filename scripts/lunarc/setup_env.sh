#!/bin/bash
# ============================================================
# LUNARC COSMOS — One-time environment setup for BENDR
# ============================================================
# Run this ONCE after first login to set up the conda environment.
#
# Usage:
#   ssh cosmos.lunarc.lu.se
#   cd ~/eeg-seizure-shared
#   bash scripts/lunarc/setup_env.sh
# ============================================================

set -euo pipefail

echo "=== Setting up BENDR environment on COSMOS ==="

# Load Anaconda
module purge
module load Anaconda3/2024.06-1
source config_conda.sh

# Create conda environment with PyTorch + CUDA 12.1
echo "Creating conda environment 'bendr'..."
conda create -y -n bendr python=3.11

echo "Activating environment..."
conda activate bendr

echo "Installing PyTorch with CUDA 12.1..."
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing dependencies..."
pip install pyedflib scipy numpy tqdm

echo "Installing eeg-seizure-analyzer in editable mode..."
cd ~/eeg-seizure-shared
pip install -e "."

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate in future sessions or job scripts:"
echo "  module load Anaconda3/2024.06-1"
echo "  source config_conda.sh"
echo "  conda activate bendr"
echo ""
echo "Quick test (should print CUDA device info):"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
