#!/bin/bash

# Install dependencies for QQP triplets translation script
# Optimized for Tesla P40 with CUDA 12.2

set -e  # Exit on any error

echo "Installing dependencies for QQP triplets translation..."

# Update package list and install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv qqp_translate_env
source qqp_translate_env/bin/activate

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2)
echo "Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core Python packages
echo "Installing core Python packages..."
pip install pandas openpyxl

# Install Hugging Face libraries
echo "Installing Hugging Face transformers and datasets..."
pip install transformers datasets

echo "Installing other"
pip install sacremoses sentencepiece

# Install additional utilities
echo "Installing additional utilities..."
pip install tqdm  # For progress bars
pip install accelerate  # For better performance

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import pandas as pd; print(f'Pandas version: {pd.__version__}')"
python -c "from transformers import pipeline; print('Transformers installed successfully')"

echo ""
echo "Installation completed successfully!"
echo ""
echo "To use the environment:"
echo "source qqp_translate_env/bin/activate"
echo ""
echo "Your GPU information:"
echo "- Tesla P40 (24GB VRAM)"
echo "- CUDA 12.2 compatible"
echo "- PyTorch installed with CUDA 12.1 support (compatible with 12.2)"
echo ""
echo "Note: The script will use the Helsinki-NLP/opus-mt-en-es model by default"
echo "      which will be automatically downloaded on first run."
