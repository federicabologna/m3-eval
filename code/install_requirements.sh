#!/bin/bash

echo "Installing transformers and dependencies for Mac..."

# Install PyTorch with MPS support for Mac
pip install torch torchvision torchaudio

# Install transformers and accelerate
pip install transformers accelerate

echo "Installation complete! You can now run: python test_inference.py"
