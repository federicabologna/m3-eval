#!/bin/bash
# M3-Eval Quick Setup Script

set -e  # Exit on error

echo "=========================================="
echo "M3-Eval Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install PyTorch with CUDA 11.8 (for GPU compatibility)
echo ""
echo "Installing PyTorch 2.7.1 with CUDA 11.8..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118 --quiet
echo "✓ PyTorch installed"

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Download spacy model
echo ""
echo "Downloading spaCy medical NER model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz --quiet
echo "✓ Medical NER model installed"

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p output
echo "✓ Output directory created"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "⚠ No .env file found"
    echo ""
    echo "If you plan to use API models (GPT, Claude, Gemini), create a .env file:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your API keys"
else
    echo "✓ .env file found"
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>/dev/null || echo "⚠ Could not check GPU"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Run test experiment: python code/experiment_runner.py --experiment baseline --model Qwen3-8B --level coarse"
echo "  3. Check output: ls output/"
echo ""
echo "For more information, see README.md"
