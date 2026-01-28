#!/bin/bash
# Setup script for RadEval experiments

set -e

echo "========================================="
echo "Setting up RadEval experiment pipeline"
echo "========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p output/radeval/{original_ratings,perturbations,experiment_results/baseline}
mkdir -p external

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install datasets

# Clone RadEval repository
echo ""
echo "Cloning RadEval repository for GREEN metric..."
if [ ! -d "external/RadEval" ]; then
    git clone https://github.com/jbdel/RadEval.git external/RadEval
    echo "✓ RadEval repository cloned"
else
    echo "✓ RadEval repository already exists"
fi

# Download RadEval dataset
echo ""
echo "Downloading RadEval Expert Dataset..."
cd code
python download_radeval_data.py --output-dir ../data
cd ..

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review RADEVAL_README.md for usage instructions"
echo "2. Integrate GREEN evaluation code from external/RadEval"
echo "3. Run experiments with: python code/run_radeval_experiments.py"
