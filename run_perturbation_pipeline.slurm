#!/bin/bash
#SBATCH -J m3_typos_all_probs               # Job name
#SBATCH -o m3_typos_all_probs.out           # Name of stdout output log file
#SBATCH -e m3_typos_all_probs.err           # Name of stderr output log file
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --cpus-per-task=8                    # Total number of cores requested per task
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=fb265@cornell.edu        # Email address to send results to
#SBATCH --mem=52000                          # Total amount of memory requested in MB (per node)
#SBATCH -t 72:00:00                          # Time limit (48 hours)
#SBATCH --gres=gpu:1                         # Specify GPU type and number
#SBATCH --partition=luxlab                   # Request partition for resource allocation

# Print job info
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Print GPU info
echo "GPU Information:"
nvidia-smi -L
echo "========================================"

# Navigate to project directory
cd $HOME/m3-eval || { echo "Failed to cd to m3-eval"; exit 1; }

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Fix NCCL compatibility issue - use PyTorch's bundled NCCL
export LD_LIBRARY_PATH=$HOME/m3-eval/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Print Python and package info
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"

# Run the perturbation pipeline
echo "Starting perturbation pipeline with Qwen3-8B..."
echo "Running all perturbations: add_typos, change_dosage, remove_sentences, add_confusion"
echo "========================================"

python code/perturbation_pipeline.py --model Qwen3-8B

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Pipeline completed successfully!"
    echo "Job ended at: $(date)"
    echo "========================================"
else
    echo "========================================"
    echo "Pipeline failed with error code $?"
    echo "Job ended at: $(date)"
    echo "========================================"
    exit 1
fi
