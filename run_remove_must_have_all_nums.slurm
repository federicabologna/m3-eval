#!/bin/bash
#SBATCH -J m3_remove_sentences_all_pcts               # Job name
#SBATCH -o m3_remove_sentences_all_pcts.out           # Name of stdout output log file
#SBATCH -e m3_remove_sentences_all_pcts.err           # Name of stderr output log file
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --cpus-per-task=8                    # Total number of cores requested per task
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=fb265@cornell.edu        # Email address to send results to
#SBATCH --mem=100000                          # Total amount of memory requested in MB (per node)
#SBATCH -t 72:00:00                          # Time limit (48 hours)
#SBATCH --gres=gpu:2                         # Specify GPU type and number
#SBATCH --partition=gpu                   # Request partition for resource allocation

# Print job info
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Navigate to project directory
cd $HOME/m3-eval || { echo "Failed to cd to m3-eval"; exit 1; }

# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Fix NCCL compatibility issue - use PyTorch's bundled NCCL
export LD_LIBRARY_PATH=$HOME/m3-eval/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Run remove_sentences with all percentage values (0.3, 0.5, 0.7)
echo "Running remove_sentences perturbation with all percentage values..."
echo "This will generate 3 files per level (coarse/fine) with 30%, 50%, 70% removed"
echo "========================================"

python code/perturbation_pipeline.py --model Qwen3-8B --perturbation remove_sentences --all-remove-pct

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Pipeline completed successfully!"
    echo "Job ended at: $(date)"
    echo "========================================"
else
    echo "Pipeline failed with error code $?"
    exit 1
fi
