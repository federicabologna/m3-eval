#!/bin/bash
#SBATCH -J m3_fine_qwen3_8b                  # Job name
#SBATCH -o m3_fine_qwen3_8b.out              # Name of stdout output log file
#SBATCH -e m3_fine_qwen3_8b.err              # Name of stderr output log file
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --cpus-per-task=8                    # Total number of cores requested per task
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=fb265@cornell.edu        # Email address to send results to
#SBATCH --mem=52000                          # Total amount of memory requested in MB (per node)
#SBATCH -t 72:00:00                          # Time limit (72 hours)
#SBATCH --gres=gpu:1                         # Specify GPU type and number
#SBATCH --partition=luxlab                   # Request partition for resource allocation

# Print job info
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Navigate to project directory
cd $HOME/m3-eval || { echo "Failed to cd to m3-eval"; exit 1; }

# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Run fine level experiments with Qwen3-8B
# This will:
# 1. Generate original ratings for fine subset (if not exists)
# 2. Run change_dosage perturbation with all parameter variations
echo "Running fine level experiments with Qwen3-8B..."
echo "========================================"

python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level fine \
  --seed 42

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
