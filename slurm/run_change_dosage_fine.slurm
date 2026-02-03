#!/bin/bash
#SBATCH -J m3_change_dosage_fine            # Job name
#SBATCH -o logs/m3_change_dosage_fine.out   # Name of stdout output log file
#SBATCH -e logs/m3_change_dosage_fine.err   # Name of stderr output log file
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

# Navigate to project directory (from slurm folder)
cd $HOME/m3-eval || { echo "Failed to cd to m3-eval"; exit 1; }
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Run change_dosage perturbation for fine level only
echo "Running change_dosage perturbation for fine level..."
echo "========================================"

python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level fine \
  --perturbation change_dosage

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
