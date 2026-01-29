#!/bin/bash
#SBATCH -J m3_remove_sentences_coarse            # Job name
#SBATCH -o m3_remove_sentences_coarse.out        # Name of stdout output log file
#SBATCH -e m3_remove_sentences_coarse.err        # Name of stderr output log file
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

# Run remove_sentences perturbation with all percentage values for coarse level
echo "Running remove_sentences perturbation with all percentage values (30%, 50%, 70%) for coarse level..."
echo "========================================"

python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation remove_sentences \
  --all-remove-pct

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
