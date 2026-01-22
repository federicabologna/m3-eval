#!/bin/bash
#SBATCH -J m3_remove_all_nums               # Job name
#SBATCH -o m3_remove_all_nums.out           # Name of stdout output log file
#SBATCH -e m3_remove_all_nums.err           # Name of stderr output log file
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=fb265@cornell.edu        # Email address to send results to
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                 # Total number of cores requested
#SBATCH --mem=100000                         # Total amount of memory requested in MB (per node)
#SBATCH -t 48:00:00                          # Time limit (48 hours)
#SBATCH --gres=gpu:nvidia_rtx_a6000:1        # Specify GPU type and number
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

# Run remove_must_have with all num_remove values (1, 2, 3)
echo "Running remove_must_have perturbation with all num_remove values..."
echo "This will generate 3 files per level (coarse/fine) with removed=1, 2, 3"
echo "========================================"

python code/perturbation_pipeline.py --model Qwen3-8B --perturbation remove_must_have --all_num_remove

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
