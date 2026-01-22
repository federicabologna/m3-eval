#!/bin/bash
#SBATCH -J m3_remove_all_nums               # Job name
#SBATCH -o m3_remove_all_nums_%j.out        # output file (%j expands to jobID)
#SBATCH -e m3_remove_all_nums_%j.err        # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --cpus-per-task=4                    # Total number of cores requested per task
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32000                          # server memory requested in MB (32GB)
#SBATCH -t 12:00:00                          # Time limit (12 hours)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1   # Type:number of GPUs needed

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
