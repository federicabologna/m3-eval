#!/bin/bash
#SBATCH -J m3_eval_qwen3-8b                  # Job name
#SBATCH -o m3_eval_qwen3-8b_%j.out          # output file (%j expands to jobID)
#SBATCH -e m3_eval_qwen3-8b_%j.err          # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --cpus-per-task=4                    # Total number of cores requested per task
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32000                          # server memory requested in MB (32GB for 8B model)
#SBATCH -t 24:00:00                          # Time limit (24 hours for full pipeline)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1   # Type:number of GPUs needed

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

# Print Python and package info
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"

# Run the perturbation pipeline
echo "Starting perturbation pipeline with Qwen3-8B..."
echo "Running all perturbations: add_typos, change_dosage, remove_must_have, add_confusion"
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
