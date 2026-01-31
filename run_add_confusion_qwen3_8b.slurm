#!/bin/bash
#SBATCH -J m3_add_confusion_qwen3_8b       # Job name
#SBATCH -o m3_add_confusion_qwen3_8b.out   # Name of stdout output log file
#SBATCH -e m3_add_confusion_qwen3_8b.err   # Name of stderr output log file
#SBATCH -N 1                                # Total number of nodes requested
#SBATCH -n 1                                # Total number of cores requested
#SBATCH --cpus-per-task=8                   # Total number of cores requested per task
#SBATCH --get-user-env                      # retrieve the users login environment
#SBATCH --mail-type=ALL                     # Request status by email
#SBATCH --mail-user=fb265@cornell.edu       # Email address to send results to
#SBATCH --mem=52000                         # Total amount of memory requested in MB (per node)
#SBATCH -t 72:00:00                         # Time limit (72 hours)
#SBATCH --gres=gpu:1                        # Specify GPU type and number
#SBATCH --partition=luxlab                  # Request partition for resource allocation

# Print job info
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Navigate to project directory
cd $HOME/m3-eval || { echo "Failed to cd to m3-eval"; exit 1; }

# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Verify output directory exists
OUTPUT_DIR="output/cqa_eval"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo "========================================"

# Run add_confusion perturbation for coarse level with Qwen3-8B
echo "Running add_confusion perturbation (coarse level)..."
echo "========================================"

python -u code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation add_confusion \
  --seed 42

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Pipeline completed successfully!"
    echo "Checking output files..."

    # List generated files
    echo "Perturbation files:"
    ls -lh $OUTPUT_DIR/perturbations/add_confusion_coarse.jsonl 2>/dev/null || echo "  (none found)"

    echo "Experiment result files:"
    find $OUTPUT_DIR/experiment_results/baseline/add_confusion -name "*coarse*Qwen3-8B*" -ls 2>/dev/null || echo "  (none found)"

    echo "Job ended at: $(date)"
    echo "========================================"
else
    echo "Pipeline failed with error code $?"
    exit 1
fi
