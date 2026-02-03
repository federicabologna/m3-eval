#!/bin/bash
#SBATCH -J medinfo_change_dosage_qwen3                  # Job name
#SBATCH -o logs/medinfo_change_dosage_qwen3.out         # Name of stdout output log file
#SBATCH -e logs/medinfo_change_dosage_qwen3.err         # Name of stderr output log file
#SBATCH -N 1                                            # Total number of nodes requested
#SBATCH -n 1                                            # Total number of cores requested
#SBATCH --cpus-per-task=8                               # Total number of cores requested per task
#SBATCH --get-user-env                                  # retrieve the users login environment
#SBATCH --mail-type=ALL                                 # Request status by email
#SBATCH --mail-user=fb265@cornell.edu                   # Email address to send results to
#SBATCH --mem=52000                                     # Total amount of memory requested in MB (per node)
#SBATCH -t 72:00:00                                     # Time limit (72 hours)
#SBATCH --gres=gpu:1                                    # Specify GPU type and number
#SBATCH --partition=luxlab                              # Request partition for resource allocation

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

# Run MedInfo change_dosage experiments with Qwen3-8B
# This will:
# 1. Generate change_dosage perturbations using full MedInfo dataset (690 coarse / 2,311 fine)
# 2. Generate original ratings for IDs with successful perturbations
# 3. Rate perturbed answers
echo "Running MedInfo change_dosage experiments with Qwen3-8B..."
echo "Dataset: MedInfo2019-QA-Medications"
echo "Perturbation: change_dosage"
echo "Levels: both (coarse + fine)"
echo "========================================"

# Verify output directory exists
OUTPUT_DIR="output/medinfo"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "Output directory: $(pwd)/$OUTPUT_DIR"
echo "========================================"

python -u code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level both \
  --seed 42 \
  --dataset medinfo \
  --perturbation change_dosage

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Pipeline completed successfully!"
    echo "Checking output files..."

    # List generated files
    echo ""
    echo "Original ratings files:"
    ls -lh $OUTPUT_DIR/original_ratings/*Qwen3-8B* 2>/dev/null || echo "  (none found)"

    echo ""
    echo "Perturbation files:"
    ls -lh $OUTPUT_DIR/perturbations/change_dosage* 2>/dev/null || echo "  (none found)"

    echo ""
    echo "Experiment result files:"
    find $OUTPUT_DIR/experiment_results/baseline/change_dosage -name "*Qwen3-8B*" -ls 2>/dev/null || echo "  (none found)"

    echo ""
    echo "Summary statistics:"
    echo "  Coarse perturbations:"
    wc -l $OUTPUT_DIR/perturbations/change_dosage_coarse.jsonl 2>/dev/null || echo "    (file not found)"
    echo "  Fine perturbations:"
    wc -l $OUTPUT_DIR/perturbations/change_dosage_fine.jsonl 2>/dev/null || echo "    (file not found)"
    echo "  Coarse original ratings:"
    wc -l $OUTPUT_DIR/original_ratings/original_coarse_Qwen3-8B_rating.jsonl 2>/dev/null || echo "    (file not found)"
    echo "  Fine original ratings:"
    wc -l $OUTPUT_DIR/original_ratings/original_fine_Qwen3-8B_rating.jsonl 2>/dev/null || echo "    (file not found)"

    echo ""
    echo "Job ended at: $(date)"
    echo "========================================"
else
    echo "Pipeline failed with error code $?"
    exit 1
fi
