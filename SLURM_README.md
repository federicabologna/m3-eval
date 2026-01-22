# SLURM Job Scripts for M3-Eval Pipeline

This directory contains SLURM job scripts for running the perturbation pipeline on a GPU cluster.

## Available Scripts

### 1. `run_perturbation_pipeline.slurm`
**Purpose**: Run the complete pipeline with all perturbations

**What it does**:
- Computes original ratings for all QA pairs (if not already done)
- Runs all 4 perturbations: add_typos, change_dosage, remove_must_have, add_confusion
- Uses default parameters (typo_prob=0.5, num_remove=1)

**Usage**:
```bash
sbatch run_perturbation_pipeline.slurm
```

**Resources**:
- 1 GPU (any available)
- 100GB RAM
- 72 hours
- 2 cores
- Partition: luxlab

---

### 2. `run_add_typos_all_probs.slurm`
**Purpose**: Run add_typos with all probability values

**What it does**:
- Runs add_typos perturbation with prob=0.3, 0.5, 0.7 sequentially
- Generates 6 files: 3 for coarse level, 3 for fine level
- Computes original ratings first (if needed)

**Usage**:
```bash
sbatch run_add_typos_all_probs.slurm
```

**Resources**:
- 1 GPU (any available)
- 100GB RAM
- 48 hours
- 2 cores
- Partition: luxlab

---

### 3. `run_remove_must_have_all_nums.slurm`
**Purpose**: Run remove_must_have with all num_remove values

**What it does**:
- Runs remove_must_have perturbation with removed=1, 2, 3 sequentially
- Generates 3 files (only coarse level has Must_have data)
- Computes original ratings first (if needed)

**Usage**:
```bash
sbatch run_remove_must_have_all_nums.slurm
```

**Resources**:
- 1 GPU (any available)
- 100GB RAM
- 48 hours
- 2 cores
- Partition: luxlab

---

## Checking Job Status

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel a job
scancel <job_id>

# View output logs (while running or after completion)
tail -f m3_eval_qwen3-8b_<job_id>.out
tail -f m3_eval_qwen3-8b_<job_id>.err
```

## Output Files

Results will be saved to:
```
output/
├── original_coarse_Qwen3-8B_rating.jsonl
├── original_fine_Qwen3-8B_rating.jsonl
├── add_typos/
│   ├── add_typos_03prob_coarse_Qwen3-8B_rating.jsonl
│   ├── add_typos_05prob_coarse_Qwen3-8B_rating.jsonl
│   ├── add_typos_07prob_coarse_Qwen3-8B_rating.jsonl
│   └── ... (fine level files)
├── change_dosage/
│   ├── change_dosage_coarse_Qwen3-8B_rating.jsonl
│   └── change_dosage_fine_Qwen3-8B_rating.jsonl
├── remove_must_have/
│   ├── remove_must_have_1removed_coarse_Qwen3-8B_rating.jsonl
│   ├── remove_must_have_2removed_coarse_Qwen3-8B_rating.jsonl
│   └── remove_must_have_3removed_coarse_Qwen3-8B_rating.jsonl
└── add_confusion/
    ├── add_confusion_coarse_Qwen3-8B_rating.jsonl
    └── add_confusion_fine_Qwen3-8B_rating.jsonl
```

## Customizing Scripts

To modify resource requirements, edit the `#SBATCH` directives:

- **Memory**: `#SBATCH --mem=100GB` (can use GB/MB format)
- **Time**: `#SBATCH -t 72:00:00` (hh:mm:ss format)
- **GPUs**: `#SBATCH --gres=gpu:1` (1 GPU, any available type)
- **Cores**: `#SBATCH -n 2` (number of cores)
- **Partition**: `#SBATCH --partition=luxlab`
- **Email**: `#SBATCH --mail-user=fb265@cornell.edu`

## Running Specific Perturbations

To run a single perturbation with custom parameters, create a new SLURM script or run directly:

```bash
# Example: Run only change_dosage
sbatch --wrap="cd ~/m3-eval && source .venv/bin/activate && python code/perturbation_pipeline.py --model Qwen3-8B --perturbation change_dosage"

# Example: Run add_typos with specific probability
sbatch --wrap="cd ~/m3-eval && source .venv/bin/activate && python code/perturbation_pipeline.py --model Qwen3-8B --perturbation add_typos --typo_prob 0.7"
```

## Resume Capability

The pipeline has built-in resume functionality:
- Original ratings are computed once and reused
- Each perturbation file is checked for already-processed entries
- Resubmitting a job will continue from where it left off

This makes it safe to resubmit jobs if they time out or fail midway.

## Email Notifications

All scripts are configured to send email notifications to `fb265@cornell.edu` when:
- Job begins
- Job completes
- Job fails

You can modify the email address by editing the `#SBATCH --mail-user` line in each script.

## Troubleshooting

**Job fails immediately**:
- Check if partition `luxlab` is available: `sinfo`
- Check GPU availability: `sinfo -o "%P %G"`
- Verify you have access to the luxlab partition

**Out of memory**:
- Increase `--mem` to 64000 (64GB)
- Try reducing batch size in inference code

**Job times out**:
- Increase time limit: `-t 48:00:00` (48 hours)
- Or run specific perturbations separately

**GPU not detected**:
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU is allocated: `nvidia-smi` in job output
