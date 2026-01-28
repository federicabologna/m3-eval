# RadEval Experiment Pipeline

Evaluation pipeline for radiology report generation using the RadEval Expert Dataset and GREEN metric.

## Dataset

**Source**: [IAMJB/RadEvalExpertDataset](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset)

**Evaluation Metric**: GREEN (Generation, Ranking, and Evaluation using Expert Notes)
- Repository: https://github.com/jbdel/RadEval

## Directory Structure

```
output/radeval/
├── original_ratings/          # Original GREEN ratings
├── perturbations/             # Generated perturbations
└── experiment_results/
    └── baseline/              # Baseline experiment results
        ├── add_typos/
        ├── remove_sentences/
        ├── swap_qualifiers/
        └── swap_organs/
```

## Perturbations

### 1. Remove Sentences
Remove 30%, 50%, or 70% of sentences from radiology reports.

### 2. Add Typos
Add typos by swapping adjacent characters with probability 0.3, 0.5, or 0.7.

### 3. Swap Qualifiers
Swap medical qualifiers to their opposites:
- Left ↔ Right
- Mild ↔ Severe
- Low ↔ High
- Small ↔ Large
- Absent ↔ Present
- Minimal ↔ Extensive
- Decreased ↔ Increased
- Normal ↔ Abnormal

### 4. Swap Organs
Swap anatomical locations/organs:
- Lung ↔ Heart
- Liver ↔ Kidney
- Chest ↔ Abdomen
- Brain ↔ Spinal Cord
- Stomach ↔ Intestine

## Setup

### 1. Install Dependencies

```bash
# Install datasets library for HuggingFace
pip install datasets

# Clone RadEval repository for GREEN metric
cd external/
git clone https://github.com/jbdel/RadEval.git
cd RadEval
pip install -r requirements.txt
```

### 2. Download Data

```bash
cd code/
python download_radeval_data.py --output-dir ../data
```

This downloads the RadEval Expert Dataset and saves it to `data/radeval_expert_dataset.jsonl`.

### 3. Configure GREEN Evaluation

**TODO**: Integrate GREEN metric from RadEval repository.

The current implementation has placeholder functions in `helpers/green_eval.py` that need to be connected to the actual GREEN evaluation code.

Steps to integrate:
1. Review RadEval repository structure
2. Import GREEN evaluation functions
3. Update `compute_green_score()` in `green_eval.py`
4. Handle entity extraction and matching

## Usage

### Run All Baseline Experiments

```bash
cd code/
python run_radeval_experiments.py
```

### Run Specific Perturbation

```bash
# Add typos only
python run_radeval_experiments.py --perturbation add_typos

# Swap qualifiers only
python run_radeval_experiments.py --perturbation swap_qualifiers
```

### Custom Fields

If your dataset has different field names:

```bash
python run_radeval_experiments.py \
    --text-field generated_report \
    --reference-field ground_truth
```

### Custom Output Directory

```bash
python run_radeval_experiments.py --output-dir /path/to/output
```

## Pipeline Flow

Similar to CQA eval pipeline:

1. **Load Data**: Load RadEval dataset from JSONL
2. **Original Ratings**: Compute GREEN scores for original predictions
3. **Generate Perturbations**: Create perturbed versions of predictions
4. **Perturbed Ratings**: Compute GREEN scores for perturbed predictions
5. **Analysis**: Compare original vs. perturbed scores

## Files Created

### Code Files

- `code/download_radeval_data.py` - Download dataset from HuggingFace
- `code/run_radeval_experiments.py` - Main experiment runner
- `code/helpers/radeval_perturbations.py` - Radiology-specific perturbations
- `code/helpers/radeval_experiment_utils.py` - Experiment utilities
- `code/helpers/green_eval.py` - GREEN evaluation wrapper (TODO: integrate)

### Output Files

- `data/radeval_expert_dataset.jsonl` - Downloaded dataset
- `output/radeval/original_ratings/original_green_rating.jsonl` - Original GREEN scores
- `output/radeval/perturbations/*.jsonl` - Perturbed predictions
- `output/radeval/experiment_results/baseline/*/*.jsonl` - Experiment results

## Next Steps

1. **Integrate GREEN metric**: Connect to actual GREEN evaluation code from RadEval repo
2. **Validate perturbations**: Review generated perturbations for medical accuracy
3. **Add visualizations**: Create analysis notebooks similar to CQA eval
4. **Run experiments**: Execute baseline experiments once GREEN is integrated
5. **Extend perturbations**: Add domain-specific perturbations as needed

## Notes

- GREEN evaluation is deterministic (unlike LLM-based evaluation in CQA)
- Perturbations are cached and resumable (same as CQA pipeline)
- Random seed ensures reproducibility across runs
- Field names may need adjustment based on actual dataset structure
