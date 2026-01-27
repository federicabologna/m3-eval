# M3-Eval Experiment System

A modular framework for running perturbation experiments on medical QA evaluation.

## Quick Start

```bash
# Run baseline experiment (original pipeline)
python code/experiment_runner.py --experiment baseline --model Qwen3-8B

# Test error detection
python code/experiment_runner.py --experiment error_detection --model gpt-4o

# Test error priming effect
python code/experiment_runner.py --experiment error_priming --model claude-opus-4-5-20251101
```

## Architecture

```
code/
├── experiment_runner.py          # Central controller
├── experiments/                  # Experiment modules
│   ├── baseline.py              # Original perturbation + rating
│   ├── error_detection.py       # Detect errors without being told
│   └── error_priming.py         # Compare ratings with/without warnings
└── helpers/
    ├── experiment_utils.py      # Shared utilities
    ├── multi_llm_inference.py   # LLM API calls
    └── perturbation_functions.py # Perturbation logic
```

## Experiments

### 1. Baseline (Original Pipeline)

Applies perturbations and collects ratings.

**Output**: Ratings for original vs perturbed answers

**Example**:
```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation change_dosage \
  --seed 42
```

### 2. Error Detection

Tests if models can detect errors without being told.

**For dosage/typos**: "Is there an error in this answer?"
**For must_have**: "Is important information missing from this answer?"

**Output**: Detection result (yes/no + explanation + localization)

**Example**:
```bash
python code/experiment_runner.py \
  --experiment error_detection \
  --model gpt-4o \
  --perturbation add_typos \
  --typo-prob 0.7
```

### 3. Error Priming

Compares ratings with/without error warnings.

**Control**: Rate normally
**Primed**: Rate with "Note: This answer contains an error" prepended

**Output**: Both control and primed ratings for comparison

**Example**:
```bash
python code/experiment_runner.py \
  --experiment error_priming \
  --model claude-opus-4-5-20251101 \
  --perturbation change_dosage
```

## Common Arguments

### Model Selection
```bash
--model MODEL_NAME              # Model to use for evaluation
```

**Examples**:
- `Qwen3-8B` (local model, default)
- `gpt-4o` (OpenAI)
- `claude-opus-4-5-20251101` (Anthropic)
- `gemini-2.0-flash-exp` (Google)

### Perturbation Selection
```bash
--perturbation PERT_NAME        # Specific perturbation (default: all)
```

**Options**:
- `add_typos` - Add typos to medical terms
- `change_dosage` - Modify dosages/instructions
- `remove_must_have` - Remove critical sentences
- `add_confusion` - Shuffle sentence order

### Perturbation Parameters

**For add_typos**:
```bash
--typo-prob 0.5                 # Probability per term (0.0-1.0)
--all-typo-prob                 # Run all: 0.3, 0.5, 0.7
```

**For remove_must_have**:
```bash
--num-remove 1                  # Number of sentences to remove (1-3)
--all-num-remove                # Run all: 1, 2, 3
```

### Level Selection
```bash
--level LEVEL                   # coarse, fine, or both (default: both)
```

### Rating Parameters
```bash
--num-runs 5                    # Number of rating runs to average
--max-retries 3                 # Max retries for invalid responses
```

### Reproducibility
```bash
--seed 42                       # Random seed (default: 42)
```

## Output Structure

### Baseline Experiment
```
output/
├── original_coarse_MODEL_rating.jsonl        # Original ratings
├── add_typos/
│   └── add_typos_05prob_coarse_MODEL_rating.jsonl
├── change_dosage/
│   └── change_dosage_coarse_MODEL_rating.jsonl
└── remove_must_have/
    └── remove_must_have_1removed_coarse_MODEL_rating.jsonl
```

### Error Detection Experiment
```
output/
└── error_detection/
    ├── detection_add_typos_05prob_coarse_MODEL.jsonl
    ├── detection_change_dosage_coarse_MODEL.jsonl
    └── detection_remove_must_have_1removed_coarse_MODEL.jsonl
```

### Error Priming Experiment
```
output/
└── error_priming/
    ├── priming_add_typos_05prob_coarse_MODEL.jsonl
    ├── priming_change_dosage_coarse_MODEL.jsonl
    └── priming_remove_must_have_1removed_coarse_MODEL.jsonl
```

## Output Formats

### Baseline Output
```json
{
  "answer_id": "123",
  "question": "...",
  "answer": "...",
  "perturbation": "change_dosage",
  "perturbed_answer": "...",
  "original_rating": {
    "correctness": {"score": 4.2, "confidence": 0.9},
    "relevance": {"score": 4.5, "confidence": 0.95},
    "safety": {"score": 3.8, "confidence": 0.85}
  },
  "perturbed_rating": {
    "correctness": {"score": 2.1, "confidence": 0.7},
    "relevance": {"score": 4.3, "confidence": 0.9},
    "safety": {"score": 1.2, "confidence": 0.8}
  },
  "random_seed": 42
}
```

### Error Detection Output
```json
{
  "answer_id": "123",
  "question": "...",
  "answer": "...",
  "perturbation": "change_dosage",
  "perturbed_answer": "...",
  "detection_result": {
    "detected": "yes",
    "explanation": "The dosage of 5000mg is dangerously high",
    "location": "Take 5000mg twice daily"
  },
  "random_seed": 42
}
```

### Error Priming Output
```json
{
  "answer_id": "123",
  "question": "...",
  "answer": "...",
  "perturbation": "change_dosage",
  "perturbed_answer": "...",
  "control_rating": {
    "correctness": {"score": 3.5, "confidence": 0.8},
    "relevance": {"score": 4.0, "confidence": 0.85},
    "safety": {"score": 3.2, "confidence": 0.75}
  },
  "primed_rating": {
    "correctness": {"score": 1.8, "confidence": 0.9},
    "relevance": {"score": 3.9, "confidence": 0.85},
    "safety": {"score": 1.0, "confidence": 0.95}
  },
  "priming_text": "Note: This answer contains an error.",
  "random_seed": 42
}
```

## Comparing Experiments

### Research Questions

1. **Can models detect errors?** (Error Detection)
   - Detection rate by perturbation type
   - Localization accuracy
   - Model comparison

2. **Does knowing about errors change ratings?** (Error Priming)
   - Rating differences: control vs primed
   - Which dimensions are most affected?
   - Priming effect by model

3. **How do ratings change with perturbations?** (Baseline)
   - Rating drops by perturbation type
   - Robustness by model
   - Dimension-specific sensitivity

## Running Multiple Models

```bash
# Test all models on error detection
for model in "Qwen3-8B" "gpt-4o" "claude-opus-4-5-20251101" "gemini-2.0-flash-exp"
do
  python code/experiment_runner.py \
    --experiment error_detection \
    --model "$model" \
    --seed 42
done
```

## Backward Compatibility

The original `perturbation_pipeline.py` still works:

```bash
python code/perturbation_pipeline.py --model Qwen3-8B --perturbation change_dosage
```

This is equivalent to:

```bash
python code/experiment_runner.py --experiment baseline --model Qwen3-8B --perturbation change_dosage
```

## Tips

1. **Start small**: Test with one perturbation and one level first
2. **Use consistent seeds**: Same seed = same perturbations across models
3. **Monitor costs**: API calls can add up with multiple runs
4. **Check outputs**: Verify JSON files after each experiment
5. **Iterate quickly**: Use `--num-runs 1` for testing, increase for production
