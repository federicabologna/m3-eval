# M3-Eval: Medical Misinformation Model Evaluation

A comprehensive evaluation framework for testing how language models handle medical misinformation and radiology report generation through controlled perturbations.

**Research collaboration with Microsoft Research**

---

## 🎯 Overview

M3-Eval provides two evaluation pipelines:

### 1. CQA Eval: Clinical Question-Answer Evaluation
Evaluates how robust language models are to common types of medical errors in Q&A responses:
- **Typos** in medical terms
- **Dosage errors** (wrong amounts, timing, instructions)
- **Missing information** (removed sentences)
- **Confusion** (scrambled sentence order)

**Evaluation Levels**:
- **Coarse**: Full answer evaluation (300 Q&A pairs)
- **Fine**: Sentence-level evaluation (250 sentences with 3+ annotations)

### 2. RadEval: Radiology Report Evaluation
Evaluates radiology report generation using the GREEN metric:
- **Typos** in medical terminology
- **Missing sentences** from reports
- **Swapped qualifiers** (mild↔severe, left↔right)
- **Swapped organs** (lung↔heart, liver↔kidney)

**Evaluation Metric**: GREEN (Generation, Ranking, and Evaluation using Expert Notes)

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/m3-eval.git
cd m3-eval

# Run automated setup for CQA eval
bash setup_env.sh

# OR setup RadEval pipeline
bash setup_radeval.sh

# Activate virtual environment
source .venv/bin/activate
```

### 2. Configure API Keys (Optional)

For GPT, Claude, or Gemini models:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run Your First Experiment

**CQA Eval:**
```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation change_dosage
```

**RadEval:**
```bash
python code/run_radeval_experiments.py \
  --perturbation swap_qualifiers
```

---

## 🔧 Common Commands

### CQA Eval

#### Run Specific Perturbation
```bash
# Test typos with high probability
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation add_typos \
  --typo-prob 0.7

# Test sentence removal
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level fine \
  --perturbation remove_sentences \
  --remove-pct 0.5
```

#### Run All Perturbations
```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level both  # Run both coarse and fine
```

#### Run Multiple Parameter Values
```bash
# Test all typo probabilities (0.3, 0.5, 0.7)
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation add_typos \
  --all-typo-prob

# Test all removal percentages (30%, 50%, 70%)
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation remove_sentences \
  --all-remove-pct
```

### RadEval

```bash
# Run all perturbations with default GREEN model
python code/run_radeval_experiments.py

# Run specific perturbation
python code/run_radeval_experiments.py \
  --perturbation swap_organs

# Use GPT-4.1 for evaluation
python code/run_radeval_experiments.py \
  --model gpt-4.1-2025-04-14

# Use GPT-4o for evaluation
python code/run_radeval_experiments.py \
  --model gpt-4o

# Run on CPU (for GREEN model only)
python code/run_radeval_experiments.py \
  --cpu

# Custom field names
python code/run_radeval_experiments.py \
  --text-field generated_report \
  --reference-field ground_truth
```

### Generate Perturbations Only

```bash
# Generate CQA perturbations without ratings
python code/generate_perturbations.py \
  --perturbation add_typos \
  --level coarse \
  --all-typo-prob
```

### SLURM Cluster Jobs

```bash
# Run on HPC cluster
sbatch run_add_typos_all_probs.slurm
sbatch run_remove_sentences_all_pcts.slurm
sbatch run_perturbation_pipeline.slurm
```

---

## 🧪 CQA Eval Experiments

The framework supports three types of experiments:

### 1. Baseline: Rating Perturbations

Compare how models rate original vs perturbed answers.

```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --seed 42
```

**Output**: Ratings for both original and perturbed answers
- `output/cqa_eval/experiment_results/baseline/`
- Shows rating drops across correctness, relevance, safety
- Tests model robustness to errors

### 2. Error Detection: Can Models Spot Errors?

Ask models to detect errors without being told to look for them.

```bash
python code/experiment_runner.py \
  --experiment error_detection \
  --model gpt-4o \
  --level coarse \
  --perturbation change_dosage
```

**Output**: Detection results (yes/no + explanation + location)
- `output/cqa_eval/experiment_results/error_detection/`
- Tests error awareness
- Compares detection rates across models

### 3. Error Priming: Does Awareness Change Ratings?

Compare ratings with and without error warnings.

```bash
python code/experiment_runner.py \
  --experiment error_priming \
  --model claude-opus-4-5-20251101 \
  --level fine \
  --perturbation add_typos
```

**Output**: Both control and primed ratings
- `output/cqa_eval/experiment_results/error_priming/`
- Tests if knowledge affects judgment
- Measures priming effect strength

---

## 🏥 RadEval Experiments

### Baseline: GREEN Score Comparison

Compare GREEN scores for original vs perturbed radiology reports.

```bash
# Run all perturbations
python code/run_radeval_experiments.py

# Run specific perturbation
python code/run_radeval_experiments.py --perturbation swap_organs
```

**Output**: GREEN scores and entity matching results
- `output/radeval/experiment_results/baseline/`
- Precision, Recall, F1 scores
- Entity-level analysis

---

## 📊 Supported Models

### CQA Eval Models

**Local Models:**
- **Qwen3-8B** - Balanced performance (default)
- Uses Unsloth for 4-bit quantization and 2x faster inference

**API Models:**
- **gpt-4.1-2025-04-14** - OpenAI's GPT-4.1
- **gpt-4o** - OpenAI's GPT-4o
- **claude-opus-4-5-20251101** - Anthropic Claude
- **gemini-2.0-flash-exp** - Google Gemini

### RadEval Models

**GREEN Model (default):**
- **StanfordAIMI/GREEN-radllama2-7b** - Fine-tuned LLaMA 2 7B for radiology report evaluation
- Runs locally on GPU or CPU
- LLM-based metric using structured error categorization

**API Models:**
- **gpt-4.1-2025-04-14** - OpenAI's GPT-4.1 via MammoGREEN
- **gpt-4o** - OpenAI's GPT-4o via MammoGREEN
- **gpt-4o-mini** - OpenAI's smaller GPT-4o model

### Evaluation Metrics
- **CQA Eval**: LLM-based rating (correctness, relevance, safety)
- **RadEval**: GREEN metric (LLM-based radiology report evaluation)

---

## 🔬 Perturbation Types

### CQA Eval Perturbations

#### 1. Add Typos
Introduces typos to medical terms by swapping adjacent characters.

**Parameters**:
- `--typo-prob`: Probability per term (0.0-1.0)
- `--all-typo-prob`: Test multiple values (0.3, 0.5, 0.7)

**Example**: "acetaminophen" → "acetamnophen"

#### 2. Change Dosage
Modifies dosages, timing, and administration instructions.

**Modifications**:
- Dosages: ×10 or ÷10
- Timing: ×2 or ÷2
- Instructions: Flip (swallow ↔ chew)
- Anatomy: Change (both → one eye)

**Example**: "500mg twice daily" → "5000mg once daily"

#### 3. Remove Sentences
Randomly removes a percentage of sentences from answers.

**Parameters**:
- `--remove-pct`: Percentage to remove (0.0-1.0)
- `--all-remove-pct`: Test multiple values (0.3, 0.5, 0.7)

**Example**: Removes 30%, 50%, or 70% of sentences

#### 4. Add Confusion
Randomly shuffles sentence order.

**Example**: "A. B. C." → "C. A. B."

### RadEval Perturbations

#### 1. Add Typos
Same as CQA eval - swaps adjacent characters.

#### 2. Remove Sentences
Removes 30%, 50%, or 70% of sentences from radiology reports.

#### 3. Swap Qualifiers
Swaps medical qualifiers to their opposites:
- Left ↔ Right
- Mild ↔ Severe
- Low ↔ High
- Small ↔ Large
- Absent ↔ Present
- Minimal ↔ Extensive
- Decreased ↔ Increased
- Normal ↔ Abnormal

**Example**: "mild left lung opacity" → "severe right lung opacity"

#### 4. Swap Organs
Swaps anatomical locations/organs:
- Lung ↔ Heart
- Liver ↔ Kidney
- Chest ↔ Abdomen
- Brain ↔ Spinal Cord
- Stomach ↔ Intestine

**Example**: "lung consolidation" → "heart consolidation"

---

## 📊 Output Formats

### CQA Eval Results

```json
{
  "question_id": "12",
  "answer_id": "gpt4_12",
  "question": "Can I take ibuprofen with...",
  "answer": "Yes, you can take...",
  "perturbation": "add_typos",
  "perturbed_answer": "Yes, you can tkae...",
  "typo_probability": 0.5,
  "original_rating": {
    "correctness": {"score": 4.2, "confidence": 4.5},
    "relevance": {"score": 4.8, "confidence": 4.7},
    "safety": {"score": 3.9, "confidence": 4.1}
  },
  "perturbed_rating": {
    "correctness": {"score": 3.8, "confidence": 4.2},
    "relevance": {"score": 4.6, "confidence": 4.5},
    "safety": {"score": 3.7, "confidence": 3.9}
  },
  "random_seed": 42
}
```

### RadEval Results

```json
{
  "id": "radeval_42",
  "prediction": "No acute cardiopulmonary...",
  "reference": "No acute findings...",
  "perturbation": "swap_qualifiers",
  "perturbed_prediction": "Acute abnormal cardiopulmonary...",
  "qualifier_changes": {
    "changes": ["no -> multiple", "normal -> abnormal"],
    "num_changes": 2
  },
  "original_rating": {
    "green_score": 0.85,
    "precision": 0.87,
    "recall": 0.83,
    "f1": 0.85
  },
  "perturbed_rating": {
    "green_score": 0.42,
    "precision": 0.45,
    "recall": 0.39,
    "f1": 0.42
  },
  "random_seed": 42
}
```

---

## 🔍 Key Features

### Resumable Processing
- All experiments cache results and can resume from interruption
- Perturbations are generated once and reused across experiments
- Original ratings are computed once and reused

### Reproducible
- Random seed control ensures identical results
- Deterministic perturbation generation
- Version-controlled prompts

### Scalable
- Parallel processing support
- SLURM cluster integration
- Batch processing for large datasets

### Flexible
- Support for multiple models (local + API)
- Configurable perturbation parameters
- Extensible experiment framework

---

## 📁 Project & Code Structure

### Full Directory Tree

```
m3-eval/
├── README.md                      # This file
├── RADEVAL_README.md              # RadEval detailed guide
├── SETUP_GUIDE.md                 # Setup instructions
├── setup_env.sh                   # CQA eval setup
├── setup_radeval.sh               # RadEval setup
├── requirements_clean.txt         # Python dependencies
├── .env.example                   # API key template
│
├── data/                          # Dataset files
│   ├── coarse_5pt_expert+llm_consolidated.jsonl     # Answer-level (300)
│   ├── fine_5pt_expert+llm_consolidated.jsonl       # Sentence-level (250)
│   ├── radeval_expert_dataset.jsonl                 # RadEval dataset
│   └── old/                       # Original unconsolidated data
│       ├── consolidate_fine_data.py
│       ├── consolidate_coarse_data.py
│       └── *.json                # Original annotations
│
├── code/                          # Source code
│   ├── experiment_runner.py      # CQA eval central controller
│   ├── run_radeval_experiments.py    # RadEval experiment runner
│   ├── download_radeval_data.py  # Download RadEval dataset
│   ├── perturbation_pipeline.py  # Legacy CQA pipeline (backward compat)
│   ├── generate_perturbations.py # Standalone perturbation generator
│   │
│   ├── experiments/              # Experiment modules
│   │   ├── baseline.py           # Baseline rating experiments
│   │   ├── error_detection.py    # Error detection experiments
│   │   └── error_priming.py      # Error priming experiments
│   │
│   ├── helpers/                  # Helper modules
│   │   ├── experiment_utils.py           # CQA eval utilities
│   │   ├── radeval_experiment_utils.py   # RadEval utilities
│   │   ├── perturbation_functions.py     # CQA perturbations
│   │   ├── radeval_perturbations.py      # RadEval perturbations
│   │   ├── multi_llm_inference.py        # LLM API wrapper
│   │   └── green_eval.py                 # GREEN metric wrapper
│   │
│   └── prompts/                  # Evaluation prompts
│       ├── coarseprompt_system.txt       # Answer-level evaluation
│       ├── fineprompt_system.txt         # Sentence-level evaluation
│       ├── error_priming_coarse.txt      # Priming prompt (coarse)
│       └── error_priming_fine.txt        # Priming prompt (fine)
│
├── notebooks/                     # Jupyter notebooks
│   └── baseline_experiments.ipynb
│
├── output/
│   ├── cqa_eval/                  # CQA eval results
│   │   ├── original_ratings/
│   │   ├── perturbations/
│   │   └── experiment_results/
│   │       └── baseline/
│   │           ├── add_typos/
│   │           ├── change_dosage/
│   │           ├── remove_sentences/
│   │           └── add_confusion/
│   │
│   └── radeval/                   # RadEval results
│       ├── original_ratings/
│       ├── perturbations/
│       └── experiment_results/
│           └── baseline/
│               ├── add_typos/
│               ├── remove_sentences/
│               ├── swap_qualifiers/
│               └── swap_organs/
│
└── external/                      # External repositories
    └── RadEval/                   # GREEN evaluation code
```

### Key Code Files

**Main Scripts:**
- `experiment_runner.py` - CQA eval central controller
- `run_radeval_experiments.py` - RadEval experiment runner
- `download_radeval_data.py` - Download RadEval dataset
- `perturbation_pipeline.py` - Legacy CQA pipeline (backward compatibility)
- `generate_perturbations.py` - Standalone perturbation generator

**Experiment Modules:**
- `baseline.py` - Baseline rating experiments
- `error_detection.py` - Error detection experiments
- `error_priming.py` - Error priming experiments

**Helper Modules:**
- `experiment_utils.py` - CQA eval utilities
- `radeval_experiment_utils.py` - RadEval utilities
- `perturbation_functions.py` - CQA perturbations
- `radeval_perturbations.py` - RadEval perturbations
- `multi_llm_inference.py` - LLM API wrapper
- `green_eval.py` - GREEN metric wrapper

**Prompts:**
- `coarseprompt_system.txt` - Answer-level evaluation
- `fineprompt_system.txt` - Sentence-level evaluation
- `error_priming_coarse.txt` - Priming prompt (coarse)
- `error_priming_fine.txt` - Priming prompt (fine)

**Data Consolidation Scripts:**
- `consolidate_fine_data.py` - Consolidate sentence annotations (3+ annotations)
- `consolidate_coarse_data.py` - Consolidate answer annotations

---

## 🔍 Troubleshooting

### GPU/CUDA Issues

**Problem**: `CUDA error` or `NCCL` errors

**Solution**: Use matching CUDA versions
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Import Errors

**Problem**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
source .venv/bin/activate
uv pip install -r requirements_clean.txt
python -m spacy download en_core_web_sm
```

### Model Loading Issues

**Problem**: `max_seq_length` warnings or truncation

**Solution**: Already fixed in code - `max_seq_length=8192` for long prompts

### API Rate Limits

**Problem**: `RateLimitError` when using GPT/Claude/Gemini

**Solution**:
- Use local models (Qwen3-8B)
- Reduce `--num-runs` (default: 5)
- Add delays between requests

### Data Filtering

**Problem**: Mismatched sentence counts between datasets

**Solution**: Use consolidation scripts in `data/old/` to filter for sentences with 3+ annotations

---

## 📖 Documentation

- **RadEval Guide**: `RADEVAL_README.md`
- **Setup Guide**: `SETUP_GUIDE.md`
- **Detailed Scripts**: See docstrings in each Python file

---

## 🤝 Contributing

This is a research project in collaboration with Microsoft Research. For questions or contributions, please contact the project maintainers.

---

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{m3eval2026,
  title={M3-Eval: Medical Misinformation Model Evaluation},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/your-org/m3-eval}
}
```
