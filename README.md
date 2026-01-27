# M3-Eval: Medical Misinformation Model Evaluation

A comprehensive evaluation framework for testing how language models handle medical misinformation through controlled perturbations.

**Research collaboration with Microsoft Research**

---

## 🎯 Overview

M3-Eval evaluates how robust language models are to common types of medical errors:
- **Typos** in medical terms
- **Dosage errors** (wrong amounts, timing, instructions)
- **Missing critical information** (omitted warnings, contraindications)
- **Confusion** (scrambled sentence order)

The framework supports three types of experiments:
1. **Baseline**: Compare ratings of original vs perturbed answers
2. **Error Detection**: Can models detect errors when asked?
3. **Error Priming**: Does knowing about errors change ratings?

---

## 📋 Requirements

- Python 3.11 or 3.12
- CUDA-compatible GPU (optional, for local models)
- 16GB+ RAM (for local model inference)
- API keys (optional, for GPT/Claude/Gemini)

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/m3-eval.git
cd m3-eval

# Run automated setup
bash setup.sh

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

```bash
# Test with local model (Qwen3-8B)
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation change_dosage

# Check results
ls output/change_dosage/
```

---

## 🧪 Experiments

### Baseline: Rating Perturbations

Compare how models rate original vs perturbed answers.

```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --seed 42
```

**Output**: Ratings for both original and perturbed answers
- Shows rating drops across correctness, relevance, safety
- Tests model robustness to errors

### Error Detection: Can Models Spot Errors?

Ask models to detect errors without being told to look for them.

```bash
python code/experiment_runner.py \
  --experiment error_detection \
  --model gpt-4o \
  --perturbation change_dosage
```

**Output**: Detection results (yes/no + explanation + location)
- Tests error awareness
- Compares detection rates across models

### Error Priming: Does Awareness Change Ratings?

Compare ratings with and without error warnings.

```bash
python code/experiment_runner.py \
  --experiment error_priming \
  --model claude-opus-4-5-20251101 \
  --perturbation add_typos
```

**Output**: Both control and primed ratings
- Tests if knowledge affects judgment
- Measures priming effect strength

---

## 📊 Supported Models

### Local Models
- **Qwen3-1.7B** - Fast, lightweight
- **Qwen3-8B** - Balanced performance (default)

### API Models
- **gpt-4o** - OpenAI's latest
- **claude-opus-4-5-20251101** - Anthropic Claude
- **gemini-2.0-flash-exp** - Google Gemini

---

## 🔧 Common Commands

### Run Specific Perturbation

```bash
# Test typos only
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation add_typos \
  --typo-prob 0.7

# Test dosage errors only
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation change_dosage
```

### Run All Perturbations

```bash
# Run all perturbations (default)
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B
```

### Compare Multiple Models

```bash
# Test all models on error detection
for model in "Qwen3-8B" "gpt-4o" "claude-opus-4-5-20251101"
do
  python code/experiment_runner.py \
    --experiment error_detection \
    --model "$model" \
    --perturbation change_dosage \
    --seed 42
done
```

### Adjust Parameters

```bash
# More averaging runs for stability
python code/experiment_runner.py \
  --experiment baseline \
  --model gpt-4o \
  --num-runs 10

# Higher typo probability
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation add_typos \
  --typo-prob 0.9

# Remove more must-have sentences
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation remove_must_have \
  --num-remove 3
```

---

## 📁 Project Structure

```
m3-eval/
├── README.md                       # This file
├── setup.sh                        # Automated setup script
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── data/                          # Dataset files
│   ├── coarse_5pt_expert+llm_consolidated.jsonl
│   └── fine_5pt_expert+llm_consolidated.jsonl
├── code/
│   ├── experiment_runner.py       # Central controller
│   ├── perturbation_pipeline.py   # Original pipeline (backward compat)
│   ├── EXPERIMENTS_README.md      # Detailed experiment docs
│   ├── experiments/               # Experiment modules
│   │   ├── baseline.py
│   │   ├── error_detection.py
│   │   └── error_priming.py
│   ├── helpers/
│   │   ├── experiment_utils.py    # Shared utilities
│   │   ├── multi_llm_inference.py # Model API calls
│   │   └── perturbation_functions.py # Perturbation logic
│   └── prompts/
│       ├── coarseprompt_system.txt
│       └── fineprompt_system.txt
└── output/                        # Experiment results
    ├── original_*.jsonl           # Original ratings
    ├── add_typos/
    ├── change_dosage/
    ├── remove_must_have/
    ├── add_confusion/
    ├── error_detection/
    └── error_priming/
```

---

## 🔬 Perturbation Types

### 1. Add Typos
Introduces typos to medical terms (drugs, conditions).

**Parameters**:
- `--typo-prob`: Probability per term (0.0-1.0)
- `--all-typo-prob`: Test multiple values (0.3, 0.5, 0.7)

**Example**: "acetaminophen" → "acetamnophen"

### 2. Change Dosage
Modifies dosages, timing, and administration instructions.

**Modifications**:
- Dosages: ×10 or ÷10
- Timing: ×2 or ÷2
- Instructions: Flip (swallow ↔ chew)
- Anatomy: Change (both → one eye)

**Example**: "500mg twice daily" → "5000mg every 12 hours"

### 3. Remove Must-Have
Removes critical sentences identified by medical experts.

**Parameters**:
- `--num-remove`: Number of sentences (1-3)
- `--all-num-remove`: Test all values

**Example**: Removes contraindication warnings

### 4. Add Confusion
Randomly shuffles sentence order.

**Example**: "A. B. C." → "C. A. B."

---

## 📖 Documentation

- **Detailed Experiment Guide**: `code/EXPERIMENTS_README.md`
- **Output Formats**: See experiment guide
- **API Reference**: See individual experiment modules

---

## 🔍 Troubleshooting

### GPU Issues

**Problem**: `CUDA error: no kernel image is available for execution`

**Solution**: Your GPU is too old for PyTorch 2.7+. The setup script installs PyTorch with CUDA 11.8 which supports older GPUs (sm_61+).

If still failing:
```bash
# Force CPU usage
CUDA_VISIBLE_DEVICES="" python code/experiment_runner.py ...
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'unsloth'`

**Solution**:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### API Rate Limits

**Problem**: `RateLimitError` when using GPT/Claude/Gemini

**Solution**:
- Reduce `--num-runs` (default: 5)
- Add delays between requests
- Use local models (Qwen3)

### Missing Data Files

**Problem**: `FileNotFoundError: data/coarse_5pt_expert+llm_consolidated.jsonl`

**Solution**: Ensure dataset files are in the `data/` directory

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

---

## 📧 Contact

For questions or issues:
- Open a GitHub issue
- Contact: your.email@domain.com

---

## 🙏 Acknowledgments

- Microsoft Research for collaboration and support
- Scispacy for medical NER models
- Hugging Face for model infrastructure
