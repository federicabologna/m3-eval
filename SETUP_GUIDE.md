# M3-Eval Setup Guide

Complete guide for setting up and running M3-Eval experiments.

---

## 📦 For New Users: Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/m3-eval.git
cd m3-eval

# 2. Run automated setup
bash setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Verify installation
python test_setup.py

# 5. Run a test experiment
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation change_dosage
```

That's it! You're ready to go.

---

### Option 2: Manual Setup

If the automated setup fails, follow these steps:

#### Step 1: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate     # On Windows
```

#### Step 2: Install PyTorch

**For CUDA 11.8 (supports older GPUs):**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Download Medical NER Model

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

#### Step 5: Configure API Keys (Optional)

```bash
cp .env.example .env
# Edit .env with your API keys
```

#### Step 6: Verify Installation

```bash
python test_setup.py
```

---

## 🔑 API Configuration

### Setting Up API Keys

API keys are only needed for GPT, Claude, or Gemini models. Local models (Qwen3) don't require keys.

#### 1. Copy Template

```bash
cp .env.example .env
```

#### 2. Get API Keys

**OpenAI (GPT models)**:
- Go to https://platform.openai.com/api-keys
- Create a new API key
- Copy to `.env`: `OPENAI_API_KEY=sk-...`

**Anthropic (Claude models)**:
- Go to https://console.anthropic.com/
- Create a new API key
- Copy to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

**Google (Gemini models)**:
- Go to https://aistudio.google.com/app/apikey
- Create a new API key
- Copy to `.env`: `GOOGLE_API_KEY=...`

#### 3. Verify

```bash
# Test OpenAI
python -c "from openai import OpenAI; import os; from dotenv import load_dotenv; load_dotenv(); client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('✓ OpenAI key valid')"

# Test Anthropic
python -c "from anthropic import Anthropic; import os; from dotenv import load_dotenv; load_dotenv(); client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')); print('✓ Anthropic key valid')"

# Test Google
python -c "from google import genai; import os; from dotenv import load_dotenv; load_dotenv(); client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY')); print('✓ Google key valid')"
```

---

## 🧪 Running Your First Experiment

### 1. Test with Local Model (No API key needed)

```bash
# Small test: One perturbation, coarse level only
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --level coarse \
  --perturbation change_dosage \
  --num-runs 1

# Check output
ls output/change_dosage/
```

Expected output:
```
output/
├── original_coarse_Qwen3-8B_rating.jsonl
└── change_dosage/
    └── change_dosage_coarse_Qwen3-8B_rating.jsonl
```

### 2. Run All Perturbations

```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --seed 42
```

This runs all 4 perturbations on both coarse and fine levels.

### 3. Test Error Detection

```bash
python code/experiment_runner.py \
  --experiment error_detection \
  --model Qwen3-8B \
  --perturbation change_dosage
```

### 4. Test Error Priming

```bash
python code/experiment_runner.py \
  --experiment error_priming \
  --model Qwen3-8B \
  --perturbation change_dosage
```

---

## 🐛 Common Issues

### Issue 1: Import Error for Unsloth

**Error**: `ModuleNotFoundError: No module named 'unsloth'`

**Solution**:
```bash
source .venv/bin/activate
pip install unsloth
```

### Issue 2: CUDA Not Available

**Error**: `RuntimeError: CUDA error: no kernel image is available`

**This means**: Your GPU is too old for the installed PyTorch version.

**Solutions**:

1. **Use PyTorch with CUDA 11.8** (setup.sh does this automatically):
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

2. **Force CPU mode**:
```bash
CUDA_VISIBLE_DEVICES="" python code/experiment_runner.py ...
```

### Issue 3: Missing Data Files

**Error**: `FileNotFoundError: data/coarse_5pt_expert+llm_consolidated.jsonl`

**Solution**: Ensure your data files are in the `data/` directory:
```
data/
├── coarse_5pt_expert+llm_consolidated.jsonl
└── fine_5pt_expert+llm_consolidated.jsonl
```

### Issue 4: Rate Limit Errors (API Models)

**Error**: `RateLimitError: You exceeded your current quota`

**Solutions**:
- Reduce `--num-runs` (default is 5)
- Use `--level coarse` only (skip fine)
- Use `--perturbation` to test one at a time
- Switch to local model (Qwen3)

### Issue 5: Slow Performance

**Problem**: Experiments take too long

**Solutions**:
- Use `--num-runs 1` for testing
- Use `--level coarse` only
- Use smaller model (Qwen3-1.7B)
- Use GPU instead of CPU

---

## 📊 Understanding Output Files

### Original Ratings
```
output/original_coarse_Qwen3-8B_rating.jsonl
```
Contains ratings for unmodified answers (computed once, reused across experiments).

### Perturbation Results
```
output/change_dosage/change_dosage_coarse_Qwen3-8B_rating.jsonl
```
Contains both original and perturbed ratings for comparison.

### Error Detection Results
```
output/error_detection/detection_change_dosage_coarse_Qwen3-8B.jsonl
```
Contains detection results (yes/no + explanation + location).

### Error Priming Results
```
output/error_priming/priming_change_dosage_coarse_Qwen3-8B.jsonl
```
Contains both control and primed ratings for comparison.

---

## 🔄 Updating

### Update Code

```bash
git pull origin main
```

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### Verify After Update

```bash
python test_setup.py
```

---

## 💡 Best Practices

1. **Always use --seed** for reproducibility:
   ```bash
   --seed 42
   ```

2. **Start with small tests**:
   ```bash
   --level coarse --perturbation change_dosage --num-runs 1
   ```

3. **Use consistent seeds across models**:
   ```bash
   # Same seed = same perturbations for all models
   for model in "Qwen3-8B" "gpt-4o" "claude-opus-4-5-20251101"
   do
     python code/experiment_runner.py \
       --experiment baseline \
       --model "$model" \
       --seed 42
   done
   ```

4. **Check outputs regularly**:
   ```bash
   # View latest results
   tail -1 output/change_dosage/*.jsonl | python -m json.tool
   ```

5. **Monitor costs for API models**:
   - Use `--num-runs 1` initially
   - Test on `--level coarse` first
   - Track API usage in provider dashboards

---

## 📞 Getting Help

1. **Verify installation**: Run `python test_setup.py`
2. **Check documentation**: See `README.md` and `code/EXPERIMENTS_README.md`
3. **Check troubleshooting**: See "Common Issues" section above
4. **Report issues**: Open a GitHub issue with:
   - Output of `python test_setup.py`
   - Full error message
   - Command you ran

---

## ✅ Checklist

Before running experiments, make sure:

- [ ] Python 3.11 or 3.12 installed
- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] Data files present (`ls data/`)
- [ ] Setup verified (`python test_setup.py`)
- [ ] API keys configured (if using API models)
- [ ] Output directory created (`mkdir -p output`)

---

## 🚀 Ready to Go!

You're all set! Run your first experiment:

```bash
python code/experiment_runner.py \
  --experiment baseline \
  --model Qwen3-8B \
  --perturbation change_dosage \
  --seed 42
```

Good luck with your research! 🎯
