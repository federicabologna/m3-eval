# RadEval Analysis Scripts

Analysis scripts for evaluating RadEval experiment results.

## Scripts

### 1. `analyze_chexbert_results.py`
Analyzes CheXbert evaluation results comparing original vs perturbed reports.

**Metrics analyzed:**
- All metrics computed and included in summary report:
  - Accuracy (TOP5 conditions)
  - Micro F1 (TOP5 and all 14 conditions)
  - Macro F1 (TOP5 and all 14 conditions)
  - Weighted F1 (TOP5 and all 14 conditions)

**Output:**
- Bar plot: Weighted F1 (all 14 conditions)
  - Y-axis: Mean Weighted F1 across all instances
  - Error bars: 95% confidence intervals
  - One bar for all original scores combined
  - Separate bars for each perturbation (ordered: swap organs, swap qualifiers, remove sentences 30%/50%/70%, add typos p=0.3/0.5/0.7)
- Summary report with statistics for ALL metrics and significance tests

**Usage:**
```bash
python code/analysis/analyze_chexbert_results.py
```

**Requirements:**
- CheXbert result files in: `output/radeval/experiment_results/baseline/*/`
- Files must end with `_chexbert_rating.jsonl`

---

### 2. `analyze_green_results.py`
Analyzes GREEN evaluation results comparing original vs perturbed reports.

**Metrics analyzed:**
- GREEN score (0-1 scale)

**Output:**
- Bar plot: GREEN scores
  - Y-axis: Mean GREEN score across all instances
  - Error bars: 95% confidence intervals
  - One bar for all original scores combined
  - Separate bars for each perturbation (ordered: swap organs, swap qualifiers, remove sentences 30%/50%/70%, add typos p=0.3/0.5/0.7)
- Summary report with statistics and significance tests

**Usage:**
```bash
python code/analysis/analyze_green_results.py
```

**Requirements:**
- GREEN result files in: `output/radeval/experiment_results/baseline/*/`
- Files must end with `_green_rating.jsonl`

---

## Output Directory

All analysis outputs are saved to:
```
output/radeval/analysis/
```

**CheXbert outputs:**
- `chexbert_all_weighted_f1_barplot.png` (all 14 conditions, mean ± 95% CI)
- `chexbert_summary_report.txt` (includes ALL 7 metrics)

**GREEN outputs:**
- `green_barplot_{model_name}.png` (mean ± 95% CI)
- `green_summary_report_{model_name}.txt`

---

## Statistical Tests

Both scripts use the **Wilcoxon signed-rank test** to compare original vs perturbed scores:
- Tests if perturbations significantly decrease scores
- Reports p-values with significance indicators:
  - `***` p < 0.001 (highly significant)
  - `**` p < 0.01 (very significant)
  - `*` p < 0.05 (significant)
  - `ns` p >= 0.05 (not significant)

---

## Example Output

### Summary Report Format
```
================================================================================
Perturbation: Swap Qualifiers
================================================================================
File: swap_qualifiers_chexbert_rating.jsonl
Number of samples: 624

Accuracy (TOP5)
--------------------------------------------------------------------------------
  Original:  0.7452 ± 0.3214
  Perturbed: 0.6891 ± 0.3456
  Mean degradation: 0.0561 ± 0.1234

  Score decreased: 45.2%
  Score increased: 12.3%
  Score unchanged: 42.5%

  Wilcoxon signed-rank test:
    Statistic: 12345.67
    p-value: 1.2345e-10 ***
```

---

## Dependencies

Both scripts require:
```bash
pip install numpy matplotlib scipy
```

Already included in the project's environment.
