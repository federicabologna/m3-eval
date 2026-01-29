# RadEval Analysis Scripts

Analysis scripts for evaluating RadEval experiment results.

## Scripts

### 1. `analyze_chexbert_results.py`
Analyzes CheXbert evaluation results comparing original vs perturbed reports.

**Metrics analyzed:**
- All metrics computed and included in summary report (all 14 conditions):
  - Micro F1
  - Macro F1
  - Weighted F1

**Output:**
- Bar plot: Weighted F1 (all 14 conditions)
  - Y-axis: Mean Weighted F1 across all instances
  - Error bars: 95% confidence intervals
  - One bar for all original scores combined
  - Separate bars for each perturbation (ordered: swap organs, swap qualifiers, remove sentences 30%/50%/70%, add typos p=0.3/0.5/0.7)
- Severity effect plots:
  - Add Typos: Score degradation vs typo probability (0.3, 0.5, 0.7)
  - Remove Sentences: Score degradation vs percentage removed (30%, 50%, 70%)
  - Shows how increasing perturbation severity affects score degradation
- Summary report with Wilcoxon signed-rank test results for ALL metrics

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
- Severity effect plots:
  - Add Typos: Score degradation vs typo probability (0.3, 0.5, 0.7)
  - Remove Sentences: Score degradation vs percentage removed (30%, 50%, 70%)
  - Shows how increasing perturbation severity affects score degradation
- Summary report with Wilcoxon signed-rank test results

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
- `chexbert_all_weighted_f1_typo_severity.png` (degradation vs typo probability)
- `chexbert_all_weighted_f1_remove_severity.png` (degradation vs sentences removed)
- `chexbert_summary_report.txt` (includes ALL 7 metrics)

**GREEN outputs:**
- `green_barplot_gpt-4_1-2025-04-14.png` (mean ± 95% CI)
- `green_typo_severity_gpt-4_1-2025-04-14.png` (degradation vs typo probability)
- `green_remove_severity_gpt-4_1-2025-04-14.png` (degradation vs sentences removed)
- `green_summary_report_gpt-4_1-2025-04-14.txt`

---

## Statistical Tests

Both scripts use the **Wilcoxon signed-rank test** to compare original vs perturbed scores:
- Tests if perturbations significantly decrease scores (paired samples test)
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
Number of samples: 606

Micro F1 (All 14): W=123678.00, p=3.4567e-08 ***
Macro F1 (All 14): W=123890.00, p=5.6789e-06 ***
Weighted F1 (All 14): W=124012.00, p=7.8901e-04 ***
```

---

## Dependencies

Both scripts require:
```bash
pip install numpy matplotlib scipy
```

Already included in the project's environment.
