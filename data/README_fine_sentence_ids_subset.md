# Fine Sentence IDs Subset

## Background

This file ensures consistency across different model runs for the fine-level CQA evaluation.

## Problem

When running experiments with different models (Qwen3-8B and GPT-4.1), the original ratings were generated on different subsets of sentence_ids:
- **Qwen3-8B**: 98 unique questions, 250 total sentence ratings (1-3 per question)
- **GPT-4.1**: 21 unique questions, 250 total sentence ratings (8-16 per question)

This happened because they used different data files or orderings, causing indices 0-250 to select different sentences.

## Solution

Created `fine_sentence_ids_subset.json` containing the **union of all sentence_ids** from both existing rating files:
- Total unique sentence_ids: **454**
- Includes all work from both Qwen3-8B and GPT-4.1 runs

## Usage

The experiment runner now automatically uses this file when running fine-level experiments:

```bash
# This will now use the 454 sentence_ids from the subset file
python code/experiment_runner.py --experiment baseline --model <model_name> --level fine --generate-only
```

## How It Works

1. When `level='fine'`, the experiment runner checks for `data/fine_sentence_ids_subset.json`
2. If found, it filters the loaded data to only include sentence_ids in this list
3. This ensures all models evaluate the same 454 sentences, making results comparable

## Files Involved

- **Source data**: `data/fine_5pt_expert+llm_consolidated.jsonl` (all sentences)
- **Subset filter**: `data/fine_sentence_ids_subset.json` (454 selected sentence_ids)
- **Original ratings**:
  - `output/cqa_eval/original_ratings/original_fine_Qwen3-8B_rating.jsonl` (250 ratings)
  - `output/cqa_eval/original_ratings/original_fine_gpt-4_1-2025-04-14_rating.jsonl` (250 ratings)
  - Combined unique sentence_ids: 454 (some overlap between files)

## Verifying the Subset

To see which sentence_ids are included:
```bash
jq '.' data/fine_sentence_ids_subset.json | head -20
```

To check how many are in the subset:
```bash
jq '. | length' data/fine_sentence_ids_subset.json
```

## Regenerating (if needed)

To recreate this file from the current rating files:
```bash
jq -r '.sentence_id' output/cqa_eval/original_ratings/original_fine_Qwen3-8B_rating.jsonl \
  output/cqa_eval/original_ratings/original_fine_gpt-4_1-2025-04-14_rating.jsonl | \
  sort -u | jq -R . | jq -s . > data/fine_sentence_ids_subset.json
```
