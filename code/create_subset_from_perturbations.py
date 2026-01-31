"""
Create fine_sentence_ids_subset.json from successful perturbations.

This script reads all fine-level perturbation files and creates a subset
of sentence_ids that have valid perturbations. This ensures we only rate
answers where perturbations were successfully applied.
"""

import json
import os
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
output_dir = project_root / 'output' / 'cqa_eval'
perturbations_dir = output_dir / 'perturbations'
data_dir = project_root / 'data'

# Output file
subset_output_file = data_dir / 'fine_sentence_ids_subset.json'

print("="*80)
print("Creating Subset from Fine-Level Perturbations")
print("="*80)

# Find all fine-level perturbation files
fine_perturbation_files = list(perturbations_dir.glob('*_fine.jsonl'))

if not fine_perturbation_files:
    print(f"No fine-level perturbation files found in {perturbations_dir}")
    print("Please generate perturbations first using --generate-only")
    exit(1)

print(f"\nFound {len(fine_perturbation_files)} fine-level perturbation files:")
for f in sorted(fine_perturbation_files):
    print(f"  - {f.name}")

# Collect all sentence_ids from perturbation files
sentence_ids = set()

for filepath in fine_perturbation_files:
    print(f"\nProcessing: {filepath.name}")
    count = 0

    with open(filepath, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Get sentence_id
                if 'sentence_id' in entry:
                    sentence_ids.add(entry['sentence_id'])
                    count += 1

            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON line: {e}")
                continue

    print(f"  Found {count} sentence IDs")

# Convert to sorted list
sentence_ids_list = sorted(list(sentence_ids))

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total unique sentence IDs: {len(sentence_ids_list)}")
print(f"Output file: {subset_output_file}")

# Save to JSON file
with open(subset_output_file, 'w') as f:
    json.dump(sentence_ids_list, f, indent=2)

print(f"\n✅ Subset file created successfully!")
print(f"   {len(sentence_ids_list)} sentence IDs saved to: {subset_output_file.name}")

# Show some statistics by perturbation type
print(f"\n{'='*80}")
print("Perturbation Coverage:")
print(f"{'='*80}")

for filepath in sorted(fine_perturbation_files):
    with open(filepath, 'r') as f:
        count = sum(1 for line in f if line.strip())
    perturbation_name = filepath.stem.replace('_fine', '')
    print(f"  {perturbation_name}: {count} perturbations")

print(f"\nYou can now use this subset for rating experiments.")
print(f"The subset ensures you only rate answers with valid perturbations.")
