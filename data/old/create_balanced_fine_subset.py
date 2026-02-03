"""
Create a balanced subset of fine-level data for add_typos and add_confusion experiments.

Randomly samples 300 sentence_ids ensuring equal representation from
physician, llama, and gpt4 answers.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_dir = project_root / 'data'
fine_data_file = data_dir / 'fine_5pt_expert+llm_consolidated.jsonl'
output_file = data_dir / 'fine_sentence_ids_balanced_subset.json'

print("=" * 80)
print("Creating Balanced Fine-Level Subset")
print("=" * 80)
print(f"Input: {fine_data_file}")
print(f"Output: {output_file}")
print(f"Target size: 300 examples (100 per source)")
print(f"Random seed: {SEED}\n")

# Load all data and group by source
source_ids = defaultdict(list)

with open(fine_data_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        sentence_id = entry['sentence_id']

        # Determine source from sentence_id prefix
        if sentence_id.startswith('physician_'):
            source = 'physician'
        elif sentence_id.startswith('llama_'):
            source = 'llama'
        elif sentence_id.startswith('gpt4_'):
            source = 'gpt4'
        else:
            print(f"Warning: Unknown source for {sentence_id}")
            continue

        source_ids[source].append(sentence_id)

print("Source distribution in full dataset:")
for source, ids in sorted(source_ids.items()):
    print(f"  {source}: {len(ids)} examples")

# Sample 100 from each source
target_per_source = 100
selected_ids = []

for source in ['physician', 'llama', 'gpt4']:
    available = source_ids[source]

    if len(available) < target_per_source:
        print(f"\nWarning: {source} has only {len(available)} examples, less than target {target_per_source}")
        sampled = available
    else:
        sampled = random.sample(available, target_per_source)

    selected_ids.extend(sampled)
    print(f"  Sampled {len(sampled)} from {source}")

# Sort for consistent output
selected_ids.sort()

print(f"\nTotal selected: {len(selected_ids)} sentence IDs")

# Save to JSON
with open(output_file, 'w') as f:
    json.dump(selected_ids, f, indent=2)

print(f"\n✓ Saved to: {output_file}")

# Verify distribution
print("\nVerifying distribution in subset:")
subset_sources = defaultdict(int)
for sid in selected_ids:
    if sid.startswith('physician_'):
        subset_sources['physician'] += 1
    elif sid.startswith('llama_'):
        subset_sources['llama'] += 1
    elif sid.startswith('gpt4_'):
        subset_sources['gpt4'] += 1

for source, count in sorted(subset_sources.items()):
    print(f"  {source}: {count} examples ({count/len(selected_ids)*100:.1f}%)")

print("\n" + "=" * 80)
print("Subset creation complete!")
print("=" * 80)
