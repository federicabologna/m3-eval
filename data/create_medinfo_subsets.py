#!/usr/bin/env python3
"""
Create 300-sample subsets for MedInfo2019 datasets.

Creates:
- Coarse subset: 300 randomly sampled answers
- Fine subset: 300 randomly sampled sentences
"""

import json
import os
import random


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def create_subset(input_path, output_path, n_samples=300, seed=42):
    """
    Create a random subset of n_samples from the dataset.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output subset JSONL file
        n_samples: Number of samples to select
        seed: Random seed for reproducibility
    """
    print(f"\nProcessing: {os.path.basename(input_path)}")

    # Load full dataset
    data = load_jsonl(input_path)
    print(f"  Loaded: {len(data)} entries")

    # Check if we need to sample
    if len(data) <= n_samples:
        print(f"  Dataset has {len(data)} entries (≤ {n_samples}), using all entries")
        subset = data
    else:
        # Set random seed for reproducibility
        random.seed(seed)

        # Randomly sample n_samples
        subset = random.sample(data, n_samples)
        print(f"  Sampled: {n_samples} entries (seed={seed})")

    # Save subset
    save_jsonl(subset, output_path)
    print(f"  Saved to: {os.path.basename(output_path)}")

    return len(subset)


def main():
    print("="*80)
    print("MedInfo2019 Subset Creator")
    print("="*80)
    print("Creating 300-sample subsets for both coarse and fine datasets")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input files
    coarse_input = os.path.join(script_dir, 'medinfo2019_medications_qa.jsonl')
    fine_input = os.path.join(script_dir, 'medinfo2019_medications_qa_fine.jsonl')

    # Output files
    coarse_output = os.path.join(script_dir, 'medinfo_coarse_subset300.jsonl')
    fine_output = os.path.join(script_dir, 'medinfo_fine_subset300.jsonl')

    # Check input files exist
    if not os.path.exists(coarse_input):
        print(f"Error: Coarse dataset not found: {coarse_input}")
        return

    if not os.path.exists(fine_input):
        print(f"Error: Fine dataset not found: {fine_input}")
        return

    # Create subsets
    seed = 42

    coarse_count = create_subset(
        coarse_input,
        coarse_output,
        n_samples=300,
        seed=seed
    )

    fine_count = create_subset(
        fine_input,
        fine_output,
        n_samples=300,
        seed=seed
    )

    # Summary
    print(f"\n{'='*80}")
    print("Subsets Created Successfully!")
    print(f"{'='*80}")
    print(f"Coarse subset: {coarse_count} answers")
    print(f"  → {coarse_output}")
    print(f"\nFine subset: {fine_count} sentences")
    print(f"  → {fine_output}")
    print(f"\nRandom seed: {seed}")


if __name__ == "__main__":
    main()
