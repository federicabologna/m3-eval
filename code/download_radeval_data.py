"""
Download and prepare RadEval Expert Dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset
"""

import json
import os
from datasets import load_dataset


def download_radeval_data(output_dir='../data'):
    """Download RadEval dataset and save to JSONL format."""

    print("Downloading RadEval Expert Dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset("IAMJB/RadEvalExpertDataset")

    print(f"Dataset splits: {list(dataset.keys())}")

    # Use the main split (usually 'train' or 'test')
    # Adjust based on actual dataset structure
    if 'test' in dataset:
        data = dataset['test']
    elif 'train' in dataset:
        data = dataset['train']
    else:
        data = dataset[list(dataset.keys())[0]]

    print(f"Loaded {len(data)} examples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'radeval_expert_dataset.jsonl')

    # Save to JSONL
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for i, example in enumerate(data):
            # Add unique ID if not present
            if 'id' not in example:
                example['id'] = f"radeval_{i}"

            json.dump(example, f)
            f.write('\n')

    print(f"✓ Saved {len(data)} examples to {output_file}")

    # Print dataset info
    print("\nDataset fields:")
    if len(data) > 0:
        print(f"  {list(data[0].keys())}")

    print("\nFirst example:")
    if len(data) > 0:
        print(json.dumps(data[0], indent=2))

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download RadEval dataset')
    parser.add_argument('--output-dir', type=str, default='../data',
                       help='Output directory for dataset')
    args = parser.parse_args()

    download_radeval_data(args.output_dir)
