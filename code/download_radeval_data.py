"""
Download and prepare RadEval Expert Dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset
"""

import json
import os
from datasets import load_dataset


def download_radeval_data(output_dir=None):
    """
    Download RadEval dataset and restructure it.

    Restructures so each row is a single prediction:
    - Each original example has 3 predictions (prediction1/2/3)
    - Each prediction has an annotation (annotation1/2/3)
    - Ground truth is stored as 'reference' field for comparison
    - Final dataset: 208 examples × 3 predictions = 624 rows
    """

    # Default to project's data directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'data')

    print("Downloading RadEval Expert Dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset("IAMJB/RadEvalExpertDataset")

    print(f"Dataset splits: {list(dataset.keys())}")

    # Use the main split (usually 'train' or 'test')
    if 'test' in dataset:
        data = dataset['test']
    elif 'train' in dataset:
        data = dataset['train']
    else:
        data = dataset[list(dataset.keys())[0]]

    print(f"Loaded {len(data)} examples from HuggingFace")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'radeval_expert_dataset.jsonl')

    # Restructure: expand each example into multiple rows (only predictions, not ground truth)
    print("\nRestructuring dataset...")
    restructured_data = []

    for i, example in enumerate(data):
        base_id = f"radeval_{i}"

        # Common fields for all rows from this example
        common_fields = {
            'original_example_id': base_id,
            'annotator': example.get('annotator'),
            'type': example.get('type'),
            'images_path': example.get('images_path'),
            'reference': example.get('ground_truth')  # Reference for evaluation
        }

        # Create rows only for the three predictions with their annotations
        for pred_num in [1, 2, 3]:
            pred_key = f'prediction{pred_num}'
            annot_key = f'annotation{pred_num}'

            if pred_key in example and example[pred_key]:
                prediction_entry = {
                    'id': f"{base_id}_prediction{pred_num}",
                    'prediction': example[pred_key],
                    'annotation': example.get(annot_key),
                    'prediction_source': f'model_{pred_num}',
                    **common_fields
                }
                restructured_data.append(prediction_entry)

    print(f"Restructured into {len(restructured_data)} rows (predictions only)")
    print(f"  - {len(data)} original examples")
    print(f"  - {len(restructured_data)} prediction rows (with annotations)")

    # Save to JSONL
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in restructured_data:
            json.dump(entry, f)
            f.write('\n')

    print(f"✓ Saved {len(restructured_data)} examples to {output_file}")

    # Print dataset info
    print("\nRestructured dataset fields:")
    if len(restructured_data) > 0:
        print(f"  {list(restructured_data[0].keys())}")

    print("\nExample prediction entry (model_1):")
    if len(restructured_data) > 0:
        print(json.dumps(restructured_data[0], indent=2))

    print("\nExample prediction entry (model_2):")
    if len(restructured_data) > 1:
        print(json.dumps(restructured_data[1], indent=2))

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download RadEval dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for dataset (default: project_root/data)')
    args = parser.parse_args()

    download_radeval_data(args.output_dir)
