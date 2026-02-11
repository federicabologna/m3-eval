#!/usr/bin/env python3
"""
Test LLM-based perturbations on RadEval data with new JSON format.
"""

import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Add helpers to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.radeval_perturbations import (
    inject_false_prediction,
    inject_contradiction,
    inject_false_negation
)


def load_radeval_sample(data_path, num_samples=3):
    """Load a small sample of RadEval data."""
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def test_perturbation(sample, perturbation_func, error_name, model='gpt-4.1'):
    """Test a single perturbation on a sample."""
    print(f"\n{'='*80}")
    print(f"TEST: {error_name}")
    print(f"Sample ID: {sample['id']}")
    print(f"Model: {model}")
    print(f"{'='*80}")

    # Get the text field (usually 'prediction' or 'generated_report')
    original_text = sample.get('prediction', sample.get('generated_report', ''))

    if not original_text:
        print("❌ No text field found in sample")
        return None

    print(f"\n📄 ORIGINAL REPORT ({len(original_text)} chars):")
    print("-" * 80)
    print(original_text[:500] + "..." if len(original_text) > 500 else original_text)

    print(f"\n🔄 Generating perturbation...")
    try:
        perturbed_text, metadata = perturbation_func(original_text, model=model)

        print(f"\n📄 PERTURBED REPORT ({len(perturbed_text)} chars):")
        print("-" * 80)
        print(perturbed_text[:500] + "..." if len(perturbed_text) > 500 else perturbed_text)

        print(f"\n📊 METADATA:")
        print("-" * 80)
        print(f"  Error type: {metadata.get('error_type')}")
        print(f"  Error name: {metadata.get('error_name')}")
        print(f"  Model: {metadata.get('model')}")
        print(f"  Number of changes: {metadata.get('num_changes', 0)}")

        if 'skip_reason' in metadata:
            print(f"  ⚠️  Skip reason: {metadata['skip_reason']}")
        elif 'error' in metadata:
            print(f"  ❌ Error: {metadata['error']}")
        else:
            print(f"  ✅ Success!")

            # Show changes
            if metadata.get('changes'):
                print(f"\n  Changes made:")
                for i, change in enumerate(metadata['changes'][:3], 1):
                    print(f"    {i}. {change}")
                if len(metadata['changes']) > 3:
                    print(f"    ... and {len(metadata['changes']) - 3} more")

        return {
            'sample_id': sample['id'],
            'error_name': error_name,
            'original_length': len(original_text),
            'perturbed_length': len(perturbed_text),
            'num_changes': metadata.get('num_changes', 0),
            'success': 'error' not in metadata and 'skip_reason' not in metadata
        }

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run tests on RadEval samples."""
    import argparse

    parser = argparse.ArgumentParser(description='Test LLM perturbations on RadEval data')
    parser.add_argument('--data', type=str,
                       default='/Users/Federica_1/Documents/GitHub/m3-eval/data/radeval_expert_dataset.jsonl',
                       help='Path to RadEval data')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to test (default: 3)')
    parser.add_argument('--model', type=str, default='gpt-4.1',
                       help='Model to use (default: gpt-4.1)')
    parser.add_argument('--error-type', type=int, choices=[5, 10, 11],
                       help='Test only specific error type (5, 10, or 11)')

    args = parser.parse_args()

    print("="*80)
    print("LLM PERTURBATION TEST - RADEVAL DATA")
    print("="*80)
    print(f"Data: {args.data}")
    print(f"Samples: {args.num_samples}")
    print(f"Model: {args.model}")
    print("="*80)

    # Load samples
    print(f"\n📥 Loading {args.num_samples} samples...")
    samples = load_radeval_sample(args.data, args.num_samples)
    print(f"✓ Loaded {len(samples)} samples")

    # Define perturbations to test
    perturbations = []
    if args.error_type is None or args.error_type == 5:
        perturbations.append((inject_false_prediction, "Error Type 5: False Prediction"))
    if args.error_type is None or args.error_type == 10:
        perturbations.append((inject_contradiction, "Error Type 10: Add Contradiction"))
    if args.error_type is None or args.error_type == 11:
        perturbations.append((inject_false_negation, "Error Type 11: False Negation"))

    # Run tests
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n\n{'#'*80}")
        print(f"SAMPLE {i}/{len(samples)}")
        print(f"{'#'*80}")

        for pert_func, pert_name in perturbations:
            result = test_perturbation(sample, pert_func, pert_name, args.model)
            if result:
                results.append(result)

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests run: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")

    if results:
        print(f"\nAverage changes per successful test: {sum(r['num_changes'] for r in results if r['success']) / max(sum(1 for r in results if r['success']), 1):.1f}")

    print("="*80)


if __name__ == '__main__':
    main()
