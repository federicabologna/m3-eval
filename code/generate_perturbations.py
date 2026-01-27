#!/usr/bin/env python3
"""
Generate and save perturbations for all QA pairs.

This script creates perturbed versions of answers WITHOUT running experiments.
The perturbed files can then be used as input for multiple experiments
(baseline, error_detection, error_priming).

Note: This script automatically resumes from partial files. If perturbations
already exist, it will only generate missing ones.

Usage:
    # Generate all perturbations
    python code/generate_perturbations.py --seed 42

    # Generate specific perturbation
    python code/generate_perturbations.py --perturbation change_dosage --seed 42

    # Generate with specific parameters
    python code/generate_perturbations.py \
      --perturbation add_typos \
      --typo-prob 0.7 \
      --seed 42

Alternative: Use experiment_runner.py --generate-only to generate both
original ratings AND perturbations.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.experiment_utils import (
    setup_paths,
    load_qa_data,
    get_id_key,
    get_or_create_perturbations
)


def generate_perturbations(args):
    """Generate and save perturbations for all QA pairs."""

    # Set random seed for reproducibility
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"{'='*80}")
    print("GENERATING PERTURBATIONS")
    print(f"{'='*80}\n")

    # Setup paths
    paths = setup_paths(args.output_dir)
    output_dir = paths['output_dir']

    # Create perturbations subdirectory
    perturbations_dir = os.path.join(output_dir, 'perturbations')
    os.makedirs(perturbations_dir, exist_ok=True)

    # Define perturbation types
    all_perturbations = ['add_typos', 'change_dosage', 'remove_sentences', 'add_confusion']

    # Determine which perturbations to generate
    if args.perturbation:
        if args.perturbation not in all_perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}")
        perturbation_names = [args.perturbation]
        print(f"Generating single perturbation: {args.perturbation}\n")
    else:
        perturbation_names = all_perturbations
        print(f"Generating all perturbations: {', '.join(perturbation_names)}\n")

    # Determine which levels to process
    levels = ['coarse', 'fine'] if args.level == 'both' else [args.level]

    # Statistics
    total_generated = 0
    total_skipped = 0

    # Process each level
    for level in levels:
        print(f"{'='*80}")
        print(f"LEVEL: {level.upper()}")
        print(f"{'='*80}\n")

        # Load data
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']
        qa_pairs = load_qa_data(data_path)
        id_key = get_id_key(qa_pairs)

        print(f"Loaded {len(qa_pairs)} QA pairs from {os.path.basename(data_path)}\n")

        # Process each perturbation
        for perturbation_name in perturbation_names:
            print(f"{'-'*80}")
            print(f"Perturbation: {perturbation_name.upper()}")
            print(f"{'-'*80}")

            # Determine parameter values
            remove_pct_values = [args.remove_pct]
            if perturbation_name == 'remove_sentences' and args.all_remove_pct:
                remove_pct_values = [0.3, 0.5, 0.7]
                print(f"Generating for remove_pct: {remove_pct_values}")

            typo_prob_values = [args.typo_prob]
            if perturbation_name == 'add_typos' and args.all_typo_prob:
                typo_prob_values = [0.3, 0.5, 0.7]
                print(f"Generating for typo_prob: {typo_prob_values}")

            # Iterate over parameter combinations
            for remove_pct in remove_pct_values:
                for typo_prob in typo_prob_values:
                    if perturbation_name == 'remove_sentences' and len(remove_pct_values) > 1:
                        print(f"  remove_pct={remove_pct}")
                    if perturbation_name == 'add_typos' and len(typo_prob_values) > 1:
                        print(f"  typo_prob={typo_prob}")

                    # Use get_or_create_perturbations to handle partial files automatically
                    perturbations_dict = get_or_create_perturbations(
                        perturbation_name=perturbation_name,
                        level=level,
                        qa_pairs=qa_pairs,
                        typo_prob=typo_prob,
                        remove_pct=remove_pct,
                        seed=args.seed,
                        output_dir=output_dir
                    )

                    generated = len(perturbations_dict)
                    skipped = len(qa_pairs) - generated

                    total_generated += generated
                    total_skipped += skipped

            print()  # Blank line between perturbations

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total perturbations: {total_generated}")
    print(f"Total skipped: {total_skipped}")
    print(f"\nPerturbation files saved to: {perturbations_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review perturbations: ls {perturbations_dir}/")
    print(f"  2. Run experiments:")
    print(f"     python experiment_runner.py --experiment baseline --model <model> --seed {args.seed}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate perturbations for M3-Eval experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all perturbations with seed 42
  python code/generate_perturbations.py --seed 42

  # Generate only typos with high probability
  python code/generate_perturbations.py --perturbation add_typos --typo-prob 0.9 --seed 42

  # Generate all removal percentage variations
  python code/generate_perturbations.py --perturbation remove_sentences --all-remove-pct --seed 42
        """
    )

    # Perturbation selection
    parser.add_argument(
        '--perturbation',
        type=str,
        default=None,
        help='Specific perturbation to generate. Options: add_typos, change_dosage, remove_sentences, add_confusion. If not specified, generates all.'
    )

    # Level selection
    parser.add_argument(
        '--level',
        type=str,
        default='both',
        choices=['coarse', 'fine', 'both'],
        help='Evaluation level (default: both)'
    )

    # Perturbation parameters
    parser.add_argument(
        '--remove-pct',
        type=float,
        default=0.3,
        help='For remove_sentences: percentage of sentences to remove (0.0-1.0). Default: 0.3 (30%%)'
    )
    parser.add_argument(
        '--all-remove-pct',
        action='store_true',
        help='For remove_sentences: generate all percentage values (0.3, 0.5, 0.7)'
    )
    parser.add_argument(
        '--typo-prob',
        type=float,
        default=0.5,
        help='For add_typos: probability of applying typo (0.0-1.0). Default: 0.5'
    )
    parser.add_argument(
        '--all-typo-prob',
        action='store_true',
        help='For add_typos: generate all probability values (0.3, 0.5, 0.7)'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: project_root/output)'
    )

    args = parser.parse_args()

    generate_perturbations(args)


if __name__ == "__main__":
    main()
