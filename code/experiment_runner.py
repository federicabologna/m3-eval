#!/usr/bin/env python3
"""
Central experiment runner for M3-Eval perturbation experiments.

Supports multiple experiment types:
- baseline: Original perturbation + rating pipeline
- error_detection: Ask models to detect errors in perturbed answers
- error_priming: Compare ratings with/without error warnings

Usage:
    python experiment_runner.py --experiment baseline --model Qwen3-8B
    python experiment_runner.py --experiment error_detection --model gpt-4o
    python experiment_runner.py --experiment error_priming --model claude-opus-4-5-20251101
"""

import argparse
import sys
import os
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_data_only(args):
    """
    Generate original ratings and perturbations without running experiments.
    """
    from helpers.experiment_utils import (
        setup_paths,
        load_qa_data,
        clean_model_name,
        get_or_create_original_ratings,
        get_or_create_perturbations
    )
    from helpers.multi_llm_inference import get_provider_from_model

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    paths = setup_paths(args.output_dir)
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Define perturbations to generate (starting with change_dosage and remove_sentences)
    all_perturbations = ['change_dosage', 'remove_sentences', 'add_typos', 'add_confusion']
    if args.perturbation:
        if args.perturbation not in all_perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}")
        perturbation_names = [args.perturbation]
    else:
        perturbation_names = all_perturbations

    # Determine which levels to process
    levels = ['coarse', 'fine'] if args.level == 'both' else [args.level]

    print(f"\n{'='*80}")
    print("DATA GENERATION MODE")
    print(f"{'='*80}")
    print(f"Levels: {', '.join(levels)}")
    print(f"Perturbations: {', '.join(perturbation_names)}")

    # Process each level
    for level in levels:
        print(f"\n{'='*80}")
        print(f"LEVEL: {level.upper()}")
        print(f"{'='*80}")

        # Select data path and prompt path
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']
        prompt_path = os.path.join(paths['prompts_dir'], f'{level}prompt_system.txt')
        print(f"Using data: {data_path}")

        # Load data
        all_qa_pairs = load_qa_data(data_path)
        print(f"Loaded {len(all_qa_pairs)} examples")

        # Apply start/end index filtering if specified
        if args.start_idx is not None or args.end_idx is not None:
            start = args.start_idx if args.start_idx is not None else 0
            end = args.end_idx if args.end_idx is not None else len(all_qa_pairs)
            qa_pairs = all_qa_pairs[start:end]
            print(f"Using subset: indices {start} to {end} ({len(qa_pairs)} examples)")
        else:
            qa_pairs = all_qa_pairs

        # Step 1: Generate/check original ratings
        print(f"\n{'-'*80}")
        print("STEP 1: ORIGINAL RATINGS")
        print(f"{'-'*80}")

        original_ratings_dict = get_or_create_original_ratings(
            qa_pairs=qa_pairs,
            level=level,
            prompt_path=prompt_path,
            model=args.model,
            output_dir=output_dir,
            model_name_clean=model_name_clean,
            num_runs=args.num_runs
        )

        # Step 2: Generate/check perturbations
        print(f"\n{'-'*80}")
        print("STEP 2: PERTURBATIONS")
        print(f"{'-'*80}")

        for perturbation_name in perturbation_names:
            print(f"\n[{perturbation_name}]")

            # Determine parameter values
            remove_pct_values = [args.remove_pct]
            if perturbation_name == 'remove_sentences' and args.all_remove_pct:
                remove_pct_values = [0.3, 0.5, 0.7]

            typo_prob_values = [args.typo_prob]
            if perturbation_name == 'add_typos' and args.all_typo_prob:
                typo_prob_values = [0.3, 0.5, 0.7]

            # Generate for each parameter combination
            for remove_pct in remove_pct_values:
                for typo_prob in typo_prob_values:
                    if perturbation_name == 'remove_sentences' and len(remove_pct_values) > 1:
                        print(f"  remove_pct={remove_pct}")
                    if perturbation_name == 'add_typos' and len(typo_prob_values) > 1:
                        print(f"  typo_prob={typo_prob}")

                    perturbations_dict = get_or_create_perturbations(
                        perturbation_name=perturbation_name,
                        level=level,
                        qa_pairs=qa_pairs,
                        typo_prob=typo_prob,
                        remove_pct=remove_pct,
                        seed=args.seed,
                        output_dir=output_dir
                    )

    print(f"\n{'='*80}")
    print("DATA GENERATION COMPLETED")
    print(f"{'='*80}")
    print(f"\nOriginal ratings saved to: {output_dir}/original_ratings/")
    print(f"Perturbations saved to: {output_dir}/perturbations/")
    print(f"\nYou can now run experiments with: --experiment [baseline|error_detection|error_priming]")


def main():
    parser = argparse.ArgumentParser(
        description='Run M3-Eval perturbation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Types:
  baseline          Original perturbation + rating pipeline
  error_detection   Detect if answers contain errors (open-ended + yes/no)
  error_priming     Compare ratings with/without error warnings

Workflow:
  1. Generate original ratings and perturbations (--generate-only)
  2. Run experiments (--experiment)

Examples:
  # Step 1: Generate data only (no experiments)
  python experiment_runner.py --generate-only --model Qwen3-8B --seed 42

  # Step 2: Run baseline experiment
  python experiment_runner.py --experiment baseline --model Qwen3-8B --seed 42

  # Run error detection experiment
  python experiment_runner.py --experiment error_detection --model gpt-4o

  # Run error priming experiment
  python experiment_runner.py --experiment error_priming --model claude-opus-4-5-20251101

  # Run everything in one command (generates data + runs experiment)
  python experiment_runner.py --experiment baseline --model Qwen3-8B
        """
    )

    # Experiment selection
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['baseline', 'error_detection', 'error_priming'],
        help='Type of experiment to run'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen3-8B',
        help='Model name (default: Qwen3-8B). Examples: gpt-4o, claude-opus-4-5-20251101, gemini-2.0-flash-exp'
    )

    # Perturbation selection
    parser.add_argument(
        '--perturbation',
        type=str,
        default=None,
        help='Specific perturbation to run. Options: add_typos, change_dosage, remove_sentences, add_confusion. If not specified, runs all.'
    )

    # Level selection
    parser.add_argument(
        '--level',
        type=str,
        default='both',
        choices=['coarse', 'fine', 'both'],
        help='Evaluation level to run (default: both)'
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
        help='For remove_sentences: run all percentage values (0.3, 0.5, 0.7) sequentially'
    )
    parser.add_argument(
        '--typo-prob',
        type=float,
        default=0.5,
        help='For add_typos: probability of applying typo to each term (0.0-1.0). Default: 0.5'
    )
    parser.add_argument(
        '--all-typo-prob',
        action='store_true',
        help='For add_typos: run all probability values (0.3, 0.5, 0.7) sequentially'
    )

    # Rating parameters
    parser.add_argument(
        '--num-runs',
        type=int,
        default=5,
        help='Number of rating runs to average (default: 5)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Max retries for invalid ratings (default: 3)'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Data subsetting
    parser.add_argument(
        '--start-idx',
        type=int,
        default=None,
        help='Start index for data subset (default: 0)'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='End index for data subset (default: all data)'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: project_root/output)'
    )

    # Data generation mode
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate original ratings and perturbations, do not run experiments'
    )

    args = parser.parse_args()

    # If generate-only mode, run data generation and exit
    if args.generate_only:
        generate_data_only(args)
        return

    # Import and run the selected experiment
    if args.experiment == 'baseline':
        from experiments.baseline import run_baseline_experiment
        run_baseline_experiment(args)

    elif args.experiment == 'error_detection':
        from experiments.error_detection import run_error_detection_experiment
        run_error_detection_experiment(args)

    elif args.experiment == 'error_priming':
        from experiments.error_priming import run_error_priming_experiment
        run_error_priming_experiment(args)

    else:
        parser.error(f"Unknown experiment type: {args.experiment}")


if __name__ == "__main__":
    main()
