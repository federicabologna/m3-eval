#!/usr/bin/env python3
"""
Experiment runner for MedInfo medication QA perturbation experiments.

Supports multiple experiment types:
- baseline: Original perturbation + rating pipeline
- error_detection: Ask models to detect errors in perturbed answers
- error_priming: Compare ratings with/without error warnings

Default perturbations: LLM-based (critical and non-critical medication errors)
Optional: Regex-based (change_dosage, remove_sentences, add_typos)

Usage:
    python run_medinfo_experiments.py --experiment baseline --model gpt-4.1
    python run_medinfo_experiments.py --experiment baseline --model gpt-4.1 --perturbation inject_critical_error
    python run_medinfo_experiments.py --experiment baseline --model gpt-4.1 --perturbation change_dosage  # regex-based
"""

import argparse
import sys
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

    # Setup paths (fixed to medinfo dataset)
    paths = setup_paths(args.output_dir, dataset='medinfo')
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Define perturbations to generate
    # LLM-based perturbations (default)
    llm_perturbations_coarse = ['inject_critical_error', 'inject_noncritical_error']
    llm_perturbations_fine = ['inject_critical_error', 'inject_noncritical_error']

    # Regex-based perturbations (optional)
    regex_perturbations_coarse = ['change_dosage', 'remove_sentences', 'add_typos']
    regex_perturbations_fine = ['change_dosage', 'add_typos']

    # Combine all available perturbations
    all_perturbations_coarse = llm_perturbations_coarse + regex_perturbations_coarse
    all_perturbations_fine = llm_perturbations_fine + regex_perturbations_fine

    if args.perturbation:
        # Check if perturbation is valid for the level
        if args.level == 'fine' and args.perturbation not in all_perturbations_fine:
            print(f"Warning: {args.perturbation} is not supported for fine level")
            print(f"Fine level only supports: {', '.join(all_perturbations_fine)}")
            return
        perturbation_names = [args.perturbation]
    else:
        # Default to LLM-based perturbations
        if args.level == 'fine':
            perturbation_names = llm_perturbations_fine
        elif args.level == 'coarse':
            perturbation_names = llm_perturbations_coarse
        else:  # both
            # For 'both', use LLM perturbations for both levels
            perturbation_names = llm_perturbations_coarse

    # Determine which levels to process
    levels = ['coarse', 'fine'] if args.level == 'both' else [args.level]

    print(f"\n{'='*80}")
    print("DATA GENERATION MODE - MEDINFO")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Levels: {', '.join(levels)}")
    print(f"\nPerturbations to generate (LLM-based by default):")

    for level in levels:
        print(f"\n  {level.upper()}:")
        if args.perturbation:
            level_perturbations = [args.perturbation]
        else:
            level_perturbations = llm_perturbations_fine if level == 'fine' else llm_perturbations_coarse

        for pert_name in level_perturbations:
            if pert_name == 'remove_sentences':
                if args.remove_pct is None:
                    print(f"    - {pert_name}: 30%, 50%, 70%")
                else:
                    print(f"    - {pert_name}: {int(args.remove_pct * 100)}%")
            elif pert_name == 'add_typos':
                if args.typo_prob is None:
                    print(f"    - {pert_name}: probability 0.3, 0.5, 0.7")
                else:
                    print(f"    - {pert_name}: probability {args.typo_prob}")
            else:
                print(f"    - {pert_name}")

    print(f"{'='*80}")

    # Process each level
    for level in levels:
        print(f"\n{'='*80}")
        print(f"LEVEL: {level.upper()}")
        print(f"{'='*80}")

        # Select data path and prompt path
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']
        prompt_path = os.path.join(paths['prompts_dir'], f'{level}prompt_system.txt')
        print(f"Using data: {data_path}")

        # For fine level: Use full dataset for perturbation generation
        # Subset will be created later from successful perturbations
        all_qa_pairs = load_qa_data(data_path)
        print(f"Loaded {len(all_qa_pairs)} examples")

        if level == 'fine':
            print(f"Note: Using full fine dataset to maximize perturbation coverage")
            print(f"      Subset will be created from successful perturbations")

        # Apply start/end index filtering if specified
        if args.start_idx is not None or args.end_idx is not None:
            start = args.start_idx if args.start_idx is not None else 0
            end = args.end_idx if args.end_idx is not None else len(all_qa_pairs)
            qa_pairs = all_qa_pairs[start:end]
            print(f"Using subset: indices {start} to {end} ({len(qa_pairs)} examples)")
        else:
            qa_pairs = all_qa_pairs

        # For fine level in generate-only mode: skip original ratings
        # (they will be generated later from the subset after running create_subset_from_perturbations.py)
        if level != 'fine':
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
        step_num = "STEP 1" if level == 'fine' else "STEP 2"
        print(f"{step_num}: PERTURBATIONS")
        print(f"{'-'*80}")

        for perturbation_name in perturbation_names:
            print(f"\n[{perturbation_name}]")

            # Determine parameter values
            # If None (not specified), use all values; otherwise use the specified value
            if perturbation_name == 'remove_sentences':
                remove_pct_values = [0.3, 0.5, 0.7] if args.remove_pct is None else [args.remove_pct]
            else:
                remove_pct_values = [0.3]  # Default for other perturbations

            if perturbation_name == 'add_typos':
                typo_prob_values = [0.3, 0.5, 0.7] if args.typo_prob is None else [args.typo_prob]
            else:
                typo_prob_values = [0.5]  # Default for other perturbations

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
                        output_dir=output_dir,
                        model=args.model
                    )

    print(f"\n{'='*80}")
    print("DATA GENERATION COMPLETED")
    print(f"{'='*80}")

    if 'fine' in levels:
        print(f"\nPerturbations saved to: {output_dir}/perturbations/")
        print(f"\nNext steps for fine level:")
        print(f"  1. Run: python code/create_subset_from_perturbations.py")
        print(f"  2. Run experiments with: --experiment [baseline|error_detection|error_priming]")
        print(f"     (Original ratings will be generated automatically for the subset)")
    else:
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

Default Perturbations (LLM-based):
  inject_critical_error      - Inject life-threatening medication errors (10x-50x overdoses)
  inject_noncritical_error   - Inject unusual but safe errors (2-3x typical dose)

Optional Perturbations (Regex-based):
  change_dosage              - Modify medication dosages
  remove_sentences           - Remove sentences from answers
  add_typos                  - Add typos to answers

Examples:
  # Run all LLM perturbations (default)
  python run_medinfo_experiments.py --experiment baseline --model gpt-4.1

  # Run specific LLM perturbation
  python run_medinfo_experiments.py --experiment baseline --model gpt-4.1 --perturbation inject_critical_error

  # Run regex-based perturbation
  python run_medinfo_experiments.py --experiment baseline --model gpt-4.1 --perturbation change_dosage

  # Use different model for perturbations vs rating
  python run_medinfo_experiments.py --experiment baseline --model gpt-4.1 --perturbation-model gpt-4o

  # Generate data only (no experiments)
  python run_medinfo_experiments.py --generate-only --model gpt-4.1

  # Run error detection
  python run_medinfo_experiments.py --experiment error_detection --model gpt-4o

  # Run on subset for testing
  python run_medinfo_experiments.py --experiment baseline --model gpt-4.1 --start-idx 0 --end-idx 10
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
        required=True,
        help='Model to use for rating/answering and LLM perturbations. Examples: gpt-4.1, gpt-4o, claude-opus-4-5-20251101'
    )

    # Perturbation selection
    parser.add_argument(
        '--perturbation',
        type=str,
        default=None,
        help='Specific perturbation to run. Default: LLM-based (inject_critical_error, inject_noncritical_error). Regex-based: change_dosage, remove_sentences, add_typos'
    )

    parser.add_argument(
        '--perturbation-model',
        type=str,
        default=None,
        help='Model to use for LLM-based perturbations (default: same as --model). Examples: gpt-4.1, gpt-4o, claude-opus-4-5'
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
        default=None,
        help='For remove_sentences: specific percentage to use (0.0-1.0). If not specified, runs all values: 0.3, 0.5, 0.7'
    )
    parser.add_argument(
        '--typo-prob',
        type=float,
        default=None,
        help='For add_typos: specific probability to use (0.0-1.0). If not specified, runs all values: 0.3, 0.5, 0.7'
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

    # Dataset is fixed to medinfo for this pipeline
    # For CQA eval experiments, use run_cqa_experiments.py

    # Data generation mode
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate original ratings and perturbations, do not run experiments'
    )

    args = parser.parse_args()

    # Add dataset attribute (fixed to medinfo for this pipeline)
    args.dataset = 'medinfo'

    # If generate-only mode, run data generation and exit
    if args.generate_only:
        generate_data_only(args)
        return

    # Import and run the selected experiment
    if args.experiment == 'baseline':
        from experiments.baseline_medinfo import run_baseline_experiment
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
