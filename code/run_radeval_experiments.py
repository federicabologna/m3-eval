"""
Run RadEval experiments with GREEN metric.

Supports multiple experiment types:
- baseline: Original perturbation + rating pipeline
- error_priming: Compare ratings with/without error warnings

Usage:
    python run_radeval_experiments.py --experiment baseline --model gpt-4.1-2025-04-14
    python run_radeval_experiments.py --experiment error_priming --model gpt-4.1-2025-04-14
"""

import argparse
import random
import time
from helpers.radeval_experiment_utils import (
    setup_radeval_paths,
    load_radeval_data,
    get_processed_ids,
    get_or_create_radeval_perturbations,
    get_or_create_radeval_original_ratings,
    get_or_create_radeval_chexbert_ratings,
    save_result,
    clean_model_name
)
from helpers.green_eval import get_green_rating
from helpers.chexbert_eval import get_chexbert_rating


def run_radeval_baseline_experiments(args):
    """Run baseline experiments on RadEval dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Determine evaluation mode
    if args.chexbert:
        print(f"Running CheXbert evaluation on device: {args.device}")
    else:
        print(f"Using GREEN model: {args.model or 'StanfordAIMI/GREEN-radllama2-7b'}")

    # Map dataset names to file paths
    dataset_paths = {
        'radeval': 'data/radeval_expert_dataset.jsonl',
        'rexerr': 'data/rexerr_acceptable_dataset.jsonl'
    }

    # Determine data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, dataset_paths[args.dataset])

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', args.dataset)
    else:
        output_dir = args.output_dir

    # Setup paths
    paths = setup_radeval_paths(output_dir, data_path)
    output_dir = paths['output_dir']
    data_path = paths['data_path']

    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")

    # Load data
    print("\nLoading RadEval data...")
    all_data = load_radeval_data(data_path)
    print(f"Loaded {len(all_data)} examples")

    # Apply start/end index filtering if specified
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else len(all_data)
        data = all_data[start:end]
        print(f"Using subset: indices {start} to {end} ({len(data)} examples)")
    else:
        data = all_data

    # Determine field names based on dataset structure
    # Adjust these based on actual RadEval dataset fields
    text_field = args.text_field  # e.g., 'prediction', 'generated_report'
    reference_field = args.reference_field  # e.g., 'reference', 'ground_truth'

    # Define perturbations with only their relevant parameters
    all_perturbations = {
        'swap_qualifiers': {},  # No parameters
        'swap_organs': {},      # No parameters
        'add_typos': {'typo_prob': [0.3, 0.5, 0.7]},
        'remove_sentences': {'remove_pct': [0.3, 0.5, 0.7]},
        # LLM-based perturbations using rexerr prompts (error types 5, 10, 11)
        'inject_false_prediction': {},  # No parameters (LLM decides)
        'inject_contradiction': {},     # No parameters (LLM decides)
        'inject_false_negation': {}     # No parameters (LLM decides)
    }

    # Default to LLM-based perturbations only
    default_llm_perturbations = {
        'inject_false_prediction': {},
        'inject_contradiction': {},
        'inject_false_negation': {}
    }

    if args.perturbation:
        perturbations_to_run = {args.perturbation: all_perturbations[args.perturbation]}
    else:
        perturbations_to_run = default_llm_perturbations

    print(f"\nPerturbations to run: {list(perturbations_to_run.keys())}")

    # Step 1: Get/compute original ratings
    print(f"\n{'='*80}")
    if args.chexbert:
        print("STEP 1: ORIGINAL CHEXBERT RATINGS")
    else:
        print("STEP 1: ORIGINAL GREEN RATINGS")
    print(f"{'='*80}")

    original_ratings_dict = {}
    original_chexbert_ratings_dict = {}

    if args.chexbert:
        # CheXbert evaluation
        original_chexbert_ratings_dict = get_or_create_radeval_chexbert_ratings(
            data=data,
            text_field=text_field,
            reference_field=reference_field,
            output_dir=output_dir,
            device=args.device
        )
    else:
        # GREEN evaluation
        original_ratings_dict = get_or_create_radeval_original_ratings(
            data=data,
            text_field=text_field,
            reference_field=reference_field,
            output_dir=output_dir,
            model=args.model,
            cpu=args.cpu,
            num_runs=5  # Use n=5 for averaging (for API models like GPT-4)
        )

    # Step 2: Process each perturbation
    print(f"\n{'='*80}")
    print("STEP 2: PERTURBATIONS AND EVALUATION")
    print(f"{'='*80}")

    # Create baseline experiment directory
    baseline_dir = os.path.join(output_dir, 'experiment_results', 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)

    for perturbation_name, params in perturbations_to_run.items():
        print(f"\n[{perturbation_name.upper()}]")

        # Create perturbation-specific subdirectory
        perturbation_dir = os.path.join(baseline_dir, perturbation_name)
        os.makedirs(perturbation_dir, exist_ok=True)

        # Build parameter combinations based on what this perturbation actually uses
        if 'typo_prob' in params and 'remove_pct' in params:
            # Both parameters (shouldn't happen now, but handle it)
            param_combinations = [(tp, rp) for tp in params['typo_prob'] for rp in params['remove_pct']]
        elif 'typo_prob' in params:
            # Only typo_prob (e.g., add_typos)
            param_combinations = [(tp, None) for tp in params['typo_prob']]
        elif 'remove_pct' in params:
            # Only remove_pct (e.g., remove_sentences)
            param_combinations = [(None, rp) for rp in params['remove_pct']]
        else:
            # No parameters (e.g., swap_qualifiers, swap_organs, LLM-based)
            param_combinations = [(None, None)]

        for typo_prob, remove_pct in param_combinations:
            # Determine output filename based on metric and parameters
            if args.chexbert:
                # CheXbert filenames (no model name)
                if perturbation_name == 'remove_sentences':
                    pct_str = str(int(remove_pct * 100))
                    output_filename = f"{perturbation_name}_{pct_str}pct_chexbert_rating.jsonl"
                elif perturbation_name == 'add_typos':
                    prob_str = str(typo_prob).replace('.', '')
                    output_filename = f"{perturbation_name}_{prob_str}prob_chexbert_rating.jsonl"
                else:
                    output_filename = f"{perturbation_name}_chexbert_rating.jsonl"
            else:
                # GREEN filenames (include model name)
                model_name_clean = clean_model_name(args.model) if args.model else "GREEN-radllama2-7b"
                if perturbation_name == 'remove_sentences':
                    pct_str = str(int(remove_pct * 100))
                    output_filename = f"{perturbation_name}_{pct_str}pct_{model_name_clean}_green_rating.jsonl"
                elif perturbation_name == 'add_typos':
                    prob_str = str(typo_prob).replace('.', '')
                    output_filename = f"{perturbation_name}_{prob_str}prob_{model_name_clean}_green_rating.jsonl"
                else:
                    output_filename = f"{perturbation_name}_{model_name_clean}_green_rating.jsonl"

            output_path = os.path.join(perturbation_dir, output_filename)

            # Check which entries have already been processed
            processed_ids = get_processed_ids(output_path)
            remaining_data = [item for item in data if item['id'] not in processed_ids]

            if len(remaining_data) == 0:
                print(f"  ✓ {output_filename}: All {len(data)} entries complete")
                continue

            print(f"  Processing: {output_filename}")
            print(f"    {len(remaining_data)} remaining (out of {len(data)})")

            # Load or generate perturbations (use defaults for None values)
            perturbations_dict = get_or_create_radeval_perturbations(
                perturbation_name=perturbation_name,
                data=data,
                text_field=text_field,
                typo_prob=typo_prob if typo_prob is not None else 0.5,
                remove_pct=remove_pct if remove_pct is not None else 0.3,
                seed=args.seed,
                output_dir=output_dir,
                llm_model=args.perturbation_model if hasattr(args, 'perturbation_model') else 'gpt-4.1'
            )

            # Process each entry
            for idx, item in enumerate(remaining_data, 1):
                reference = item[reference_field]
                item_id = item['id']

                # Get pre-generated perturbation
                perturbation_entry = perturbations_dict.get(item_id)

                if perturbation_entry is None:
                    print(f"    Skipping {item_id} - no perturbation found")
                    continue

                perturbed_text = perturbation_entry[f'perturbed_{text_field}']

                print(f"    [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

                start_time = time.time()

                if args.chexbert:
                    # Process CheXbert
                    perturbed_rating = get_chexbert_rating(
                        perturbed_text, reference,
                        device=args.device
                    )
                    elapsed_time = time.time() - start_time
                    print(f"{elapsed_time:.1f}s")

                    # Get original rating from dict
                    original_rating = original_chexbert_ratings_dict.get(item_id)

                    if original_rating is None:
                        print(f"    WARNING: No original CheXbert rating found for {item_id}, skipping...")
                        continue

                    # Build result
                    result = item.copy()
                    result['perturbation'] = perturbation_name
                    result[f'perturbed_{text_field}'] = perturbed_text
                    result['original_chexbert_rating'] = original_rating
                    result['perturbed_chexbert_rating'] = perturbed_rating
                    result['random_seed'] = args.seed

                else:
                    # Process GREEN
                    perturbed_rating = get_green_rating(
                        perturbed_text, reference,
                        model_name=args.model,
                        cpu=args.cpu,
                        num_runs=5  # Use n=5 for averaging
                    )
                    elapsed_time = time.time() - start_time
                    print(f"{elapsed_time:.1f}s")

                    # Get original rating from dict
                    original_rating = original_ratings_dict.get(item_id)

                    if original_rating is None:
                        print(f"    WARNING: No original GREEN rating found for {item_id}, skipping...")
                        continue

                    # Build result
                    result = item.copy()
                    result['perturbation'] = perturbation_name
                    result[f'perturbed_{text_field}'] = perturbed_text
                    result['original_rating'] = original_rating
                    result['perturbed_rating'] = perturbed_rating
                    result['random_seed'] = args.seed

                # Add perturbation metadata
                # For regex perturbations: typo_probability, removal_percentage, etc.
                # For LLM perturbations: changes_detail, num_changes, level, etc.
                metadata_keys = [
                    'typo_probability', 'removal_percentage', 'qualifier_changes', 'organ_changes', 'skip_reason',
                    'changes_detail', 'num_changes', 'parsed_successfully', 'level',
                    'model', 'error', 'raw_response'
                ]
                for key in metadata_keys:
                    if key in perturbation_entry:
                        result[key] = perturbation_entry[key]

                # Save to file
                save_result(output_path, result)

    print(f"\n{'='*80}")
    if args.chexbert:
        print("RADEVAL CHEXBERT EXPERIMENTS COMPLETED")
    else:
        print("RADEVAL GREEN EXPERIMENTS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description='Run RadEval experiments')

    parser.add_argument('--experiment', type=str, default='baseline',
                       choices=['baseline', 'error_priming'],
                       help='Experiment type to run (default: baseline)')

    parser.add_argument('--model', type=str, default=None,
                       help='Model to use for GREEN evaluation. Options: None (default: StanfordAIMI/GREEN-radllama2-7b), gpt-4.1-2025-04-14, gpt-4o, gpt-4o-mini')

    parser.add_argument('--cpu', action='store_true',
                       help='Run GREEN model on CPU instead of GPU (only applies to GREEN model, not API models)')

    parser.add_argument('--chexbert', action='store_true',
                       help='Run CheXbert evaluation instead of GREEN')

    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to use for CheXbert (default: mps for Apple Silicon)')

    parser.add_argument('--dataset', type=str, default='radeval',
                       choices=['radeval', 'rexerr'],
                       help='Dataset to use (default: radeval)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/{dataset})')

    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['add_typos', 'remove_sentences', 'swap_qualifiers', 'swap_organs',
                                'inject_false_prediction', 'inject_contradiction', 'inject_false_negation'],
                       help='Specific perturbation to run (default: LLM-based perturbations only - inject_false_prediction, inject_contradiction, inject_false_negation). Regex-based: add_typos, remove_sentences, swap_qualifiers, swap_organs')

    parser.add_argument('--perturbation-model', type=str, default='gpt-4.1',
                       help='Model to use for LLM-based perturbations (default: gpt-4.1). Options: gpt-4.1, gpt-4o, claude-opus-4-5, etc.')

    parser.add_argument('--text-field', type=str, default='prediction',
                       help='Field name containing text to perturb (default: prediction)')

    parser.add_argument('--reference-field', type=str, default='reference',
                       help='Field name containing reference text (default: reference)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--start-idx', type=int, default=None,
                       help='Start index for data subset (default: 0)')

    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for data subset (default: all data)')

    args = parser.parse_args()

    # Route to appropriate experiment
    if args.experiment == 'baseline':
        run_radeval_baseline_experiments(args)
    elif args.experiment == 'error_priming':
        from experiments.error_priming_radeval import run_error_priming_radeval
        run_error_priming_radeval(args)
