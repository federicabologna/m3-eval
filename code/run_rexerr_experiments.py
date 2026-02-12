"""
Run RexErr experiments with GREEN metric.

Supports multiple experiment types:
- baseline: Evaluate pre-existing RexErr perturbations
- error_priming: Compare ratings with/without error warnings
- error_detection: Reference-free error detection
- error_detection_with_reference: Reference-based error detection
- error_criticality: Classify error criticality (minor, moderate, critical)

Usage:
    python run_rexerr_experiments.py --experiment baseline --model gpt-4.1-2025-04-14
    python run_rexerr_experiments.py --experiment error_priming --model gpt-4.1-2025-04-14
    python run_rexerr_experiments.py --experiment error_criticality --model gpt-4.1-2025-04-14
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


def run_rexerr_baseline_experiments(args):
    """Run baseline experiments on RexErr dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Determine evaluation mode
    if args.chexbert:
        print(f"Running CheXbert evaluation on device: {args.device}")
    else:
        print(f"Using GREEN model: {args.model or 'StanfordAIMI/GREEN-radllama2-7b'}")

    # Use RexErr dataset
    dataset = 'rexerr'

    # Determine data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'rexerr_acceptable_dataset.jsonl')

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'rexerr')
    else:
        output_dir = args.output_dir

    # Setup paths
    paths = setup_radeval_paths(output_dir, data_path)
    output_dir = paths['output_dir']
    data_path = paths['data_path']

    print(f"Dataset: RexErr")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")

    # Load data
    print("\nLoading RexErr data...")
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

    # RexErr dataset structure
    # - 'prediction': error_report (report with errors already injected)
    # - 'reference': original_report (ground truth without errors)
    # - 'errors_sampled': list of error types that were injected
    text_field = 'prediction'  # Perturbed report (with errors)
    reference_field = 'reference'  # Original report (ground truth)

    print(f"\n{'='*80}")
    print("REXERR DATASET STRUCTURE")
    print(f"{'='*80}")
    print("RexErr already contains pre-generated perturbations:")
    print("  - 'prediction': Report with errors (perturbed)")
    print("  - 'reference': Original correct report (ground truth)")
    print("  - 'errors_sampled': Types of errors injected")
    print("\nNo perturbation generation needed - will evaluate existing pairs.")

    # Determine model name for output files
    if args.chexbert:
        model_name_clean = "chexbert"
        metric_name = "chexbert"
    else:
        model_name_clean = clean_model_name(args.model) if args.model else "GREEN-radllama2-7b"
        metric_name = "green"

    # First, evaluate original reports against themselves (control)
    print(f"\n{'='*80}")
    if args.chexbert:
        print("EVALUATING ORIGINAL REPORTS WITH CHEXBERT (CONTROL)")
    else:
        print("EVALUATING ORIGINAL REPORTS WITH GREEN (CONTROL)")
    print(f"{'='*80}")
    print("Computing: rating = similarity(reference, reference)")
    print("  - reference: original correct report")
    print("  - Should yield near-perfect scores (≈1.0)")

    # Create original ratings directory
    original_dir = os.path.join(output_dir, 'original_ratings')
    os.makedirs(original_dir, exist_ok=True)

    # Determine output filename for originals
    if args.chexbert:
        original_filename = f"original_rexerr_{model_name_clean}_chexbert.jsonl"
    else:
        original_filename = f"original_rexerr_{model_name_clean}_green.jsonl"

    original_output_path = os.path.join(original_dir, original_filename)

    # Check which original entries have already been processed
    processed_original_ids = get_processed_ids(original_output_path)
    remaining_original_data = [item for item in data if item['id'] not in processed_original_ids]

    if len(remaining_original_data) == 0:
        print(f"✓ All {len(data)} original reports already processed")
        print(f"✓ Results saved to: {original_output_path}")
    else:
        print(f"Processing {len(remaining_original_data)} remaining original reports")

        for idx, item in enumerate(remaining_original_data, 1):
            reference = item[reference_field]
            item_id = item['id']

            print(f"  [{idx}/{len(remaining_original_data)}] {item_id}...", end=" ")

            start_time = time.time()

            if args.chexbert:
                # Evaluate reference against itself
                rating = get_chexbert_rating(
                    reference, reference,
                    device=args.device
                )
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.1f}s")

                result = item.copy()
                result['chexbert_rating'] = rating
                result['random_seed'] = args.seed
                result['report_type'] = 'original'

            else:
                # Evaluate reference against itself
                rating = get_green_rating(
                    reference, reference,
                    model_name=args.model,
                    cpu=args.cpu,
                    num_runs=5
                )
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.1f}s")

                result = item.copy()
                result['green_rating'] = rating
                result['random_seed'] = args.seed
                result['report_type'] = 'original'

            save_result(original_output_path, result)

        print(f"✓ Completed original reports")
        print(f"✓ Results saved to: {original_output_path}")

    # Now evaluate how well perturbed reports (prediction) match originals (reference)
    print(f"\n{'='*80}")
    if args.chexbert:
        print("EVALUATING PERTURBED REPORTS WITH CHEXBERT")
    else:
        print("EVALUATING PERTURBED REPORTS WITH GREEN")
    print(f"{'='*80}")
    print("Computing: rating = similarity(prediction, reference)")
    print("  - prediction: error-injected report (from RexErr dataset)")
    print("  - reference: original correct report")
    print("  - Lower rating = more deviation due to injected errors")

    # Create baseline experiment results directory
    results_dir = os.path.join(output_dir, 'experiment_results', 'baseline')
    os.makedirs(results_dir, exist_ok=True)

    # Determine output filename
    output_filename = f"rexerr_evaluation_{model_name_clean}_{metric_name}.jsonl"
    output_path = os.path.join(results_dir, output_filename)

    # Check which entries have already been processed
    processed_ids = get_processed_ids(output_path)
    remaining_data = [item for item in data if item['id'] not in processed_ids]

    if len(remaining_data) == 0:
        print(f"✓ All {len(data)} entries already processed")
        print(f"✓ Results saved to: {output_path}")
    else:
        print(f"Processing {len(remaining_data)} remaining entries (out of {len(data)})")

        # Process each entry
        for idx, item in enumerate(remaining_data, 1):
            reference = item[reference_field]
            perturbed_text = item[text_field]
            item_id = item['id']

            print(f"  [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

            start_time = time.time()

            if args.chexbert:
                # Evaluate perturbed report with CheXbert
                rating = get_chexbert_rating(
                    perturbed_text, reference,
                    device=args.device
                )
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.1f}s")

                # Build result
                result = item.copy()
                result['chexbert_rating'] = rating
                result['random_seed'] = args.seed

            else:
                # Evaluate perturbed report with GREEN
                rating = get_green_rating(
                    perturbed_text, reference,
                    model_name=args.model,
                    cpu=args.cpu,
                    num_runs=5
                )
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.1f}s")

                # Build result
                result = item.copy()
                result['green_rating'] = rating
                result['random_seed'] = args.seed

            # Save to file
            save_result(output_path, result)

        print(f"\n✓ Completed processing {len(remaining_data)} entries")
        print(f"✓ Results saved to: {output_path}")

    print(f"\n{'='*80}")
    if args.chexbert:
        print("REXERR CHEXBERT EXPERIMENTS COMPLETED")
    else:
        print("REXERR GREEN EXPERIMENTS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description='Run RexErr experiments')

    parser.add_argument('--experiment', type=str, default='baseline',
                       choices=['baseline', 'error_priming', 'error_detection', 'error_detection_with_reference', 'error_criticality'],
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

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/rexerr)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--start-idx', type=int, default=None,
                       help='Start index for data subset (default: 0)')

    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for data subset (default: all data)')

    parser.add_argument('--level', type=str, default='coarse',
                       choices=['coarse', 'fine'],
                       help='Detection level for error_detection experiment: coarse (full report) or fine (sentence-level)')

    args = parser.parse_args()

    # Route to appropriate experiment
    if args.experiment == 'baseline':
        run_rexerr_baseline_experiments(args)
    elif args.experiment == 'error_priming':
        from experiments.error_priming_rexerr import run_error_priming_rexerr
        run_error_priming_rexerr(args)
    elif args.experiment == 'error_detection':
        from experiments.error_detection_rexerr import run_error_detection_rexerr
        run_error_detection_rexerr(args)
    elif args.experiment == 'error_detection_with_reference':
        from experiments.error_detection_with_reference_rexerr import run_error_detection_rexerr as run_error_detection_with_reference_rexerr
        run_error_detection_with_reference_rexerr(args)
    elif args.experiment == 'error_criticality':
        from experiments.error_criticality_rexerr import run_error_criticality_rexerr
        run_error_criticality_rexerr(args)
