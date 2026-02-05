"""
Run WoundCare perturbation experiments with 5-run averaging.

This script follows the same pattern as cqa_eval and medinfo experiments:
- Load WoundCare test data with GPT-4o responses
- Apply answer text perturbations (infection, location, time, urgency, severity)
- Evaluate with LLM judge using reference responses
- Average ratings across 5 runs
"""

import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple

# Add helpers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))
from woundcare_answer_perturbations import apply_woundcare_answer_perturbation
from woundcare_prompt_formatter import format_woundcare_evaluation_prompt_with_metadata
from multimodal_inference import load_woundcare_images, get_multimodal_response
from experiment_utils import clean_model_name, get_processed_ids


def parse_woundcare_rating(response_text: str) -> Dict:
    """
    Parse WoundCare rating from LLM response.

    Expected format:
    {
      "rating": 0.5,
      "explanation": "..."
    }
    """
    try:
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            rating_data = json.loads(json_match.group())
            return {
                'rating': float(rating_data.get('rating', 0)),
                'explanation': rating_data.get('explanation', '')
            }
    except:
        pass

    # Fallback: extract rating number
    try:
        rating_match = re.search(r'rating["\s:]*([0-9.]+)', response_text, re.IGNORECASE)
        if rating_match:
            return {
                'rating': float(rating_match.group(1)),
                'explanation': response_text
            }
    except:
        pass

    return {
        'rating': None,
        'explanation': response_text,
        'error': 'Could not parse rating'
    }


def average_woundcare_ratings(ratings_list: List[Dict]) -> Dict:
    """
    Average WoundCare ratings across multiple runs.

    Args:
        ratings_list: List of rating dicts with 'rating' and 'explanation'

    Returns:
        Averaged rating dict
    """
    # Filter out errors
    valid_ratings = [r for r in ratings_list if r.get('rating') is not None]

    if not valid_ratings:
        return {
            'avg_rating': None,
            'explanation': 'No valid ratings',
            'num_runs': len(ratings_list),
            'num_valid': 0,
            'individual_ratings': ratings_list,
            'error': 'No valid ratings'
        }

    # Average ratings
    avg_rating = sum(r['rating'] for r in valid_ratings) / len(valid_ratings)

    return {
        'avg_rating': round(avg_rating, 3),
        'explanation': valid_ratings[0]['explanation'],  # Use first as representative
        'num_runs': len(ratings_list),
        'num_valid': len(valid_ratings),
        'individual_ratings': ratings_list
    }


def get_woundcare_rating_with_averaging(
    qa_pair: Dict,
    candidate_response: str,
    model: str,
    num_runs: int = 5
) -> Dict:
    """
    Get WoundCare rating with averaging across multiple completions.

    For OpenAI models: Uses n parameter (single API call with n completions)
    For other models: Makes n separate API calls

    Args:
        qa_pair: QA pair with reference responses and metadata
        candidate_response: Response to evaluate (GPT-4o answer)
        model: LLM judge model
        num_runs: Number of completions to average (default: 5)

    Returns:
        Averaged rating dict
    """
    from helpers.multi_llm_inference import get_provider_from_model

    provider = get_provider_from_model(model)
    if provider == "openai":
        print(f"  Collecting {num_runs} ratings to average (single API call with n={num_runs})...")
    else:
        print(f"  Collecting {num_runs} ratings to average ({num_runs} separate calls)...")

    # Load images once
    images = []
    if qa_pair.get('image_ids'):
        images = load_woundcare_images(
            qa_pair['image_ids'],
            split=qa_pair.get('split', 'test')
        )

    # Create evaluation prompt
    prompt = format_woundcare_evaluation_prompt_with_metadata(
        qa_pair=qa_pair,
        candidate_response=candidate_response,
        prompts_dir=None
    )

    try:
        # Get n ratings (OpenAI uses single call with n, others make n separate calls)
        rating_responses = get_multimodal_response(
            text=prompt,
            images=images,
            model=model,
            system_message=None,
            n=num_runs,
            return_all=True
        )

        # Parse all ratings
        all_ratings = []
        for i, rating_response in enumerate(rating_responses):
            rating = parse_woundcare_rating(rating_response)
            rating['full_response'] = rating_response
            all_ratings.append(rating)
            if i == 0 or provider != "openai":
                print(f"    Completion {i + 1}/{num_runs}: {rating.get('rating')}", flush=True)

    except Exception as e:
        print(f"    Error getting ratings: {e}")
        all_ratings = [{
            'rating': None,
            'explanation': str(e),
            'error': str(e)
        }]

    # Average ratings
    result = average_woundcare_ratings(all_ratings)
    result['_n_completions'] = num_runs
    result['_provider'] = provider
    return result


def generate_perturbations(
    data_path: str,
    output_dir: str,
    perturbations: List[str],
    seed: int = 42
):
    """
    Phase 1: Generate perturbation files.

    Args:
        data_path: Path to woundcare_gpt4o_coarse.jsonl
        output_dir: Output directory (experiment_results/baseline)
        perturbations: List of perturbation types
        seed: Random seed
    """
    # Set random seed
    random.seed(seed)
    print(f"Random seed set to: {seed}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    qa_pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line))
    print(f"Loaded {len(qa_pairs)} encounters")

    # Perturbations directory (at /output/woundcare/ level, not inside experiment_results)
    woundcare_output_dir = os.path.dirname(os.path.dirname(output_dir))  # Go up from baseline to woundcare
    perturbations_dir = os.path.join(woundcare_output_dir, 'perturbations')
    os.makedirs(perturbations_dir, exist_ok=True)

    # Generate perturbations
    for pert_type in perturbations:
        print(f"\n{'='*80}")
        print(f"GENERATING PERTURBATIONS: {pert_type}")
        print(f"{'='*80}")

        perturbation_path = os.path.join(perturbations_dir, f"{pert_type}_coarse.jsonl")

        # Get already processed IDs
        processed_ids = get_processed_ids(perturbation_path)
        print(f"Already processed: {len(processed_ids)} encounters")

        successful = 0
        skipped_not_applicable = 0
        skipped_already_done = 0

        for qa_pair in qa_pairs:
            question_id = qa_pair['question_id']

            if question_id in processed_ids:
                skipped_already_done += 1
                continue

            try:
                original_answer = qa_pair['gpt4o_response']
                perturbed_answer, success, pert_metadata = apply_woundcare_answer_perturbation(
                    original_answer,
                    pert_type,
                    seed=seed + hash(question_id) % 1000
                )

                if not success:
                    skipped_not_applicable += 1
                    continue

                # Save perturbation data
                pert_data = {
                    'question_id': question_id,
                    'perturbation_type': pert_type,
                    'original_answer': original_answer,
                    'perturbed_answer': perturbed_answer,
                    'perturbation_metadata': pert_metadata,
                    'split': qa_pair.get('split', 'unknown')
                }

                with open(perturbation_path, 'a') as f:
                    f.write(json.dumps(pert_data) + '\n')

                successful += 1

            except Exception as e:
                print(f"  ✗ Error: {question_id}: {e}")
                continue

        print(f"\n{'-'*80}")
        print(f"PERTURBATION SUMMARY: {pert_type}")
        print(f"{'-'*80}")
        print(f"  Successful:            {successful}")
        print(f"  Skipped (already done): {skipped_already_done}")
        print(f"  Skipped (not applicable): {skipped_not_applicable}")
        print(f"  Perturbation file: {perturbation_path}")
        print(f"{'-'*80}")


def get_woundcare_successful_perturbation_ids(
    perturbations: List[str],
    output_dir: str
) -> set:
    """
    Get IDs of successfully perturbed WoundCare cases.

    Args:
        perturbations: List of perturbation types
        output_dir: Output directory (experiment_results/baseline)

    Returns:
        Set of question_ids that were successfully perturbed
    """
    woundcare_output_dir = os.path.dirname(os.path.dirname(output_dir))  # Go up from baseline to woundcare
    perturbations_dir = os.path.join(woundcare_output_dir, 'perturbations')
    successful_ids = set()

    for pert_type in perturbations:
        perturbation_path = os.path.join(perturbations_dir, f"{pert_type}_coarse.jsonl")

        if os.path.exists(perturbation_path):
            with open(perturbation_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Only include if perturbation was successful (no skip_reason)
                    if 'skip_reason' not in entry:
                        successful_ids.add(entry['question_id'])

    return successful_ids


def generate_original_ratings(
    data_path: str,
    output_dir: str,
    model: str,
    perturbations: List[str],
    num_runs: int = 5
):
    """
    Phase 2: Generate original ratings for successfully perturbed cases.

    Args:
        data_path: Path to woundcare_gpt4o_coarse.jsonl
        output_dir: Output directory (experiment_results/baseline)
        model: LLM judge model
        perturbations: List of perturbation types
        num_runs: Number of runs for averaging
    """
    print(f"\nUsing model: {model}")
    print(f"Averaging across {num_runs} runs per rating")

    # Load original data
    print(f"\nLoading data from {data_path}...")
    qa_pairs_dict = {}
    with open(data_path, 'r') as f:
        for line in f:
            qa_pair = json.loads(line)
            qa_pairs_dict[qa_pair['question_id']] = qa_pair
    print(f"Loaded {len(qa_pairs_dict)} encounters")

    # Get successful perturbation IDs
    successful_ids = get_woundcare_successful_perturbation_ids(perturbations, output_dir)
    print(f"Found {len(successful_ids)} successfully perturbed cases")

    # Filter to only successful IDs
    qa_pairs_to_rate = [qa_pairs_dict[qid] for qid in successful_ids if qid in qa_pairs_dict]
    print(f"Will rate {len(qa_pairs_to_rate)} original answers")

    # Setup output directory (at /output/woundcare/ level, not inside experiment_results)
    woundcare_output_dir = os.path.dirname(os.path.dirname(output_dir))  # Go up from baseline to woundcare
    original_ratings_dir = os.path.join(woundcare_output_dir, 'original_ratings')
    os.makedirs(original_ratings_dir, exist_ok=True)

    model_name_clean = clean_model_name(model)
    rating_path = os.path.join(original_ratings_dir, f"original_coarse_{model_name_clean}_rating.jsonl")

    # Get already processed IDs
    processed_ids = get_processed_ids(rating_path)
    print(f"Already processed: {len(processed_ids)} ratings")

    successful = 0
    skipped_already_done = 0
    failed = 0

    for qa_pair in qa_pairs_to_rate:
        question_id = qa_pair['question_id']

        if question_id in processed_ids:
            skipped_already_done += 1
            continue

        try:
            print(f"\n{question_id}:")

            # Get averaged rating for original GPT-4o answer
            original_answer = qa_pair['gpt4o_response']
            averaged_rating = get_woundcare_rating_with_averaging(
                qa_pair=qa_pair,
                candidate_response=original_answer,
                model=model,
                num_runs=num_runs
            )

            # Save rating data
            rating_data = {
                'question_id': question_id,
                'original_rating': averaged_rating,
                'judge_model': model
            }

            with open(rating_path, 'a') as f:
                f.write(json.dumps(rating_data) + '\n')

            successful += 1
            print(f"  ✓ Avg Rating: {averaged_rating['avg_rating']} ({averaged_rating['num_valid']}/{averaged_rating['num_runs']} valid)")

        except Exception as e:
            print(f"  ✗ Error processing {question_id}: {e}")
            failed += 1
            continue

    print(f"\n{'-'*80}")
    print(f"ORIGINAL RATING SUMMARY")
    print(f"{'-'*80}")
    print(f"  Successful:            {successful}")
    print(f"  Skipped (already done): {skipped_already_done}")
    print(f"  Failed:                {failed}")
    print(f"  Rating file: {rating_path}")
    print(f"{'-'*80}")


def generate_perturbed_ratings(
    data_path: str,
    output_dir: str,
    model: str,
    perturbations: List[str],
    num_runs: int = 5
):
    """
    Phase 3: Generate ratings for perturbed answers.

    Args:
        data_path: Path to woundcare_gpt4o_coarse.jsonl (for reference responses)
        output_dir: Output directory (experiment_results/baseline)
        model: LLM judge model
        perturbations: List of perturbation types
        num_runs: Number of runs for averaging
    """
    print(f"\nUsing model: {model}")
    print(f"Averaging across {num_runs} runs per rating")

    # Load original data for reference responses and metadata
    print(f"\nLoading data from {data_path}...")
    qa_pairs_dict = {}
    with open(data_path, 'r') as f:
        for line in f:
            qa_pair = json.loads(line)
            qa_pairs_dict[qa_pair['question_id']] = qa_pair
    print(f"Loaded {len(qa_pairs_dict)} encounters")

    # Load original ratings (at /output/woundcare/ level)
    woundcare_output_dir = os.path.dirname(os.path.dirname(output_dir))  # Go up from baseline to woundcare
    original_ratings_dir = os.path.join(woundcare_output_dir, 'original_ratings')
    model_name_clean = clean_model_name(model)
    original_ratings_path = os.path.join(original_ratings_dir, f"original_coarse_{model_name_clean}_rating.jsonl")

    original_ratings_dict = {}
    if os.path.exists(original_ratings_path):
        with open(original_ratings_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                original_ratings_dict[entry['question_id']] = entry['original_rating']
        print(f"Loaded {len(original_ratings_dict)} original ratings")
    else:
        print(f"Warning: Original ratings file not found: {original_ratings_path}")
        print(f"Run with --rate-original first to generate original ratings")

    # Perturbations directory (at /output/woundcare/ level)
    perturbations_dir = os.path.join(woundcare_output_dir, 'perturbations')

    # Generate ratings for each perturbation
    for pert_type in perturbations:
        print(f"\n{'='*80}")
        print(f"GENERATING RATINGS: {pert_type}")
        print(f"{'='*80}")

        # Create perturbation-specific directory
        pert_dir = os.path.join(output_dir, pert_type)
        os.makedirs(pert_dir, exist_ok=True)

        # Paths
        perturbation_path = os.path.join(perturbations_dir, f"{pert_type}_coarse.jsonl")
        rating_path = os.path.join(pert_dir, f"{pert_type}_coarse_{model_name_clean}_rating.jsonl")

        if not os.path.exists(perturbation_path):
            print(f"  ✗ Perturbation file not found: {perturbation_path}")
            print(f"    Run with --perturb-only first to generate perturbations")
            continue

        # Load perturbations
        perturbations_data = []
        with open(perturbation_path, 'r') as f:
            for line in f:
                perturbations_data.append(json.loads(line))
        print(f"Loaded {len(perturbations_data)} perturbations")

        # Get already processed IDs
        processed_ids = get_processed_ids(rating_path)
        print(f"Already processed: {len(processed_ids)} ratings")

        successful = 0
        skipped_already_done = 0
        failed = 0

        for pert_data in perturbations_data:
            question_id = pert_data['question_id']

            if question_id in processed_ids:
                skipped_already_done += 1
                continue

            try:
                # Get original QA pair for reference responses and metadata
                qa_pair = qa_pairs_dict.get(question_id)
                if not qa_pair:
                    print(f"  ✗ {question_id}: not found in original data")
                    failed += 1
                    continue

                print(f"\n{question_id}:")
                pert_meta = pert_data['perturbation_metadata']
                print(f"  Changed: '{pert_meta['original_term']}' → '{pert_meta['new_term']}'")

                # Check if original rating exists
                original_rating = original_ratings_dict.get(question_id)
                if not original_rating:
                    print(f"  ✗ {question_id}: no original rating found")
                    failed += 1
                    continue

                # Get averaged rating for perturbed answer
                perturbed_answer = pert_data['perturbed_answer']
                perturbed_rating = get_woundcare_rating_with_averaging(
                    qa_pair=qa_pair,
                    candidate_response=perturbed_answer,
                    model=model,
                    num_runs=num_runs
                )

                # Save rating data with both original and perturbed ratings
                rating_data = {
                    'question_id': question_id,
                    'perturbation_type': pert_type,
                    'perturbation_metadata': pert_data.get('perturbation_metadata', {}),
                    'original_rating': original_rating,
                    'perturbed_rating': perturbed_rating,
                    'judge_model': model,
                    'split': qa_pair.get('split', 'unknown')
                }

                with open(rating_path, 'a') as f:
                    f.write(json.dumps(rating_data) + '\n')

                successful += 1
                orig_score = original_rating.get('avg_rating') if isinstance(original_rating, dict) else original_rating
                pert_score = perturbed_rating['avg_rating']
                print(f"  ✓ Original: {orig_score} → Perturbed: {pert_score} ({perturbed_rating['num_valid']}/{perturbed_rating['num_runs']} valid)")

            except Exception as e:
                print(f"  ✗ Error processing {question_id}: {e}")
                failed += 1
                continue

        print(f"\n{'-'*80}")
        print(f"RATING SUMMARY: {pert_type}")
        print(f"{'-'*80}")
        print(f"  Successful:            {successful}")
        print(f"  Skipped (already done): {skipped_already_done}")
        print(f"  Failed:                {failed}")
        print(f"  Rating file: {rating_path}")
        print(f"{'-'*80}")


def run_woundcare_perturbation_experiment(
    data_path: str,
    output_dir: str,
    model: str,
    perturbations: List[str] = None,
    seed: int = 42,
    num_runs: int = 5,
    perturb_only: bool = False,
    rate_original_only: bool = False,
    rate_perturbed_only: bool = False
):
    """
    Run WoundCare perturbation experiment with 3-phase pipeline.

    Args:
        data_path: Path to woundcare_gpt4o_coarse.jsonl
        output_dir: Output directory
        model: LLM judge model
        perturbations: List of perturbation types
        seed: Random seed
        num_runs: Number of runs for averaging
        perturb_only: Only generate perturbations
        rate_original_only: Only generate original ratings
        rate_perturbed_only: Only generate perturbed ratings
    """
    # Default perturbations (only high-coverage types)
    if perturbations is None:
        perturbations = [
            'swap_infection',
            'swap_time_frequency'
        ]

    # Phase 1: Generate perturbations
    if not rate_original_only and not rate_perturbed_only:
        print("\n" + "="*80)
        print("PHASE 1: GENERATING PERTURBATIONS")
        print("="*80)
        generate_perturbations(data_path, output_dir, perturbations, seed)

    # Phase 2: Generate original ratings
    if not perturb_only and not rate_perturbed_only:
        print("\n" + "="*80)
        print("PHASE 2: GENERATING ORIGINAL RATINGS")
        print("="*80)
        generate_original_ratings(data_path, output_dir, model, perturbations, num_runs)

    # Phase 3: Generate perturbed ratings
    if not perturb_only and not rate_original_only:
        print("\n" + "="*80)
        print("PHASE 3: GENERATING PERTURBED RATINGS")
        print("="*80)
        generate_perturbed_ratings(data_path, output_dir, model, perturbations, num_runs)
def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Run WoundCare perturbation experiments')
    parser.add_argument('--data', default='/Users/Federica_1/Documents/GitHub/m3-eval/data/woundcare_gpt4o_coarse.jsonl',
                       help='Path to WoundCare data (combined test+valid)')
    parser.add_argument('--output-dir', default='/Users/Federica_1/Documents/GitHub/m3-eval/output/woundcare/experiment_results/baseline',
                       help='Output directory for rating files')
    parser.add_argument('--model', default='gpt-4.1', help='LLM judge model')
    parser.add_argument('--perturbation', help='Single perturbation to run (default: all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs for averaging')
    parser.add_argument('--test', action='store_true', help='Test mode: only process 3 examples')
    parser.add_argument('--perturb-only', action='store_true', help='Only generate perturbations (no ratings)')
    parser.add_argument('--rate-original', action='store_true', help='Only generate original ratings (requires perturbations to exist)')
    parser.add_argument('--rate-perturbed', action='store_true', help='Only generate perturbed ratings (requires perturbations and original ratings)')

    args = parser.parse_args()

    # Perturbations
    if args.perturbation:
        perturbations = [args.perturbation]
    else:
        perturbations = [
            'swap_infection',
            'swap_time_frequency'
        ]

    # Test mode: create temp file with 3 examples
    data_path = args.data
    if args.test:
        print("\n" + "="*80)
        print("TEST MODE: Processing only first 3 examples")
        print("="*80 + "\n")

        import tempfile
        temp_path = tempfile.mktemp(suffix='.jsonl')

        with open(args.data, 'r') as fin:
            with open(temp_path, 'w') as fout:
                for i, line in enumerate(fin):
                    if i >= 3:
                        break
                    fout.write(line)

        data_path = temp_path

    # Run experiment
    run_woundcare_perturbation_experiment(
        data_path=data_path,
        output_dir=args.output_dir,
        model=args.model,
        perturbations=perturbations,
        seed=args.seed,
        num_runs=args.num_runs,
        perturb_only=args.perturb_only,
        rate_original_only=args.rate_original,
        rate_perturbed_only=args.rate_perturbed
    )

    # Clean up temp file
    if args.test and os.path.exists(data_path):
        os.remove(data_path)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
