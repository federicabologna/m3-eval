#!/usr/bin/env python3
import os
import sys

# Print diagnostic info BEFORE any torch imports
print("=" * 80, file=sys.stderr)
print("CUDA/PyTorch Diagnostics (before import):", file=sys.stderr)
print(f"Python: {sys.version}", file=sys.stderr)
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}", file=sys.stderr)
print("=" * 80, file=sys.stderr)

# Try to check nvidia-smi
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print("nvidia-smi output:", file=sys.stderr)
    for line in result.stdout.split('\n'):
        if 'CUDA Version' in line:
            print(line, file=sys.stderr)
except Exception as e:
    print(f"Could not run nvidia-smi: {e}", file=sys.stderr)

print("=" * 80, file=sys.stderr)
print("Attempting to import torch...", file=sys.stderr)

import json
import random
import re
import time
from dotenv import load_dotenv
from helpers.multi_llm_inference import get_response
from helpers.perturbation_functions import (
    add_confusion,
    add_typos,
    change_dosage,
    remove_sentences_by_percentage
)
from helpers.experiment_utils import (
    load_qa_data,
    get_processed_ids,
    get_or_create_original_ratings,
    clean_model_name,
    get_id_key
)

# Load environment variables from .env file
load_dotenv()


def load_prompt(prompt_path):
    """Load and split the prompt into system and user parts."""
    with open(prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    
    user_template = '''QUESTION
{question}


ANSWER
{answer}


YOUR SCORES'''

    return system_prompt, user_template


def normalize_rating_keys(rating_dict):
    """Normalize rating keys to simpler names."""
    # Mapping from long keys to short keys
    key_mapping = {
        # Coarse keys
        "The answer aligns with current medical knowledge": "correctness",
        "The answer addresses the specific medical question": "relevance",
        "The answer communicates contraindications or risks": "safety",
        # Fine keys
        "The sentence aligns with current medical knowledge": "correctness",
        "The sentence addresses the specific medical question": "relevance",
        "The sentence communicates contraindications or risks": "safety"
    }

    normalized = {}
    for key, value in rating_dict.items():
        # Check if key matches any of the long keys
        normalized_key = key_mapping.get(key, key)
        normalized[normalized_key] = value

    return normalized


def validate_rating(rating_dict):
    """Check if rating has all required keys."""
    required_keys = ['correctness', 'relevance', 'safety']
    return all(key in rating_dict for key in required_keys)


def extract_json_from_response(response):
    """Extract JSON object from model response text."""
    # Try to find JSON object in the response
    # Look for content between curly braces
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            # Parse the JSON
            rating_dict = json.loads(json_str)
            # Normalize keys
            rating_dict = normalize_rating_keys(rating_dict)
            return rating_dict
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Extracted string: {json_str}")
            return {"raw_response": response, "error": "Failed to parse JSON"}
    else:
        print("No JSON found in response")
        return {"raw_response": response, "error": "No JSON found"}


def get_rating(question, answer, system_prompt, user_template, model="Qwen3-1.7B", max_retries=3, flush_output=False):
    """Get rating for an answer using the LLM with retry logic."""
    # Format the user prompt with question and answer
    user_prompt = user_template.replace('{question}', question).replace('{answer}', answer)

    # Create messages in chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        # Get response
        print(f"    Generating response (attempt {attempt + 1}/{max_retries})...", flush=flush_output)
        response = get_response(messages, model=model)
        print(f"    Response received, validating...", flush=flush_output)

        # Extract and validate JSON
        rating = extract_json_from_response(response)

        # Check if we got a valid rating
        if validate_rating(rating):
            return rating
        else:
            print(f"Attempt {attempt + 1}/{max_retries}: Invalid rating keys. Retrying...")
            if "error" not in rating:
                print(f"Got keys: {list(rating.keys())}")

    # If all retries failed, return error with last response
    print(f"Failed to get valid rating after {max_retries} attempts")
    return {"error": "Failed to get valid rating", "last_response": rating}


def average_ratings(ratings_list):
    """Average scores and confidence across multiple rating runs."""
    if not ratings_list:
        return {"error": "No valid ratings to average"}

    # Filter out any ratings with errors
    valid_ratings = [r for r in ratings_list if "error" not in r]

    if not valid_ratings:
        return {"error": "No valid ratings to average", "all_ratings": ratings_list}

    # Initialize averaged rating
    averaged = {}

    # Average each dimension
    for dimension in ['correctness', 'relevance', 'safety']:
        if dimension not in valid_ratings[0]:
            continue

        # Check if ratings have confidence scores
        has_confidence = isinstance(valid_ratings[0][dimension], dict) and 'confidence' in valid_ratings[0][dimension]

        if has_confidence:
            # Average both score and confidence
            avg_score = sum(r[dimension]['score'] for r in valid_ratings) / len(valid_ratings)
            avg_confidence = sum(r[dimension]['confidence'] for r in valid_ratings) / len(valid_ratings)

            # Collect all reasons for reference
            all_reasons = [r[dimension].get('reason', 'N/A') for r in valid_ratings]

            averaged[dimension] = {
                'score': round(avg_score, 2),
                'confidence': round(avg_confidence, 2),
                'reason': all_reasons[0],  # Use first reason as representative
                'all_reasons': all_reasons  # Keep all reasons for reference
            }
        else:
            # Old format - just average scores
            avg_score = sum(r[dimension]['score'] for r in valid_ratings) / len(valid_ratings)
            averaged[dimension] = {
                'score': round(avg_score, 2),
                'reason': valid_ratings[0][dimension].get('reason', 'N/A')
            }

    # Add metadata about the averaging
    averaged['_meta'] = {
        'num_runs': len(ratings_list),
        'num_valid': len(valid_ratings)
    }

    return averaged


def get_rating_with_averaging(question, answer, system_prompt, user_template, model="Qwen3-1.7B", num_runs=5, flush_output=False):
    """Get multiple ratings and return the average."""
    print(f"Collecting {num_runs} ratings to average...")

    all_ratings = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        rating = get_rating(question, answer, system_prompt, user_template, model, flush_output=flush_output)
        all_ratings.append(rating)

    # Average the ratings
    averaged_rating = average_ratings(all_ratings)

    # Store individual ratings for reference
    averaged_rating['individual_ratings'] = all_ratings

    return averaged_rating


def get_perturbed_rating_only(question, perturbed_answer, prompt_path, model="Qwen3-1.7B"):
    """Get rating for perturbed answer only (original rating already computed)."""
    # Load prompt
    system_prompt, user_template = load_prompt(prompt_path)

    print("\nGetting rating for PERTURBED answer...")
    perturbed_rating = get_rating_with_averaging(question, perturbed_answer, system_prompt, user_template, model, num_runs=5)

    return perturbed_rating


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run perturbation pipeline with various LLM models')
    parser.add_argument('--model', type=str, default='Qwen3-8B',
                       help='Model name (default: Qwen3-8B). Examples: gpt-4o, claude-opus-4-20250514, gemini-2.0-flash-exp')
    parser.add_argument('--perturbation', type=str, default=None,
                       help='Specific perturbation to run. Options: add_typos, change_dosage, remove_sentences, add_confusion. If not specified, runs all in order.')
    parser.add_argument('--remove-pct', type=float, default=0.3,
                       help='For remove_sentences perturbation: percentage of sentences to remove (0.0-1.0). Default: 0.3 (30%%)')
    parser.add_argument('--all-remove-pct', action='store_true',
                       help='For remove_sentences: run all percentage values (0.3, 0.5, 0.7) sequentially')
    parser.add_argument('--typo_prob', type=float, default=0.5,
                       help='For add_typos perturbation: probability of applying typo to each medical term (0.0-1.0). Default: 0.5')
    parser.add_argument('--all_typo_prob', action='store_true',
                       help='For add_typos: run all probability values (0.3, 0.5, 0.7) sequentially')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Set random seed for reproducibility of perturbations
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    model = args.model

    # Import the helper to get provider from model name
    from helpers.multi_llm_inference import get_provider_from_model
    provider = get_provider_from_model(model)

    print(f"Using model: {model} (provider: {provider})")

    # Paths - Use relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from /code to project root
    output_dir = os.path.join(project_root, 'output', 'cqa_eval')
    coarse_data_path = os.path.join(project_root, 'data', 'coarse_5pt_expert+llm_consolidated.jsonl')
    fine_data_path = os.path.join(project_root, 'data', 'fine_5pt_expert+llm_consolidated.jsonl')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define perturbation types in order
    all_perturbations = ['add_typos', 'change_dosage', 'remove_sentences', 'add_confusion']

    # Determine which perturbations to run
    if args.perturbation:
        if args.perturbation not in all_perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}. Choose from: {', '.join(all_perturbations)}")
        perturbation_names = [args.perturbation]
        print(f"Running single perturbation: {args.perturbation}")
    else:
        perturbation_names = all_perturbations
        print(f"Running all perturbations in order: {' -> '.join(perturbation_names)}")

    # Process each level independently
    for level in ['coarse', 'fine']:
        print(f"\n{'='*80}")
        print(f"PROCESSING LEVEL: {level.upper()}")
        print(f"{'='*80}")

        # Use coarse or fine data based on level
        data_path = coarse_data_path if level == 'coarse' else fine_data_path
        print(f"Using data: {data_path}")

        # Load data
        qa_pairs = load_qa_data(data_path)

        # Select correct prompt path based on level
        prompt_path = os.path.join(script_dir, 'prompts', f'{level}prompt_system.txt')

        # Determine ID key
        id_key = get_id_key(qa_pairs)

        # Include model name in output filename (clean up slashes and dots)
        model_name_clean = clean_model_name(model)

        # Step 1: Get/compute original ratings
        print(f"\n{'='*80}")
        print("STEP 1: ORIGINAL RATINGS")
        print(f"{'='*80}")

        original_ratings_dict = get_or_create_original_ratings(
            qa_pairs=qa_pairs,
            level=level,
            prompt_path=prompt_path,
            model=model,
            output_dir=output_dir,
            model_name_clean=model_name_clean,
            num_runs=5
        )

        # Step 2: Process each perturbation
        print(f"\n{'='*80}")
        print("STEP 2: PERTURBATIONS")
        print(f"{'='*80}")

        for perturbation_name in perturbation_names:
            print(f"\n{'='*80}")
            print(f"Processing perturbation: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Create perturbation-specific subdirectory under baseline experiment
            baseline_dir = os.path.join(output_dir, 'experiment_results', 'baseline')
            perturbation_dir = os.path.join(baseline_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Determine which remove_pct values to run for remove_sentences
            remove_pct_values = [args.remove_pct]
            if perturbation_name == 'remove_sentences' and args.all_remove_pct:
                remove_pct_values = [0.3, 0.5, 0.7]
                print(f"Running all remove_pct values: {remove_pct_values}")

            # Determine which typo_prob values to run for add_typos
            typo_prob_values = [args.typo_prob]
            if perturbation_name == 'add_typos' and args.all_typo_prob:
                typo_prob_values = [0.3, 0.5, 0.7]
                print(f"Running all typo_prob values: {typo_prob_values}")

            # Iterate over parameter combinations
            for remove_pct in remove_pct_values:
                for typo_prob in typo_prob_values:
                    # Print parameter being used
                    if perturbation_name == 'remove_sentences' and len(remove_pct_values) > 1:
                        print(f"\n{'-'*80}")
                        print(f"REMOVE_PCT = {remove_pct}")
                        print(f"{'-'*80}")

                    if perturbation_name == 'add_typos' and len(typo_prob_values) > 1:
                        print(f"\n{'-'*80}")
                        print(f"TYPO_PROB = {typo_prob}")
                        print(f"{'-'*80}")

                    # Determine output filename
                    if perturbation_name == 'remove_sentences':
                        pct_str = str(int(remove_pct * 100))
                        output_filename = f"{perturbation_name}_{pct_str}pct_{level}_{model_name_clean}_rating.jsonl"
                    elif perturbation_name == 'add_typos':
                        prob_str = str(typo_prob).replace('.', '')
                        output_filename = f"{perturbation_name}_{prob_str}prob_{level}_{model_name_clean}_rating.jsonl"
                    else:
                        output_filename = f"{perturbation_name}_{level}_{model_name_clean}_rating.jsonl"

                    output_path = os.path.join(perturbation_dir, output_filename)

                    # Check which answer/sentence IDs have already been processed
                    processed_ids = get_processed_ids(output_path)
                    remaining_qa_pairs = [qa for qa in qa_pairs if qa[id_key] not in processed_ids]

                    if len(remaining_qa_pairs) == 0:
                        print(f"All {len(qa_pairs)} QA pairs already processed. Skipping.")
                        continue

                    print(f"Processing {len(remaining_qa_pairs)} remaining QA pairs (out of {len(qa_pairs)} total)")
                    print(f"Saving results to: experiment_results/baseline/{perturbation_name}/{output_filename}")

                    for qa_pair in remaining_qa_pairs:
                        question = qa_pair['question']
                        original_answer = qa_pair['answer']

                        # Apply perturbation based on type
                        perturbed_answer = None
                        change_counts = None

                        if perturbation_name == 'add_confusion':
                            perturbed_answer = add_confusion(original_answer)
                        elif perturbation_name == 'add_typos':
                            perturbed_answer = add_typos(original_answer, typo_probability=typo_prob)
                        elif perturbation_name == 'change_dosage':
                            perturbed_answer, change_counts = change_dosage(original_answer)
                        elif perturbation_name == 'remove_sentences':
                            perturbed_answer = remove_sentences_by_percentage(original_answer, percentage=remove_pct)

                        if perturbed_answer is None:
                            perturbed_answer = original_answer

                        # Check if perturbation was actually applied
                        if perturbed_answer == original_answer:
                            print(f"Skipping {qa_pair[id_key]} - no perturbation applied (text unchanged)")
                            continue

                        # Get perturbed rating only (original already computed)
                        start_time = time.time()
                        perturbed_rating = get_perturbed_rating_only(
                            question, perturbed_answer, prompt_path, model
                        )
                        elapsed_time = time.time() - start_time
                        print(f'Time taken for {qa_pair[id_key]}: {elapsed_time:.2f} seconds')

                        # Get original rating from dict
                        original_rating = original_ratings_dict.get(qa_pair[id_key])

                        if original_rating is None:
                            print(f"WARNING: No original rating found for {qa_pair[id_key]}, skipping...")
                            continue

                        # Keep original dict and add perturbation info and ratings
                        results = qa_pair.copy()
                        results['perturbation'] = perturbation_name
                        results['perturbed_answer'] = perturbed_answer
                        results['original_rating'] = original_rating
                        results['perturbed_rating'] = perturbed_rating
                        results['random_seed'] = args.seed

                        # Add change counts if available (for change_dosage perturbation)
                        if change_counts is not None:
                            results['change_counts'] = change_counts

                        # Add removal_percentage if this is remove_sentences perturbation
                        if perturbation_name == 'remove_sentences':
                            results['removal_percentage'] = remove_pct

                        # Add typo_probability if this is add_typos perturbation
                        if perturbation_name == 'add_typos':
                            results['typo_probability'] = typo_prob

                        # Save to JSON file (JSONL format - one JSON object per line)
                        with open(output_path, 'a') as f:
                            json.dump(results, f)
                            f.write('\n')


if __name__ == "__main__":
    main()
