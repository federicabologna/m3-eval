import json
import random
import re
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from helpers.multi_llm_inference import get_response
from helpers.perturbation_functions import (
    add_confusion,
    add_typos,
    change_dosage,
    remove_must_have
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


def load_qa_data(data_path):
    """Load QA pairs from JSONL file."""
    qa_pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line))
    return qa_pairs


def get_processed_ids(output_path):
    """Get set of already processed answer IDs from output file."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Use answer_id for coarse/fine, or sentence_id if present
                    id_key = 'sentence_id' if 'sentence_id' in entry else 'answer_id'
                    processed_ids.add(entry[id_key])
            print(f"Found {len(processed_ids)} already processed entries in output file")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    return processed_ids


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


def get_rating(question, answer, system_prompt, user_template, model="Qwen3-1.7B", max_retries=3):
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
        response = get_response(messages, model=model)

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


def get_rating_with_averaging(question, answer, system_prompt, user_template, model="Qwen3-1.7B", num_runs=5):
    """Get multiple ratings and return the average."""
    print(f"Collecting {num_runs} ratings to average...")

    all_ratings = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        rating = get_rating(question, answer, system_prompt, user_template, model)
        all_ratings.append(rating)

    # Average the ratings
    averaged_rating = average_ratings(all_ratings)

    # Store individual ratings for reference
    averaged_rating['individual_ratings'] = all_ratings

    return averaged_rating


def load_original_ratings(original_ratings_path):
    """Load original ratings from file into a dictionary keyed by answer_id or sentence_id."""
    ratings_dict = {}
    if os.path.exists(original_ratings_path):
        try:
            with open(original_ratings_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Use sentence_id if present, otherwise answer_id
                    id_key = 'sentence_id' if 'sentence_id' in entry else 'answer_id'
                    ratings_dict[entry[id_key]] = entry['original_rating']
            print(f"Loaded {len(ratings_dict)} original ratings from file")
        except Exception as e:
            print(f"Warning: Could not read original ratings file: {e}")
    return ratings_dict


def compute_original_ratings(qa_pairs, level, prompt_path, model, original_ratings_path):
    """Compute original ratings for all QA pairs and save to file."""
    print(f"\nComputing original ratings for {len(qa_pairs)} QA pairs...")

    # Load prompt
    system_prompt, user_template = load_prompt(prompt_path)

    # Determine ID key
    id_key = 'sentence_id' if 'sentence_id' in qa_pairs[0] else 'answer_id'

    # Load any existing ratings
    existing_ratings = load_original_ratings(original_ratings_path)

    # Filter to only compute missing ratings
    qa_pairs_to_compute = [qa for qa in qa_pairs if qa[id_key] not in existing_ratings]

    if len(qa_pairs_to_compute) == 0:
        print("All original ratings already computed!")
        return existing_ratings

    print(f"Computing {len(qa_pairs_to_compute)} new original ratings (out of {len(qa_pairs)} total)")

    for qa_pair in qa_pairs_to_compute:
        question = qa_pair['question']
        original_answer = qa_pair['answer']

        print(f"\nGetting rating for {qa_pair[id_key]}...")
        start_time = time.time()
        original_rating = get_rating_with_averaging(question, original_answer, system_prompt, user_template, model, num_runs=5)
        elapsed_time = time.time() - start_time
        print(f'Time taken: {elapsed_time:.2f} seconds')

        # Save immediately to file
        result = qa_pair.copy()
        result['original_rating'] = original_rating

        with open(original_ratings_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')

        # Add to existing_ratings dict
        existing_ratings[qa_pair[id_key]] = original_rating

    print(f"\nOriginal ratings saved to: {original_ratings_path}")
    return existing_ratings


def run_experiment(prompt_name, prompt_path, qa_pair, perturbation_name, perturbed_answer, model="Qwen3-1.7B"):
    """Run experiment with a specific prompt."""

    # Load prompt
    system_prompt, user_template = load_prompt(prompt_path)

    question = qa_pair['question']
    original_answer = qa_pair['answer']

    # Get rating for original answer (now includes retry logic)
    print("\nGetting rating for ORIGINAL answer...")
    original_rating = get_rating(question, original_answer, system_prompt, user_template, model)

    # Get rating for perturbed answer (now includes retry logic)
    print("\nGetting rating for PERTURBED answer...")
    perturbed_rating = get_rating(question, perturbed_answer, system_prompt, user_template, model)

    return {
        "original": original_rating,
        "perturbed": perturbed_rating
    }


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
                       help='Specific perturbation to run. Options: add_typos, change_dosage, remove_must_have, add_confusion. If not specified, runs all in order.')
    parser.add_argument('--num_remove', type=int, default=1,
                       help='For remove_must_have perturbation: number of must_have sentences to remove (1-3). Default: 1')
    parser.add_argument('--all_num_remove', action='store_true',
                       help='For remove_must_have: run all values (1, 2, 3) sequentially')
    parser.add_argument('--typo_prob', type=float, default=0.5,
                       help='For add_typos perturbation: probability of applying typo to each medical term (0.0-1.0). Default: 0.5')
    parser.add_argument('--all_typo_prob', action='store_true',
                       help='For add_typos: run all probability values (0.3, 0.5, 0.7) sequentially')
    args = parser.parse_args()

    model = args.model

    # Import the helper to get provider from model name
    from helpers.multi_llm_inference import get_provider_from_model
    provider = get_provider_from_model(model)

    print(f"Using model: {model} (provider: {provider})")

    # Paths - Use relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from /code to project root
    output_dir = os.path.join(project_root, 'output')
    coarse_data_path = os.path.join(project_root, 'data', 'coarse_5pt_expert+llm_consolidated.jsonl')
    fine_data_path = os.path.join(project_root, 'data', 'fine_5pt_expert+llm_consolidated.jsonl')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define perturbation types in order
    all_perturbations = ['add_typos', 'change_dosage', 'remove_must_have', 'add_confusion']

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
        id_key = 'sentence_id' if 'sentence_id' in qa_pairs[0] else 'answer_id'

        # Include model name in output filename (clean up slashes and dots)
        model_name_clean = model.replace('/', '-').replace('.', '_')

        # Step 1: Compute/load original ratings
        print(f"\n{'='*80}")
        print("STEP 1: ORIGINAL RATINGS")
        print(f"{'='*80}")

        original_ratings_filename = f"original_{level}_{model_name_clean}_rating.jsonl"
        original_ratings_path = os.path.join(output_dir, original_ratings_filename)

        original_ratings_dict = load_original_ratings(original_ratings_path)

        # Compute any missing original ratings
        if len(original_ratings_dict) < len(qa_pairs):
            original_ratings_dict = compute_original_ratings(
                qa_pairs, level, prompt_path, model, original_ratings_path
            )
        else:
            print(f"All {len(qa_pairs)} original ratings already computed!")

        # Step 2: Process each perturbation
        print(f"\n{'='*80}")
        print("STEP 2: PERTURBATIONS")
        print(f"{'='*80}")

        for perturbation_name in perturbation_names:
            print(f"\n{'='*80}")
            print(f"Processing perturbation: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Create perturbation-specific subdirectory
            perturbation_dir = os.path.join(output_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Determine which num_remove values to run for remove_must_have
            num_remove_values = [args.num_remove]
            if perturbation_name == 'remove_must_have' and args.all_num_remove:
                num_remove_values = [1, 2, 3]
                print(f"Running all num_remove values: {num_remove_values}")

            # Determine which typo_prob values to run for add_typos
            typo_prob_values = [args.typo_prob]
            if perturbation_name == 'add_typos' and args.all_typo_prob:
                typo_prob_values = [0.3, 0.5, 0.7]
                print(f"Running all typo_prob values: {typo_prob_values}")

            # Iterate over parameter combinations
            for num_remove in num_remove_values:
                for typo_prob in typo_prob_values:
                    # Print parameter being used
                    if perturbation_name == 'remove_must_have' and len(num_remove_values) > 1:
                        print(f"\n{'-'*80}")
                        print(f"NUM_REMOVE = {num_remove}")
                        print(f"{'-'*80}")

                    if perturbation_name == 'add_typos' and len(typo_prob_values) > 1:
                        print(f"\n{'-'*80}")
                        print(f"TYPO_PROB = {typo_prob}")
                        print(f"{'-'*80}")

                    # Determine output filename
                    if perturbation_name == 'remove_must_have':
                        output_filename = f"{perturbation_name}_{num_remove}removed_{level}_{model_name_clean}_rating.jsonl"
                    elif perturbation_name == 'add_typos':
                        # Format probability as string without decimal point (e.g., 0.3 -> 03, 0.5 -> 05)
                        prob_str = str(int(typo_prob * 10))
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
                    print(f"Saving results to: {perturbation_name}/{output_filename}")

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
                        elif perturbation_name == 'remove_must_have':
                            # Only apply if Must_have exists in qa_pair
                            if 'Must_have' in qa_pair:
                                must_have = qa_pair['Must_have']
                                perturbed_answer = remove_must_have(original_answer, must_have, num_to_remove=num_remove)
                            else:
                                print(f"Skipping remove_must_have for {qa_pair[id_key]} - no Must_have field")
                                continue

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

                        # Add change counts if available (for change_dosage perturbation)
                        if change_counts is not None:
                            results['change_counts'] = change_counts

                        # Add num_remove if this is remove_must_have perturbation
                        if perturbation_name == 'remove_must_have':
                            results['num_removed'] = num_remove

                        # Add typo_probability if this is add_typos perturbation
                        if perturbation_name == 'add_typos':
                            results['typo_probability'] = typo_prob

                        # Save to JSON file (JSONL format - one JSON object per line)
                        with open(output_path, 'a') as f:
                            json.dump(results, f)
                            f.write('\n')


if __name__ == "__main__":
    main()
