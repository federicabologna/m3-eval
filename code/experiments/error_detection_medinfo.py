"""
Error Detection Experiment for MedInfo Dataset (Reference-Free).

Ask models to detect medication errors in perturbed answers based on medical knowledge alone,
without comparing to the original answer.

Tests if models can identify:
- Critical medication errors (fatal overdoses, dangerous dosing)
- Non-critical errors (suboptimal but safe dosing)
- Dosage, administration, safety, and clinical accuracy issues
"""

import json
import os
import random
import re
import time
from typing import Dict

from helpers.experiment_utils import (
    setup_paths,
    load_qa_data,
    get_processed_ids,
    save_result,
    clean_model_name,
    get_id_key
)
from helpers.multi_llm_inference import get_response, get_provider_from_model


def get_detection_prompt(question: str, answer: str, level: str = 'coarse') -> tuple:
    """
    Build detection prompt for medication Q&A.

    Args:
        question: Medical question
        answer: Answer to evaluate (full answer or sentence)
        level: 'coarse' (answer-level) or 'fine' (sentence-level)

    Returns:
        (system_prompt, user_prompt)
    """
    # Load system prompt from file
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_file = f'error_detection_medinfo_{level}_system.txt'
    prompt_path = os.path.join(script_dir, 'prompts', prompt_file)

    with open(prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    text_type = 'sentence' if level == 'fine' else 'answer'

    user_prompt = f"""QUESTION:
{question}

{text_type.upper()} TO EVALUATE:
{answer}

Does this {text_type} contain any medication errors, dosage inaccuracies, safety violations, or clinically harmful information? If yes, identify ALL errors.

Provide your response in JSON format with "detected", "explanation", and "errors" array (containing all errors with sentence_index, error_type, incorrect_sentence, corrected_sentence, severity, and harm_potential for each)."""

    return system_prompt, user_prompt


def extract_detection_response(response: str) -> Dict:
    """Extract detection result from model response."""
    # Try to find JSON object in the response (handle nested arrays)
    json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            # Normalize detected field to yes/no
            if 'detected' in result:
                detected_value = str(result['detected']).lower()
                result['detected'] = 'yes' if detected_value in ['yes', 'true', '1'] else 'no'

            # Ensure errors array exists
            if 'errors' not in result:
                result['errors'] = []

            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return {"raw_response": response, "error": "Failed to parse JSON"}
    else:
        print("No JSON found in response")
        return {"raw_response": response, "error": "No JSON found"}


def get_detection_result(
    question: str,
    answer: str,
    model: str,
    level: str = 'coarse',
    max_retries: int = 3
) -> Dict:
    """
    Get error detection result for an answer.

    Args:
        question: Medical question
        answer: Answer to evaluate for errors
        model: Model name
        level: 'coarse' or 'fine'
        max_retries: Max retry attempts

    Returns:
        {
            "detected": "yes" or "no",
            "explanation": str,
            "errors": [...]
        }
    """
    system_prompt, user_prompt = get_detection_prompt(question, answer, level)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        response = get_response(messages, model=model)
        result = extract_detection_response(response)

        # Check if we got a valid result
        if "detected" in result and "explanation" in result:
            return result
        else:
            print(f"Attempt {attempt + 1}/{max_retries}: Invalid detection response. Retrying...")

    # If all retries failed
    print(f"Failed to get valid detection result after {max_retries} attempts")
    return {"error": "Failed to get valid detection result", "last_response": result}


def run_error_detection_medinfo(args):
    """Run error detection experiment on MedInfo dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    paths = setup_paths(args.output_dir, dataset='medinfo')
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Define perturbations to test (LLM-based medication errors)
    perturbations = ['inject_critical_error', 'inject_noncritical_error']

    if args.perturbation:
        if args.perturbation not in perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}. Choose from: {perturbations}")
        perturbations_to_run = [args.perturbation]
    else:
        perturbations_to_run = perturbations

    print(f"\n{'='*80}")
    print("ERROR DETECTION EXPERIMENT - MEDINFO (REFERENCE-FREE)")
    print(f"{'='*80}")
    print(f"Perturbations: {', '.join(perturbations_to_run)}")
    print("Testing error detection based on medical knowledge alone")

    # Create experiment directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_detection')
    os.makedirs(experiment_dir, exist_ok=True)

    # Determine which levels to process
    levels = ['coarse', 'fine'] if args.level == 'both' else [args.level]

    for level in levels:
        print(f"\n{'='*80}")
        print(f"PROCESSING LEVEL: {level.upper()}")
        print(f"{'='*80}")

        # Load full QA dataset
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']
        all_qa_data = load_qa_data(data_path)
        print(f"Loaded {len(all_qa_data)} QA pairs")

        # Create lookup dict
        id_key = get_id_key(all_qa_data)
        qa_lookup = {qa[id_key]: qa for qa in all_qa_data}

        # First, process original answers ONCE (control for false positives)
        print(f"\n{'='*80}")
        print("PROCESSING ORIGINAL ANSWERS (ONCE)")
        print(f"{'='*80}")

        # Create original ratings directory
        original_dir = os.path.join(output_dir, 'original_ratings')
        os.makedirs(original_dir, exist_ok=True)

        output_filename = f"original_{level}_{model_name_clean}_error_detection.jsonl"
        output_path = os.path.join(original_dir, output_filename)

        # Check which entries have already been processed
        processed_ids = get_processed_ids(output_path)
        remaining_data = [item for item in all_qa_data if item[id_key] not in processed_ids]

        if len(remaining_data) == 0:
            print(f"✓ All {len(all_qa_data)} original answers already processed")
        else:
            print(f"Processing {len(remaining_data)} remaining original answers")

            for idx, item in enumerate(remaining_data, 1):
                question = item['question']
                original_answer = item['answer']
                item_id = item[id_key]

                print(f"  [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

                start_time = time.time()

                # Get detection result on ORIGINAL answer (should find no/few errors)
                detection_result = get_detection_result(
                    question, original_answer,
                    model=args.model,
                    level=level,
                    max_retries=3
                )

                elapsed_time = time.time() - start_time
                detected_status = detection_result.get('detected', 'error')
                print(f"{detected_status} ({elapsed_time:.1f}s)")

                # Build result
                result = item.copy()
                result['detection_result'] = detection_result
                result['answer_type'] = 'original'
                result['detection_level'] = level
                result['detection_model'] = args.model
                result['random_seed'] = args.seed

                # Save to file
                save_result(output_path, result)

            print(f"\n✓ Completed original answers")
            print(f"✓ Results saved to: {output_path}")

        # Now process perturbed answers for each perturbation
        print(f"\n{'='*80}")
        print("PROCESSING PERTURBED ANSWERS (PER PERTURBATION)")
        print(f"{'='*80}")

        for perturbation_name in perturbations_to_run:
            print(f"\n{'='*80}")
            print(f"PERTURBATION: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Load perturbations from file
            perturbations_dir = os.path.join(output_dir, 'perturbations')
            pert_filename = f"{perturbation_name}_{level}.jsonl"
            pert_filepath = os.path.join(perturbations_dir, pert_filename)

            if not os.path.exists(pert_filepath):
                print(f"  Perturbation file not found: {pert_filename}")
                print(f"  Please run baseline experiment first to generate perturbations")
                continue

            # Load perturbations
            perturbed_data = []
            with open(pert_filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Only include successful perturbations
                        if 'skip_reason' not in entry and 'perturbed_answer' in entry:
                            perturbed_data.append(entry)

            print(f"  Loaded {len(perturbed_data)} perturbed examples")

            # Create perturbation-specific subdirectory
            perturbation_output_dir = os.path.join(experiment_dir, perturbation_name)
            os.makedirs(perturbation_output_dir, exist_ok=True)

            output_filename = f"{perturbation_name}_{level}_{model_name_clean}_error_detection.jsonl"
            output_path = os.path.join(perturbation_output_dir, output_filename)

            # Check which entries have already been processed
            processed_ids = get_processed_ids(output_path)
            remaining_data = [item for item in perturbed_data if item[id_key] not in processed_ids]

            if len(remaining_data) == 0:
                print(f"  ✓ All {len(perturbed_data)} entries already processed")
            else:
                print(f"  Processing {len(remaining_data)} remaining entries")

                # Process each entry
                for idx, item in enumerate(remaining_data, 1):
                    question = item['question']
                    perturbed_answer = item['perturbed_answer']
                    item_id = item[id_key]

                    print(f"    [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

                    start_time = time.time()

                    # Get detection result
                    detection_result = get_detection_result(
                        question, perturbed_answer,
                        model=args.model,
                        level=level,
                        max_retries=3
                    )

                    elapsed_time = time.time() - start_time
                    detected_status = detection_result.get('detected', 'error')
                    print(f"{detected_status} ({elapsed_time:.1f}s)")

                    # Build result
                    result = item.copy()
                    result['detection_result'] = detection_result
                    result['detection_level'] = level
                    result['detection_model'] = args.model
                    result['random_seed'] = args.seed

                    # Save to file
                    save_result(output_path, result)

                print(f"    ✓ Completed detection for {perturbation_name}")

    print(f"\n{'='*80}")
    print("ERROR DETECTION EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {experiment_dir}/{{perturbation}}/")
    print("\nNext step: Analyze detection accuracy and error classification")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run error detection experiment on MedInfo (reference-free)')

    parser.add_argument('--model', type=str, required=True,
                       help='Model to use for error detection (e.g., gpt-4.1-2025-04-14)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/medinfo)')

    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['inject_critical_error', 'inject_noncritical_error'],
                       help='Specific perturbation to test (default: both)')

    parser.add_argument('--level', type=str, default='coarse',
                       choices=['coarse', 'fine', 'both'],
                       help='Detection level: coarse (answer-level) or fine (sentence-level) or both')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--start-idx', type=int, default=None,
                       help='Start index for data subset (default: 0)')

    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for data subset (default: all data)')

    args = parser.parse_args()

    run_error_detection_medinfo(args)
