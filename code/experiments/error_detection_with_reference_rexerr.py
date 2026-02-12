"""
Error Detection Experiment for RexErr Dataset.

Ask models to detect errors in perturbed radiology reports without being told there is an error.

RexErr already contains pre-generated perturbations (prediction field has errors).
We test if models can detect these errors compared to the reference (original correct) report.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict

from helpers.radeval_experiment_utils import (
    load_radeval_data,
    get_processed_ids,
    save_result,
    clean_model_name
)
from helpers.multi_llm_inference import get_response, get_provider_from_model


def get_detection_prompt(candidate_report: str, reference_report: str, level: str = 'coarse') -> tuple:
    """
    Build detection prompt for radiology report.

    Args:
        candidate_report: Perturbed report to evaluate
        reference_report: Original correct report
        level: 'coarse' (full report) or 'fine' (sentence-level)

    Returns:
        (system_prompt, user_prompt)
    """
    # Load system prompt from file
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_file = f'error_detection_with_reference_{level}_system.txt'
    prompt_path = os.path.join(script_dir, 'prompts', prompt_file)

    with open(prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    user_prompt = f"""REFERENCE REPORT (Correct):
{reference_report}

CANDIDATE REPORT (To Evaluate):
{candidate_report}

Does the candidate report contain any errors compared to the reference report? If yes, identify ALL incorrect sentences.

Provide your response in JSON format with "detected", "explanation", and "errors" array (containing all errors with sentence_index, error_type, incorrect_sentence, and corrected_sentence for each)."""

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
    candidate_report: str,
    reference_report: str,
    model: str,
    level: str = 'coarse',
    max_retries: int = 3
) -> Dict:
    """
    Get error detection result for a report.

    Args:
        candidate_report: Perturbed report to evaluate
        reference_report: Original correct report
        model: Model name
        level: 'coarse' or 'fine'
        max_retries: Max retry attempts

    Returns:
        {
            "detected": "yes" or "no",
            "error_type": str or None,
            "explanation": str,
            "sentence_index": int or None,
            "incorrect_sentence": str or None,
            "corrected_sentence": str or None
        }
    """
    system_prompt, user_prompt = get_detection_prompt(candidate_report, reference_report, level)

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


def run_error_detection_rexerr(args):
    """Run error detection experiment on RexErr dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'rexerr_acceptable_dataset.jsonl')

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'rexerr')
    else:
        output_dir = args.output_dir

    print(f"\nDataset: RexErr")
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

    # RexErr fields
    text_field = 'prediction'  # Perturbed report (with errors)
    reference_field = 'reference'  # Original report (ground truth)

    print(f"\n{'='*80}")
    print("ERROR DETECTION EXPERIMENT - REXERR")
    print(f"{'='*80}")
    print("Testing error detection on both ORIGINAL and PERTURBED reports:")
    print("  1. Original reports (control for false positives)")
    print("  2. Perturbed reports (test error detection)")
    print("  - prediction: error-injected report (from RexErr)")
    print("  - reference: original correct report")

    # Create experiment directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_detection_with_reference')
    os.makedirs(experiment_dir, exist_ok=True)

    model_name_clean = clean_model_name(args.model)

    # Determine level
    level = args.level if hasattr(args, 'level') else 'coarse'

    # First, process original reports ONCE
    print(f"\n{'='*80}")
    print("PROCESSING ORIGINAL REPORTS (ONCE)")
    print(f"{'='*80}")

    # Create original ratings directory
    original_ratings_dir = os.path.join(output_dir, 'original_ratings')
    os.makedirs(original_ratings_dir, exist_ok=True)

    output_filename = f"original_{model_name_clean}_error_detection_with_reference.jsonl"
    output_path = os.path.join(original_ratings_dir, output_filename)

    # Check which entries have already been processed
    processed_ids = get_processed_ids(output_path)
    remaining_data = [item for item in data if item['id'] not in processed_ids]

    if len(remaining_data) == 0:
        print(f"✓ All {len(data)} original reports already processed")
    else:
        print(f"Processing {len(remaining_data)} remaining original reports")

        for idx, item in enumerate(remaining_data, 1):
            reference = item[reference_field]
            original_text = reference  # RexErr: reference is the original
            item_id = item['id']

            print(f"  [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

            start_time = time.time()

            # Get detection result on ORIGINAL text (should find no/few errors)
            detection_result = get_detection_result(
                original_text, reference,
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
            result['report_type'] = 'original'
            result['detection_level'] = level
            result['detection_model'] = args.model
            result['random_seed'] = args.seed

            # Save to file
            save_result(output_path, result)

        print(f"\n✓ Completed original reports")
        print(f"✓ Results saved to: {output_path}")

    # Now process perturbed reports
    print(f"\n{'='*80}")
    print("PROCESSING PERTURBED REPORTS")
    print(f"{'='*80}")

    output_filename = f"{model_name_clean}_error_detection_with_reference.jsonl"
    output_path = os.path.join(experiment_dir, output_filename)

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

            # Get detection result
            detection_result = get_detection_result(
                perturbed_text, reference,
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

        print(f"\n✓ Completed processing {len(remaining_data)} entries")
        print(f"✓ Results saved to: {output_path}")

    print(f"\n{'='*80}")
    print("ERROR DETECTION EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print("\nNext step: Analyze detection accuracy and error type classification")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run error detection experiment on RexErr')

    parser.add_argument('--model', type=str, required=True,
                       help='Model to use for error detection (e.g., gpt-4.1-2025-04-14)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/rexerr)')

    parser.add_argument('--level', type=str, default='coarse',
                       choices=['coarse', 'fine'],
                       help='Detection level: coarse (full report) or fine (sentence-level)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--start-idx', type=int, default=None,
                       help='Start index for data subset (default: 0)')

    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for data subset (default: all data)')

    args = parser.parse_args()

    run_error_detection_rexerr(args)
