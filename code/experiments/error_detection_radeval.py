"""
Error Detection Experiment for RadEval Dataset.

Ask models to detect errors in perturbed radiology reports without being told there is an error.

Tests if the model can identify:
- False predictions (incorrect findings)
- Contradictions (conflicting statements)
- False negations (missing findings)
- And other error types from RexErr
"""

import json
import os
import random
import re
import time
import base64
from pathlib import Path
from typing import Dict, List

from helpers.radeval_experiment_utils import (
    setup_radeval_paths,
    load_radeval_data,
    get_processed_ids,
    save_result,
    clean_model_name
)
from helpers.multi_llm_inference import get_response, get_provider_from_model


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_images_for_report(images_path_str: str, project_root: str) -> List[str]:
    """
    Load images for a report and return as base64-encoded strings.

    Args:
        images_path_str: Comma-separated image paths from dataset
        project_root: Project root directory

    Returns:
        List of base64-encoded image strings
    """
    if not images_path_str:
        return []

    # Split comma-separated paths
    image_paths = [p.strip() for p in images_path_str.split(',')]
    encoded_images = []

    for img_path in image_paths:
        # Convert to absolute path - images downloaded to data/old/physionet.org/
        # Image path format: mimic-cxr-images-512/files/p13/p13291370/s50971742/xxx.jpg
        # We need to point to: data/old/physionet.org/mimic-cxr-jpg/2.1.0/files/p13/...

        # Extract the path after mimic-cxr-images-512/
        if img_path.startswith('mimic-cxr-images-512/'):
            relative_path = img_path.replace('mimic-cxr-images-512/', '')
        else:
            relative_path = img_path

        full_path = os.path.join(project_root, 'data', 'old', 'physionet.org', 'mimic-cxr-jpg', '2.1.0', relative_path)

        if os.path.exists(full_path):
            try:
                encoded_images.append(encode_image_to_base64(full_path))
            except Exception as e:
                print(f"  Warning: Failed to load image {img_path}: {e}")
        else:
            print(f"  Warning: Image not found: {full_path}")

    return encoded_images


def get_detection_prompt(candidate_report: str, encoded_images: List[str], level: str = 'coarse'):
    """
    Build detection prompt for radiology report with images.

    Args:
        candidate_report: Report to evaluate for clinical errors
        encoded_images: List of base64-encoded images
        level: 'coarse' (full report) or 'fine' (sentence-level)

    Returns:
        (system_prompt, user_content) where user_content is list for vision models
    """
    # Load system prompt from file
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_file = f'error_detection_{level}_system.txt'
    prompt_path = os.path.join(script_dir, 'prompts', prompt_file)

    with open(prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    text_prompt = f"""REPORT TO EVALUATE:
{candidate_report}

Does this report contain any clinical errors, factual inaccuracies, or logical inconsistencies? If yes, identify ALL incorrect sentences.

Provide your response in JSON format with "detected", "explanation", and "errors" array (containing all errors with sentence_index, error_type, incorrect_sentence, corrected_sentence, severity, and harm_potential for each)."""

    # Build content array with images + text for vision models
    user_content = []

    # Add all images first
    for img_b64 in encoded_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    # Add text prompt
    user_content.append({
        "type": "text",
        "text": text_prompt
    })

    return system_prompt, user_content


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
    encoded_images: List[str],
    model: str,
    level: str = 'coarse',
    max_retries: int = 3
) -> Dict:
    """
    Get error detection result for a report with images.

    Args:
        candidate_report: Report to evaluate for clinical errors
        encoded_images: List of base64-encoded images
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
            "corrected_sentence": str or None,
            "severity": str or None,
            "harm_potential": str or None
        }
    """
    system_prompt, user_content = get_detection_prompt(candidate_report, encoded_images, level)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
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


def run_error_detection_radeval(args):
    """Run error detection experiment on RadEval dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'radeval_expert_dataset.jsonl')

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'radeval')
    else:
        output_dir = args.output_dir

    # Setup paths
    paths = setup_radeval_paths(output_dir, data_path)
    output_dir = paths['output_dir']

    print(f"\nDataset: RadEval")
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

    text_field = 'prediction'
    reference_field = 'reference'

    # Define perturbations to test (LLM-based only)
    perturbations = ['inject_false_prediction', 'inject_contradiction', 'inject_false_negation']

    if args.perturbation:
        if args.perturbation not in perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}. Choose from: {perturbations}")
        perturbations_to_run = [args.perturbation]
    else:
        perturbations_to_run = perturbations

    print(f"\n{'='*80}")
    print("ERROR DETECTION EXPERIMENT - RADEVAL")
    print(f"{'='*80}")
    print(f"Perturbations: {', '.join(perturbations_to_run)}")
    print("Testing error detection on both ORIGINAL and PERTURBED reports:")
    print("  1. Original reports (control for false positives)")
    print("  2. Perturbed reports (test error detection)")

    # Create experiment directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_detection')
    os.makedirs(experiment_dir, exist_ok=True)

    model_name_clean = clean_model_name(args.model)

    # Determine level
    level = args.level if hasattr(args, 'level') else 'coarse'

    # Process each perturbation
    baseline_dir = os.path.join(output_dir, 'experiment_results', 'baseline')

    # First, process original reports ONCE (same across all perturbations)
    print(f"\n{'='*80}")
    print("PROCESSING ORIGINAL REPORTS (ONCE FOR ALL PERTURBATIONS)")
    print(f"{'='*80}")

    # Load original reports from the first perturbation's baseline
    first_perturbation = perturbations_to_run[0]
    first_perturbation_dir = os.path.join(baseline_dir, first_perturbation)
    baseline_files = [f for f in os.listdir(first_perturbation_dir)
                     if f.endswith('_green_rating.jsonl')]

    if baseline_files:
        baseline_file = os.path.join(first_perturbation_dir, baseline_files[0])

        # Load original data
        original_data = []
        with open(baseline_file, 'r') as f:
            for line in f:
                if line.strip():
                    original_data.append(json.loads(line))

        print(f"Loaded {len(original_data)} original examples")

        # Create original ratings directory
        original_ratings_dir = os.path.join(output_dir, 'original_ratings')
        os.makedirs(original_ratings_dir, exist_ok=True)

        output_filename = f"original_{model_name_clean}_error_detection.jsonl"
        output_path = os.path.join(original_ratings_dir, output_filename)

        # Check which entries have already been processed
        processed_ids = get_processed_ids(output_path)
        remaining_data = [item for item in original_data if item['id'] not in processed_ids]

        if len(remaining_data) == 0:
            print(f"✓ All {len(original_data)} original reports already processed")
        else:
            print(f"Processing {len(remaining_data)} remaining original reports")

            for idx, item in enumerate(remaining_data, 1):
                original_text = item[text_field]  # Original (no errors)
                item_id = item['id']

                # Load images for this report
                images_path_str = item.get('images_path', '')
                encoded_images = load_images_for_report(images_path_str, project_root)

                print(f"  [{idx}/{len(remaining_data)}] {item_id} ({len(encoded_images)} images)...", end=" ")

                start_time = time.time()

                # Get detection result on ORIGINAL text (should find no/few errors)
                detection_result = get_detection_result(
                    original_text,
                    encoded_images,
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

            print(f"✓ Completed original reports")
            print(f"✓ Results saved to: {output_path}")

    # Now process perturbed reports for each perturbation
    print(f"\n{'='*80}")
    print("PROCESSING PERTURBED REPORTS (PER PERTURBATION)")
    print(f"{'='*80}")

    for perturbation_name in perturbations_to_run:
        print(f"\n{'='*80}")
        print(f"PERTURBATION: {perturbation_name.upper()}")
        print(f"{'='*80}")

        # Load pre-generated perturbations from baseline
        perturbation_baseline_dir = os.path.join(baseline_dir, perturbation_name)

        # Find the baseline file with perturbations
        baseline_files = [f for f in os.listdir(perturbation_baseline_dir)
                         if f.endswith('_green_rating.jsonl')]
        if not baseline_files:
            print(f"  Warning: No baseline files found for {perturbation_name}")
            continue

        baseline_file = os.path.join(perturbation_baseline_dir, baseline_files[0])
        print(f"  Loading perturbations from: {baseline_files[0]}")

        # Load perturbed data
        perturbed_data = []
        with open(baseline_file, 'r') as f:
            for line in f:
                if line.strip():
                    perturbed_data.append(json.loads(line))

        print(f"  Loaded {len(perturbed_data)} perturbed examples")

        # Create perturbation-specific subdirectory (matching baseline structure)
        perturbation_output_dir = os.path.join(experiment_dir, perturbation_name)
        os.makedirs(perturbation_output_dir, exist_ok=True)

        print(f"\n  Processing PERTURBED reports")

        output_filename = f"{perturbation_name}_{model_name_clean}_error_detection.jsonl"
        output_path = os.path.join(perturbation_output_dir, output_filename)

        # Check which entries have already been processed
        processed_ids = get_processed_ids(output_path)
        remaining_data = [item for item in perturbed_data if item['id'] not in processed_ids]

        if len(remaining_data) == 0:
            print(f"    ✓ All {len(perturbed_data)} entries already processed")
        else:
            print(f"    Processing {len(remaining_data)} remaining entries")

            # Process each entry
            for idx, item in enumerate(remaining_data, 1):
                perturbed_text = item[f'perturbed_{text_field}']
                item_id = item['id']

                # Load images for this report
                images_path_str = item.get('images_path', '')
                encoded_images = load_images_for_report(images_path_str, project_root)

                print(f"    [{idx}/{len(remaining_data)}] {item_id} ({len(encoded_images)} images)...", end=" ")

                start_time = time.time()

                # Get detection result
                detection_result = get_detection_result(
                    perturbed_text,
                    encoded_images,
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
    print("\nNext step: Analyze detection accuracy and error type classification")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run error detection experiment on RadEval')

    parser.add_argument('--model', type=str, required=True,
                       help='Model to use for error detection (e.g., gpt-4.1-2025-04-14)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/radeval)')

    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['inject_false_prediction', 'inject_contradiction', 'inject_false_negation'],
                       help='Specific perturbation to test (default: all three)')

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

    run_error_detection_radeval(args)
