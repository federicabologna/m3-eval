"""
Error Criticality Classification Experiment for RadEval Dataset.

Classifies detected errors as clinically critical vs non-critical.
Uses error detection results as input.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List

from helpers.radeval_experiment_utils import (
    get_processed_ids,
    save_result,
    clean_model_name
)
from helpers.multi_llm_inference import get_response, get_provider_from_model


def get_criticality_prompt(error_info: Dict) -> tuple:
    """
    Build criticality classification prompt for a detected error.

    Args:
        error_info: Dictionary containing error details (sentence, type, etc.)

    Returns:
        (system_prompt, user_prompt)
    """
    # Load system prompt from file
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(script_dir, 'prompts', 'error_criticality_system.txt')

    with open(prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    user_prompt = f"""ERROR TO CLASSIFY:

Error Type: {error_info.get('error_type', 'Unknown')}
Incorrect Sentence: {error_info.get('incorrect_sentence', 'N/A')}
Corrected Sentence: {error_info.get('corrected_sentence', 'N/A')}

Is this error clinically CRITICAL or NON-CRITICAL?

Provide your response in JSON format with "severity" ("critical" or "non-critical") and "harm_potential" (explain the potential harm and impact)."""

    return system_prompt, user_prompt


def extract_criticality_response(response: str) -> Dict:
    """Extract criticality classification from model response."""
    # Try to find JSON object in the response
    json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            # Normalize severity field to "critical" or "non-critical"
            if 'severity' in result:
                severity_value = str(result['severity']).lower()
                if 'critical' in severity_value and 'non' not in severity_value:
                    result['severity'] = 'critical'
                else:
                    result['severity'] = 'non-critical'
            # Also handle old "critical" field for backward compatibility
            elif 'critical' in result:
                critical_value = str(result['critical']).lower()
                result['severity'] = 'critical' if critical_value in ['yes', 'true', '1', 'critical'] else 'non-critical'
                del result['critical']  # Remove old field
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return {"raw_response": response, "error": "Failed to parse JSON"}
    else:
        print("No JSON found in response")
        return {"raw_response": response, "error": "No JSON found"}


def classify_error_criticality(
    error_info: Dict,
    model: str,
    max_retries: int = 3
) -> Dict:
    """
    Classify a single error as critical or non-critical.

    Args:
        error_info: Error details
        model: Model name
        max_retries: Max retry attempts

    Returns:
        Classification result
    """
    system_prompt, user_prompt = get_criticality_prompt(error_info)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        response = get_response(messages, model=model)
        result = extract_criticality_response(response)

        # Check if we got a valid result
        if "severity" in result and "harm_potential" in result:
            return result
        else:
            print(f"Attempt {attempt + 1}/{max_retries}: Invalid response. Retrying...")

    # If all retries failed
    print(f"Failed to get valid result after {max_retries} attempts")
    return {"error": "Failed to get valid result", "last_response": result}


def run_error_criticality_radeval(args):
    """Run error criticality classification on RadEval error detection results."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)

    # Determine directories
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'radeval')
    else:
        output_dir = args.output_dir

    # Path to error detection results
    detection_dir = os.path.join(output_dir, 'experiment_results', 'error_detection')

    print(f"\\nDataset: RadEval")
    print(f"Output directory: {output_dir}")
    print(f"Detection results: {detection_dir}")

    # Define perturbations
    perturbations = ['inject_false_prediction', 'inject_contradiction', 'inject_false_negation']

    if args.perturbation:
        if args.perturbation not in perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}")
        perturbations_to_run = [args.perturbation]
    else:
        perturbations_to_run = perturbations

    print(f"\\n{'='*80}")
    print("ERROR CRITICALITY CLASSIFICATION - RADEVAL")
    print(f"{'='*80}")
    print(f"Perturbations: {', '.join(perturbations_to_run)}")

    # Create experiment directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_criticality')
    os.makedirs(experiment_dir, exist_ok=True)

    model_name_clean = clean_model_name(args.model)

    # Process each perturbation
    for perturbation_name in perturbations_to_run:
        print(f"\\n{'='*80}")
        print(f"PERTURBATION: {perturbation_name.upper()}")
        print(f"{'='*80}")

        # Load error detection results
        perturbation_detection_dir = os.path.join(detection_dir, perturbation_name)
        detection_files = list(Path(perturbation_detection_dir).glob('*_error_detection.jsonl'))

        if not detection_files:
            print(f"  Warning: No detection files found for {perturbation_name}")
            continue

        detection_file = detection_files[0]
        print(f"  Loading detections from: {detection_file.name}")

        # Load detection data
        detection_data = []
        with open(detection_file, 'r') as f:
            for line in f:
                if line.strip():
                    detection_data.append(json.loads(line))

        print(f"  Loaded {len(detection_data)} detection results")

        # Create perturbation-specific output directory
        perturbation_output_dir = os.path.join(experiment_dir, perturbation_name)
        os.makedirs(perturbation_output_dir, exist_ok=True)

        output_filename = f"{perturbation_name}_{model_name_clean}_criticality.jsonl"
        output_path = os.path.join(perturbation_output_dir, output_filename)

        # Check which entries have already been processed
        processed_ids = get_processed_ids(output_path)
        remaining_data = [item for item in detection_data if item['id'] not in processed_ids]

        if len(remaining_data) == 0:
            print(f"  ✓ All {len(detection_data)} entries already processed")
        else:
            print(f"  Processing {len(remaining_data)} remaining entries")

            # Process each detection result
            for idx, item in enumerate(remaining_data, 1):
                item_id = item['id']
                detection_result = item.get('detection_result', {})
                errors = detection_result.get('errors', [])

                if not errors:
                    print(f"    [{idx}/{len(remaining_data)}] {item_id}: No errors detected, skipping")
                    continue

                print(f"    [{idx}/{len(remaining_data)}] {item_id}: {len(errors)} errors...", end=" ")

                start_time = time.time()

                # Classify each error
                error_classifications = []
                for error in errors:
                    classification = classify_error_criticality(
                        error,
                        model=args.model,
                        max_retries=3
                    )
                    error_classifications.append({
                        'error': error,
                        'classification': classification
                    })

                elapsed_time = time.time() - start_time

                # Count critical vs non-critical
                critical_count = sum(1 for ec in error_classifications
                                   if ec['classification'].get('severity') == 'critical')
                print(f"{critical_count}/{len(errors)} critical ({elapsed_time:.1f}s)")

                # Build result
                result = item.copy()
                result['error_classifications'] = error_classifications
                result['criticality_summary'] = {
                    'total_errors': len(errors),
                    'critical_errors': critical_count,
                    'non_critical_errors': len(errors) - critical_count
                }
                result['criticality_model'] = args.model
                result['random_seed'] = args.seed

                # Save to file
                save_result(output_path, result)

            print(f"  ✓ Completed criticality classification for {perturbation_name}")

    print(f"\\n{'='*80}")
    print("ERROR CRITICALITY CLASSIFICATION COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {experiment_dir}/{{perturbation}}/")
    print("\\nNext step: Analyze distribution of critical vs non-critical errors")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run error criticality classification on RadEval')

    parser.add_argument('--model', type=str, required=True,
                       help='Model to use for classification (e.g., gpt-4.1-2025-04-14)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/radeval)')

    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['inject_false_prediction', 'inject_contradiction', 'inject_false_negation'],
                       help='Specific perturbation to process (default: all three)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    run_error_criticality_radeval(args)
