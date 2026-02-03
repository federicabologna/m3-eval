"""
Error Detection Experiment.

Ask models to detect errors in perturbed answers without being told there is an error.

For change_dosage and add_typos: "Is there an error in this answer?"
For remove_sentences: "Is important information missing from this answer?"

Expected output:
- yes_no: "yes" or "no"
- explanation: Open-ended explanation
- localization: Specific part of answer with error (if detected)
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
    clean_model_name,
    get_id_key,
    get_or_create_perturbations,
    save_result,
    get_perturbation_prompt_mapping
)
from helpers.multi_llm_inference import get_response, get_provider_from_model


def get_detection_prompt(perturbation_name: str, question: str, answer: str, level: str = 'coarse') -> tuple:
    """
    Build detection prompt for a given perturbation type.

    Args:
        perturbation_name: Name of perturbation
        question: Medical question
        answer: Answer text (answer or sentence depending on level)
        level: 'coarse' (answer-level) or 'fine' (sentence-level)

    Returns:
        (system_prompt, user_prompt)
    """
    mapping = get_perturbation_prompt_mapping(level)
    prompt_info = mapping[perturbation_name]

    text_type = 'sentence' if level == 'fine' else 'answer'

    system_prompt = f"""You are a medical expert evaluating the quality of {'sentences in' if level == 'fine' else ''} answers to medical questions.

Your task is to carefully analyze the {text_type} and determine if there are any errors or if important information is missing.

Provide your response in the following JSON format:
{{
  "detected": "yes" or "no",
  "explanation": "Brief explanation of your decision",
  "location": "Specific part of answer with error/issue, or null if none detected"
}}"""

    user_prompt = f"""QUESTION:
{question}

ANSWER:
{answer}

{prompt_info['question']}

Provide your response in JSON format with "detected" (yes/no), "explanation", and "location" fields."""

    return system_prompt, user_prompt


def extract_detection_response(response: str) -> Dict:
    """Extract detection result from model response."""
    # Try to find JSON object in the response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
            # Normalize detected field to yes/no
            if 'detected' in result:
                detected_value = str(result['detected']).lower()
                result['detected'] = 'yes' if detected_value in ['yes', 'true', '1'] else 'no'
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return {"raw_response": response, "error": "Failed to parse JSON"}
    else:
        print("No JSON found in response")
        return {"raw_response": response, "error": "No JSON found"}


def get_detection_result(question: str, answer: str, perturbation_name: str, model: str, level: str = 'coarse', max_retries: int = 3) -> Dict:
    """
    Get error detection result for an answer.

    Args:
        question: Medical question
        answer: Answer text (answer or sentence depending on level)
        perturbation_name: Name of perturbation
        model: Model name
        level: 'coarse' or 'fine'
        max_retries: Max retry attempts

    Returns:
        {
            "detected": "yes" or "no",
            "explanation": str,
            "location": str or None
        }
    """
    system_prompt, user_prompt = get_detection_prompt(perturbation_name, question, answer, level)

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


def run_error_detection_experiment(args):
    """
    Run error detection experiment.

    Test if models can detect errors in perturbed answers without being told.
    """
    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    paths = setup_paths(args.output_dir, dataset=args.dataset)
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Create experiment-specific output directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_detection')
    os.makedirs(experiment_dir, exist_ok=True)

    # Define perturbations to test
    all_perturbations = ['change_dosage', 'remove_sentences', 'add_typos']

    # Determine which perturbations to run
    if args.perturbation:
        if args.perturbation not in all_perturbations:
            print(f"Warning: {args.perturbation} not typically used for error detection")
            perturbation_names = [args.perturbation]
        else:
            perturbation_names = [args.perturbation]
    else:
        perturbation_names = all_perturbations

    print(f"\nTesting error detection for: {', '.join(perturbation_names)}")

    # Determine which levels to run
    levels = ['coarse', 'fine'] if args.level == 'both' else [args.level]

    # Process each level
    for level in levels:
        print(f"\n{'='*80}")
        print(f"PROCESSING LEVEL: {level.upper()}")
        print(f"{'='*80}")

        # Select data path for loading full QA data
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']

        # Load full QA dataset (we'll filter to IDs with perturbations)
        if args.dataset == 'medinfo':
            if level == 'coarse':
                full_data_path = os.path.join(paths['project_root'], 'data', 'medinfo2019_medications_qa_coarse.jsonl')
            else:
                full_data_path = os.path.join(paths['project_root'], 'data', 'medinfo2019_medications_qa_fine.jsonl')
        else:
            full_data_path = data_path

        all_qa_data = load_qa_data(full_data_path)
        print(f"Loaded {len(all_qa_data)} QA pairs")

        # Create a lookup dict for fast access
        id_key = get_id_key(all_qa_data)
        qa_lookup = {qa[id_key]: qa for qa in all_qa_data}

        # Process each perturbation
        for perturbation_name in perturbation_names:
            print(f"\n{'='*80}")
            print(f"Testing detection for: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Create perturbation-specific subdirectory
            perturbation_dir = os.path.join(experiment_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Determine parameter values
            if perturbation_name == 'remove_sentences':
                remove_pct_values = [0.3, 0.5, 0.7] if args.remove_pct is None else [args.remove_pct]
            else:
                remove_pct_values = [0.3]

            if perturbation_name == 'add_typos':
                typo_prob_values = [0.3, 0.5, 0.7] if args.typo_prob is None else [args.typo_prob]
            else:
                typo_prob_values = [0.5]

            # Iterate over parameter combinations
            for remove_pct in remove_pct_values:
                for typo_prob in typo_prob_values:
                    # Load perturbations from file
                    perturbations_dir = os.path.join(output_dir, 'perturbations')
                    if perturbation_name == 'remove_sentences':
                        pct_str = str(int(remove_pct * 100))
                        pert_filename = f"{perturbation_name}_{pct_str}pct_{level}.jsonl"
                    elif perturbation_name == 'add_typos':
                        prob_str = str(typo_prob).replace('.', '')
                        pert_filename = f"{perturbation_name}_{prob_str}prob_{level}.jsonl"
                    else:
                        pert_filename = f"{perturbation_name}_{level}.jsonl"

                    pert_filepath = os.path.join(perturbations_dir, pert_filename)

                    if not os.path.exists(pert_filepath):
                        print(f"  Perturbation file not found: {pert_filename}")
                        print(f"  Please run baseline experiment first to generate perturbations")
                        continue

                    # Load perturbations
                    perturbations_dict = {}
                    with open(pert_filepath, 'r') as f:
                        for line in f:
                            entry = json.loads(line)
                            # Only include successful perturbations
                            if 'skip_reason' not in entry:
                                perturbations_dict[entry[id_key]] = entry

                    print(f"  Loaded {len(perturbations_dict)} successful perturbations from {pert_filename}")

                    # Determine output filename
                    if perturbation_name == 'remove_sentences':
                        pct_str = str(int(remove_pct * 100))
                        output_filename = f"detection_{perturbation_name}_{pct_str}pct_{level}_{model_name_clean}.jsonl"
                    elif perturbation_name == 'add_typos':
                        prob_str = str(typo_prob).replace('.', '')
                        output_filename = f"detection_{perturbation_name}_{prob_str}prob_{level}_{model_name_clean}.jsonl"
                    else:
                        output_filename = f"detection_{perturbation_name}_{level}_{model_name_clean}.jsonl"

                    output_path = os.path.join(perturbation_dir, output_filename)

                    # Check which entries have already been processed
                    processed_ids = get_processed_ids(output_path)
                    ids_to_process = [id_ for id_ in perturbations_dict.keys() if id_ not in processed_ids]

                    if len(ids_to_process) == 0:
                        print(f"  All perturbed answers already processed. Skipping.")
                        continue

                    print(f"  Processing {len(ids_to_process)} remaining entries")

                    # Process each perturbed answer
                    for entry_id in ids_to_process:
                        perturbation_entry = perturbations_dict[entry_id]
                        qa_pair = qa_lookup.get(entry_id)

                        if qa_pair is None:
                            print(f"  Warning: QA pair not found for {entry_id}")
                            continue
                        question = qa_pair['question']
                        perturbed_answer = perturbation_entry['perturbed_answer']

                        # Get detection result
                        start_time = time.time()
                        detection_result = get_detection_result(
                            question, perturbed_answer, perturbation_name,
                            args.model, level=level, max_retries=args.max_retries
                        )
                        elapsed_time = time.time() - start_time
                        print(f'Detection for {qa_pair[id_key]}: {detection_result.get("detected", "error")} ({elapsed_time:.2f}s)')

                        # Build result
                        result = qa_pair.copy()
                        result['perturbation'] = perturbation_name
                        result['perturbed_answer'] = perturbed_answer
                        result['detection_result'] = detection_result
                        result['random_seed'] = args.seed

                        # Add perturbation metadata (typo_probability, remove_pctd, etc.)
                        for key in ['typo_probability', 'remove_pctd', 'change_counts', 'skip_reason']:
                            if key in perturbation_entry:
                                result[key] = perturbation_entry[key]

                        # Save to file
                        save_result(output_path, result)

    print(f"\n{'='*80}")
    print("ERROR DETECTION EXPERIMENT COMPLETED")
    print(f"{'='*80}")
