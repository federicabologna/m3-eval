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
{
  "detected": "yes" or "no",
  "explanation": "Brief explanation of your decision",
  "location": "Specific part of answer with error/issue, or null if none detected"
}"""

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
    paths = setup_paths(args.output_dir)
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Create experiment-specific output directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_detection')
    os.makedirs(experiment_dir, exist_ok=True)

    # Define perturbations to test (all except add_confusion which is too obvious)
    all_perturbations = ['add_typos', 'change_dosage', 'remove_sentences']

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

        # Select data path
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']
        print(f"Using data: {data_path}")

        # Load data
        qa_pairs = load_qa_data(data_path)
        id_key = get_id_key(qa_pairs)

        # Process each perturbation
        for perturbation_name in perturbation_names:
            print(f"\n{'='*80}")
            print(f"Testing detection for: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Create perturbation-specific subdirectory
            perturbation_dir = os.path.join(experiment_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Determine parameter values
            remove_pct_values = [args.remove_pct]
            if perturbation_name == 'remove_sentences' and args.all_remove_pct:
                remove_pct_values = [0.3, 0.5, 0.7]

            typo_prob_values = [args.typo_prob]
            if perturbation_name == 'add_typos' and args.all_typo_prob:
                typo_prob_values = [0.3, 0.5, 0.7]

            # Iterate over parameter combinations
            for remove_pct in remove_pct_values:
                for typo_prob in typo_prob_values:
                    # Determine output filename
                    if perturbation_name == 'remove_sentences':
                        output_filename = f"detection_{perturbation_name}_{remove_pct}removed_{level}_{model_name_clean}.jsonl"
                    elif perturbation_name == 'add_typos':
                        prob_str = str(typo_prob).replace('.', '')
                        output_filename = f"detection_{perturbation_name}_{prob_str}prob_{level}_{model_name_clean}.jsonl"
                    else:
                        output_filename = f"detection_{perturbation_name}_{level}_{model_name_clean}.jsonl"

                    output_path = os.path.join(perturbation_dir, output_filename)

                    # Check which entries have already been processed
                    processed_ids = get_processed_ids(output_path)
                    remaining_qa_pairs = [qa for qa in qa_pairs if qa[id_key] not in processed_ids]

                    if len(remaining_qa_pairs) == 0:
                        print(f"All {len(qa_pairs)} QA pairs already processed. Skipping.")
                        continue

                    print(f"Processing {len(remaining_qa_pairs)} remaining QA pairs (out of {len(qa_pairs)} total)")
                    print(f"Saving results to: error_detection/{perturbation_name}/{output_filename}")

                    # Load or generate perturbations
                    perturbations_dict = get_or_create_perturbations(
                        perturbation_name=perturbation_name,
                        level=level,
                        qa_pairs=qa_pairs,
                        typo_prob=typo_prob,
                        remove_pct=remove_pct,
                        seed=args.seed,
                        output_dir=output_dir
                    )

                    # Process each QA pair
                    for qa_pair in remaining_qa_pairs:
                        question = qa_pair['question']
                        original_answer = qa_pair['answer']

                        # Get pre-generated perturbation
                        perturbation_entry = perturbations_dict.get(qa_pair[id_key])

                        if perturbation_entry is None:
                            print(f"Skipping {qa_pair[id_key]} - no perturbation found")
                            continue

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
