"""
Error Priming Experiment.

Test if warning about errors in the system prompt affects ratings.

Compare two conditions:
- Control: Uses baseline prompt (coarseprompt_system.txt / fineprompt_system.txt)
- Primed: Uses error_priming_coarse.txt / error_priming_fine.txt

The primed prompts embed the error warning directly in the system message:
- Coarse: "...an answer to that question which contains errors or misses important information"
- Fine: "...highlighted sentence from the answer. Note that the sentence contains errors."
"""

import json
import os
import random
import time
from typing import Dict

from helpers.experiment_utils import (
    setup_paths,
    load_qa_data,
    get_processed_ids,
    clean_model_name,
    get_id_key,
    get_or_create_perturbations,
    save_result
)
from helpers.multi_llm_inference import get_response, get_provider_from_model

# Import rating functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perturbation_pipeline import (
    load_prompt,
    get_rating_with_averaging
)




def run_error_priming_experiment(args):
    """
    Run error priming experiment.

    Test if warning about errors affects ratings.
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
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_priming')
    os.makedirs(experiment_dir, exist_ok=True)

    # Define perturbations to test
    all_perturbations = ['change_dosage', 'remove_sentences', 'add_typos', 'add_confusion']

    # Determine which perturbations to run
    if args.perturbation:
        if args.perturbation not in all_perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}")
        perturbation_names = [args.perturbation]
    else:
        perturbation_names = all_perturbations

    print(f"\nTesting error priming for: {', '.join(perturbation_names)}")

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

        # For fine level: Use subset; for coarse: Use full dataset
        if level == 'fine':
            subset_file = paths['fine_subset_path']
            all_qa_pairs = load_qa_data(data_path, sentence_ids_subset_file=subset_file)
            print(f"Loaded {len(all_qa_pairs)} examples from subset")
        else:
            all_qa_pairs = load_qa_data(data_path)
            print(f"Loaded {len(all_qa_pairs)} examples")

        # Apply start/end index filtering if specified
        if args.start_idx is not None or args.end_idx is not None:
            start = args.start_idx if args.start_idx is not None else 0
            end = args.end_idx if args.end_idx is not None else len(all_qa_pairs)
            qa_pairs = all_qa_pairs[start:end]
            print(f"Using subset: indices {start} to {end} ({len(qa_pairs)} examples)")
        else:
            qa_pairs = all_qa_pairs

        id_key = get_id_key(qa_pairs)

        # Select prompt path
        prompt_path = os.path.join(paths['prompts_dir'], f'{level}prompt_system.txt')

        # Process each perturbation
        for perturbation_name in perturbation_names:
            print(f"\n{'='*80}")
            print(f"Testing priming for: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Create perturbation-specific subdirectory
            perturbation_dir = os.path.join(experiment_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Determine parameter values
            # If None (not specified), use all values; otherwise use the specified value
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
                    # Determine output filename
                    if perturbation_name == 'remove_sentences':
                        output_filename = f"priming_{perturbation_name}_{remove_pct}removed_{level}_{model_name_clean}.jsonl"
                    elif perturbation_name == 'add_typos':
                        prob_str = str(typo_prob).replace('.', '')
                        output_filename = f"priming_{perturbation_name}_{prob_str}prob_{level}_{model_name_clean}.jsonl"
                    else:
                        output_filename = f"priming_{perturbation_name}_{level}_{model_name_clean}.jsonl"

                    output_path = os.path.join(perturbation_dir, output_filename)

                    # Check which entries have already been processed
                    processed_ids = get_processed_ids(output_path)
                    remaining_qa_pairs = [qa for qa in qa_pairs if qa[id_key] not in processed_ids]

                    if len(remaining_qa_pairs) == 0:
                        print(f"All {len(qa_pairs)} QA pairs already processed. Skipping.")
                        continue

                    print(f"Processing {len(remaining_qa_pairs)} remaining QA pairs (out of {len(qa_pairs)} total)")
                    print(f"Saving results to: error_priming/{perturbation_name}/{output_filename}")

                    # Load control prompt (baseline, no priming)
                    system_prompt_control, user_template = load_prompt(prompt_path)

                    # Load primed prompt (with error warning in system message)
                    priming_prompt_path = os.path.join(paths['prompts_dir'], f'error_priming_{level}.txt')
                    system_prompt_primed, _ = load_prompt(priming_prompt_path)

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

                        # Get CONTROL rating (baseline prompt, no error warning)
                        print(f"\n{qa_pair[id_key]} - Getting CONTROL rating (baseline prompt)...")
                        start_time = time.time()
                        control_rating = get_rating_with_averaging(
                            question, perturbed_answer, system_prompt_control, user_template,
                            args.model, num_runs=args.num_runs, flush_output=True
                        )
                        control_time = time.time() - start_time

                        # Get PRIMED rating (error warning in system prompt)
                        print(f"{qa_pair[id_key]} - Getting PRIMED rating (error warning in system prompt)...")
                        start_time = time.time()
                        primed_rating = get_rating_with_averaging(
                            question, perturbed_answer, system_prompt_primed, user_template,
                            args.model, num_runs=args.num_runs, flush_output=True
                        )
                        primed_time = time.time() - start_time

                        total_time = control_time + primed_time
                        print(f'Total time for {qa_pair[id_key]}: {total_time:.2f}s (control: {control_time:.2f}s, primed: {primed_time:.2f}s)')

                        # Build result
                        result = qa_pair.copy()
                        result['perturbation'] = perturbation_name
                        result['perturbed_answer'] = perturbed_answer
                        result['control_rating'] = control_rating  # Baseline prompt (no error warning)
                        result['primed_rating'] = primed_rating    # Error warning in system prompt
                        result['priming_method'] = 'system_prompt'  # Error warning embedded in system message
                        result['primed_prompt_file'] = f'error_priming_{level}.txt'
                        result['random_seed'] = args.seed

                        # Add perturbation metadata (typo_probability, remove_pctd, etc.)
                        for key in ['typo_probability', 'remove_pctd', 'change_counts', 'skip_reason']:
                            if key in perturbation_entry:
                                result[key] = perturbation_entry[key]

                        # Save to file
                        save_result(output_path, result)

    print(f"\n{'='*80}")
    print("ERROR PRIMING EXPERIMENT COMPLETED")
    print(f"{'='*80}")
