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
    paths = setup_paths(args.output_dir, dataset=args.dataset)
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Create experiment-specific output directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_priming')
    os.makedirs(experiment_dir, exist_ok=True)

    # Define perturbations to test
    all_perturbations = ['change_dosage', 'remove_sentences', 'add_typos']  # 'add_confusion' commented out

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

            # Load control prompt (baseline, no priming)
            system_prompt_control, user_template = load_prompt(prompt_path)

            # Load primed prompt (with error warning in system message)
            priming_prompt_path = os.path.join(paths['prompts_dir'], f'error_priming_{level}.txt')
            system_prompt_primed, _ = load_prompt(priming_prompt_path)

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
                        output_filename = f"priming_{perturbation_name}_{pct_str}pct_{level}_{model_name_clean}.jsonl"
                    elif perturbation_name == 'add_typos':
                        prob_str = str(typo_prob).replace('.', '')
                        output_filename = f"priming_{perturbation_name}_{prob_str}prob_{level}_{model_name_clean}.jsonl"
                    else:
                        output_filename = f"priming_{perturbation_name}_{level}_{model_name_clean}.jsonl"

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
