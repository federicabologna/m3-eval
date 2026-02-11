"""
Baseline experiment for CQA-Eval dataset: Perturbation + rating pipeline.

Supports regex-based perturbations (change_dosage, remove_sentences, add_typos).
"""

import json
import os
import random
import time
from typing import Dict, List

from helpers.experiment_utils import (
    setup_paths,
    load_qa_data,
    get_processed_ids,
    clean_model_name,
    get_id_key,
    get_or_create_perturbations,
    get_or_create_original_ratings,
    save_result
)
from helpers.multi_llm_inference import get_response, get_provider_from_model

# Import rating functions from perturbation_pipeline
from helpers.perturbation_pipeline import (
    load_prompt,
    get_rating_with_averaging
)


def run_baseline_experiment(args):
    """
    Run baseline perturbation + rating experiment.

    This replicates the original perturbation_pipeline.py behavior.
    """
    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using model: {args.model} (provider: {get_provider_from_model(args.model)})")

    # Setup paths
    paths = setup_paths(args.output_dir, dataset=args.dataset)
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Define perturbation types for CQA-Eval
    # Regex-based perturbations only
    all_perturbations_coarse = ['change_dosage', 'remove_sentences', 'add_typos']
    all_perturbations_fine = ['change_dosage', 'add_typos']
    default_perturbations_coarse = all_perturbations_coarse
    default_perturbations_fine = all_perturbations_fine

    # Determine which perturbations to run
    if args.perturbation:
        # For fine level, skip unsupported perturbations
        if args.level == 'fine' and args.perturbation not in all_perturbations_fine:
            print(f"Warning: {args.perturbation} is not supported for fine level")
            print(f"Fine level only supports: {', '.join(all_perturbations_fine)}")
            return

        perturbation_names = [args.perturbation]
        print(f"Running single perturbation: {args.perturbation}")
    else:
        # Use all perturbations
        perturbation_names = default_perturbations_coarse if args.level != 'fine' else default_perturbations_fine
        print(f"Running perturbations: {' -> '.join(perturbation_names)}")

    # Determine which levels to run
    levels = ['coarse', 'fine'] if args.level == 'both' else [args.level]

    # Print summary of experiments to be run
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Levels: {', '.join(levels)}")
    print(f"\nExperiments to run:")

    for level in levels:
        print(f"\n  {level.upper()}:")
        # Use perturbation_names which is already correctly set for both levels
        if level == 'fine':
            level_perturbations = [p for p in perturbation_names if p in all_perturbations_fine]
        else:
            level_perturbations = [p for p in perturbation_names if p in all_perturbations_coarse]

        for pert_name in level_perturbations:
            if pert_name == 'remove_sentences':
                if args.remove_pct is None:
                    print(f"    - {pert_name}: 30%, 50%, 70%")
                else:
                    print(f"    - {pert_name}: {int(args.remove_pct * 100)}%")
            elif pert_name == 'add_typos':
                if args.typo_prob is None:
                    print(f"    - {pert_name}: probability 0.3, 0.5, 0.7")
                else:
                    print(f"    - {pert_name}: probability {args.typo_prob}")
            else:
                print(f"    - {pert_name}")

    print(f"{'='*80}\n")

    # Process each level
    for level in levels:
        print(f"\n{'='*80}")
        print(f"PROCESSING LEVEL: {level.upper()}")
        print(f"{'='*80}")

        # Select data path
        data_path = paths['coarse_data_path'] if level == 'coarse' else paths['fine_data_path']
        print(f"Using data: {data_path}")

        # Handle dataset loading for CQA-Eval
        # Fine: Different datasets per perturbation (full for change_dosage, balanced for add_typos)
        # Coarse: Same dataset for all perturbations

        if level == 'fine':
            # CQA Eval fine: Load full dataset and balanced subset
            all_qa_pairs_full = load_qa_data(data_path)
            print(f"Loaded full fine dataset: {len(all_qa_pairs_full)} examples (for change_dosage)")

            # Load balanced subset (for add_typos)
            balanced_subset_file = paths['fine_balanced_subset_path']
            all_qa_pairs_balanced = load_qa_data(data_path, sentence_ids_subset_file=balanced_subset_file)
            print(f"Loaded balanced subset: {len(all_qa_pairs_balanced)} examples (for add_typos)")

            # Create a mapping of perturbation -> dataset
            perturbation_to_dataset = {
                'change_dosage': all_qa_pairs_full,
                'add_typos': all_qa_pairs_balanced,
            }
        else:
            # CQA Eval coarse: Use same dataset for all perturbations
            all_qa_pairs = load_qa_data(data_path)
            print(f"Loaded {len(all_qa_pairs)} examples")

            # For coarse, all perturbations use the same dataset
            perturbation_to_dataset = {p: all_qa_pairs for p in perturbation_names}

        # Select prompt path
        prompt_path = os.path.join(paths['prompts_dir'], f'{level}prompt_system.txt')

        # Create baseline experiment directory
        baseline_dir = os.path.join(output_dir, 'experiment_results', 'baseline')
        os.makedirs(baseline_dir, exist_ok=True)

        # =================================================================
        # STEP 1: Generate ALL perturbations using appropriate datasets
        # =================================================================
        print(f"\n{'='*80}")
        print("STEP 1: GENERATE PERTURBATIONS")
        print(f"{'='*80}")

        # Collect parameter values for later use
        perturbation_params = {}

        for perturbation_name in perturbation_names:
            print(f"\n[{perturbation_name.upper()}]")

            # Get the dataset for this perturbation
            qa_pairs_full = perturbation_to_dataset[perturbation_name]

            # Apply start/end index filtering if specified
            if args.start_idx is not None or args.end_idx is not None:
                start = args.start_idx if args.start_idx is not None else 0
                end = args.end_idx if args.end_idx is not None else len(qa_pairs_full)
                qa_pairs = qa_pairs_full[start:end]
                print(f"  Dataset: {len(qa_pairs)} examples (indices {start}-{end})")
            else:
                qa_pairs = qa_pairs_full
                print(f"  Dataset: {len(qa_pairs)} examples")

            # Determine parameter values for this perturbation
            if perturbation_name == 'remove_sentences':
                remove_pcts = [0.3, 0.5, 0.7] if args.remove_pct is None else [args.remove_pct]
            else:
                remove_pcts = [0.3]

            if perturbation_name == 'add_typos':
                typo_probs = [0.3, 0.5, 0.7] if args.typo_prob is None else [args.typo_prob]
            else:
                typo_probs = [0.5]

            # Store for later use
            perturbation_params[perturbation_name] = {
                'remove_pcts': remove_pcts,
                'typo_probs': typo_probs
            }

            # Generate perturbations
            for remove_pct in remove_pcts:
                for typo_prob in typo_probs:
                    get_or_create_perturbations(
                        perturbation_name=perturbation_name,
                        level=level,
                        qa_pairs=qa_pairs,
                        typo_prob=typo_prob,
                        remove_pct=remove_pct,
                        seed=args.seed,
                        output_dir=output_dir,
                        model=args.model
                    )

        # =================================================================
        # STEP 2: Collect successful perturbation IDs
        # =================================================================
        print(f"\n{'='*80}")
        print("STEP 2: COLLECT SUCCESSFUL PERTURBATION IDS")
        print(f"{'='*80}")

        from helpers.experiment_utils import get_successful_perturbation_ids

        # Get all typo probs and remove pcts used
        all_typo_probs = set()
        all_remove_pcts = set()
        for params in perturbation_params.values():
            all_typo_probs.update(params['typo_probs'])
            all_remove_pcts.update(params['remove_pcts'])

        successful_ids = get_successful_perturbation_ids(
            perturbation_names=perturbation_names,
            level=level,
            output_dir=output_dir,
            typo_probs=list(all_typo_probs),
            remove_pcts=list(all_remove_pcts)
        )

        print(f"✓ Found {len(successful_ids)} IDs with successful perturbations")

        # =================================================================
        # STEP 3: Generate original ratings for successful IDs only
        # =================================================================
        print(f"\n{'='*80}")
        print("STEP 3: GENERATE ORIGINAL RATINGS")
        print(f"{'='*80}")

        # Load full dataset and filter to successful IDs
        if args.dataset == 'medinfo':
            if level == 'coarse':
                full_data_path = os.path.join(paths['project_root'], 'data', 'medinfo2019_medications_qa_coarse.jsonl')
            else:
                full_data_path = os.path.join(paths['project_root'], 'data', 'medinfo2019_medications_qa_fine.jsonl')
        else:
            full_data_path = data_path

        all_qa_data = load_qa_data(full_data_path)
        id_key = get_id_key(all_qa_data)

        # Filter to only successful IDs
        qa_pairs_for_rating = [qa for qa in all_qa_data if qa[id_key] in successful_ids]
        print(f"Filtered to {len(qa_pairs_for_rating)} examples with successful perturbations")

        # Generate original ratings
        original_ratings_dict = get_or_create_original_ratings(
            qa_pairs=qa_pairs_for_rating,
            level=level,
            prompt_path=prompt_path,
            model=args.model,
            output_dir=output_dir,
            model_name_clean=model_name_clean,
            num_runs=args.num_runs,
            skip_missing=False
        )

        print(f"✓ {len(original_ratings_dict)} original ratings available")

        # =================================================================
        # STEP 4: Rate perturbed answers
        # =================================================================
        print(f"\n{'='*80}")
        print("STEP 4: RATE PERTURBED ANSWERS")
        print(f"{'='*80}")

        # Process each perturbation
        for perturbation_name in perturbation_names:
            print(f"\n[{perturbation_name.upper()}]")

            # Create perturbation-specific subdirectory
            perturbation_dir = os.path.join(baseline_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Get parameters for this perturbation
            params = perturbation_params[perturbation_name]
            remove_pcts = params['remove_pcts']
            typo_probs = params['typo_probs']

            # Iterate over parameter combinations
            for remove_pct in remove_pcts:
                for typo_prob in typo_probs:
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
                        continue

                    # Load perturbations into dict
                    perturbations_dict = {}
                    with open(pert_filepath, 'r') as f:
                        for line in f:
                            entry = json.loads(line)
                            perturbations_dict[entry[id_key]] = entry

                    # Check which entries have already been processed
                    processed_ids = get_processed_ids(output_path)

                    # Get IDs that have perturbations and original ratings but haven't been rated yet
                    ids_to_rate = [
                        id_ for id_ in perturbations_dict.keys()
                        if id_ in original_ratings_dict and id_ not in processed_ids
                        and 'skip_reason' not in perturbations_dict[id_]
                    ]

                    if len(ids_to_rate) == 0:
                        print(f"  All perturbed answers already rated. Skipping.")
                        continue

                    print(f"  Rating {len(ids_to_rate)} perturbed answers")

                    # Create QA pairs for rating (need full QA data)
                    qa_pairs_to_rate = [qa for qa in qa_pairs_for_rating if qa[id_key] in ids_to_rate]

                    # Process each QA pair
                    for qa_pair in qa_pairs_to_rate:
                        question = qa_pair['question']
                        original_answer = qa_pair['answer']

                        # Get pre-generated perturbation
                        perturbation_entry = perturbations_dict.get(qa_pair[id_key])

                        if perturbation_entry is None:
                            print(f"Skipping {qa_pair[id_key]} - no perturbation found")
                            continue

                        perturbed_answer = perturbation_entry['perturbed_answer']

                        # Get perturbed rating
                        start_time = time.time()
                        perturbed_rating = get_rating_with_averaging(
                            question, perturbed_answer, *load_prompt(prompt_path),
                            args.model, num_runs=args.num_runs, flush_output=True
                        )
                        elapsed_time = time.time() - start_time
                        print(f'Time taken for {qa_pair[id_key]}: {elapsed_time:.2f} seconds')

                        # Get original rating from dict
                        original_rating = original_ratings_dict.get(qa_pair[id_key])

                        if original_rating is None:
                            print(f"WARNING: No original rating found for {qa_pair[id_key]}, skipping...")
                            continue

                        # Build result
                        result = qa_pair.copy()
                        result['perturbation'] = perturbation_name
                        result['perturbed_answer'] = perturbed_answer
                        result['original_rating'] = original_rating
                        result['perturbed_rating'] = perturbed_rating
                        result['random_seed'] = args.seed

                        # Add perturbation metadata (typo_probability, remove_pctd, etc.)
                        for key in ['typo_probability', 'remove_pctd', 'change_counts', 'skip_reason']:
                            if key in perturbation_entry:
                                result[key] = perturbation_entry[key]

                        # Save to file
                        save_result(output_path, result)

    print(f"\n{'='*80}")
    print("BASELINE EXPERIMENT COMPLETED")
    print(f"{'='*80}")
