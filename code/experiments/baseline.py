"""
Baseline experiment: Original perturbation + rating pipeline.

This is the refactored version of perturbation_pipeline.py.
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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perturbation_pipeline import (
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
    paths = setup_paths(args.output_dir)
    output_dir = paths['output_dir']
    model_name_clean = clean_model_name(args.model)

    # Define perturbation types
    all_perturbations_coarse = ['change_dosage', 'remove_sentences', 'add_typos', 'add_confusion']
    all_perturbations_fine = ['change_dosage']  # Only change_dosage for fine level

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
        # Use appropriate set based on level
        perturbation_names = all_perturbations_coarse if args.level != 'fine' else all_perturbations_fine
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
        level_perturbations = all_perturbations_fine if level == 'fine' else (perturbation_names if args.perturbation else all_perturbations_coarse)

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

        # For fine level: Use subset for experiments (perturbations should already exist from --generate-only)
        # For coarse level: Use full dataset
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

        # Step 1: Generate perturbations
        print(f"\n{'='*80}")
        print("STEP 1: PERTURBATIONS")
        print(f"{'='*80}")

        # Generate all perturbations upfront
        all_perturbations_dict = {}
        for perturbation_name in perturbation_names:
            all_perturbations_dict[perturbation_name] = {}

            # Determine parameter values for this perturbation
            # If None (not specified), use all values; otherwise use the specified value
            if perturbation_name == 'remove_sentences':
                remove_pcts = [0.3, 0.5, 0.7] if args.remove_pct is None else [args.remove_pct]
            else:
                remove_pcts = [0.3]  # Default for other perturbations

            if perturbation_name == 'add_typos':
                typo_probs = [0.3, 0.5, 0.7] if args.typo_prob is None else [args.typo_prob]
            else:
                typo_probs = [0.5]  # Default for other perturbations

            for remove_pct in remove_pcts:
                for typo_prob in typo_probs:
                    param_key = (remove_pct, typo_prob)
                    print(f"\n[{perturbation_name}]", end="")
                    if perturbation_name == 'remove_sentences' and len(remove_pcts) > 1:
                        print(f" remove_pct={remove_pct}", end="")
                    if perturbation_name == 'add_typos' and len(typo_probs) > 1:
                        print(f" typo_prob={typo_prob}", end="")
                    print()

                    perturbations_dict = get_or_create_perturbations(
                        perturbation_name=perturbation_name,
                        level=level,
                        qa_pairs=qa_pairs,  # Use subset for fine, full for coarse
                        typo_prob=typo_prob,
                        remove_pct=remove_pct,
                        seed=args.seed,
                        output_dir=output_dir
                    )
                    all_perturbations_dict[perturbation_name][param_key] = perturbations_dict

        # Step 2: Get/compute original ratings
        print(f"\n{'='*80}")
        print("STEP 2: ORIGINAL RATINGS")
        print(f"{'='*80}")

        # Always generate missing ratings (don't skip)
        original_ratings_dict = get_or_create_original_ratings(
            qa_pairs=qa_pairs,
            level=level,
            prompt_path=prompt_path,
            model=args.model,
            output_dir=output_dir,
            model_name_clean=model_name_clean,
            num_runs=args.num_runs,
            skip_missing=False  # Generate missing ratings automatically
        )

        # Filter qa_pairs to only include IDs that have original ratings
        qa_pairs_to_rate = [qa for qa in qa_pairs if qa[id_key] in original_ratings_dict]
        print(f"✓ Processing {len(qa_pairs_to_rate)} examples with original ratings")

        # Step 3: Rate perturbed answers
        print(f"\n{'='*80}")
        print("STEP 3: RATE PERTURBED ANSWERS")
        print(f"{'='*80}")

        # Create baseline experiment directory
        baseline_dir = os.path.join(output_dir, 'experiment_results', 'baseline')
        os.makedirs(baseline_dir, exist_ok=True)

        for perturbation_name in perturbation_names:
            print(f"\n{'='*80}")
            print(f"Processing perturbation: {perturbation_name.upper()}")
            print(f"{'='*80}")

            # Create perturbation-specific subdirectory under baseline
            perturbation_dir = os.path.join(baseline_dir, perturbation_name)
            os.makedirs(perturbation_dir, exist_ok=True)

            # Determine parameter values for this perturbation
            # If None (not specified), use all values; otherwise use the specified value
            if perturbation_name == 'remove_sentences':
                remove_pcts = [0.3, 0.5, 0.7] if args.remove_pct is None else [args.remove_pct]
            else:
                remove_pcts = [0.3]  # Default for other perturbations

            if perturbation_name == 'add_typos':
                typo_probs = [0.3, 0.5, 0.7] if args.typo_prob is None else [args.typo_prob]
            else:
                typo_probs = [0.5]  # Default for other perturbations

            # Iterate over parameter combinations
            for remove_pct in remove_pcts:
                for typo_prob in typo_probs:
                    param_key = (remove_pct, typo_prob)

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

                    # Check which entries have already been processed
                    processed_ids = get_processed_ids(output_path)
                    remaining_qa_pairs = [qa for qa in qa_pairs_to_rate if qa[id_key] not in processed_ids]

                    if len(remaining_qa_pairs) == 0:
                        print(f"All {len(qa_pairs_to_rate)} QA pairs already processed. Skipping.")
                        continue

                    print(f"Processing {len(remaining_qa_pairs)} remaining QA pairs (out of {len(qa_pairs_to_rate)} total)")
                    print(f"Saving results to: {perturbation_name}/{output_filename}")

                    # Get pre-generated perturbations
                    perturbations_dict = all_perturbations_dict[perturbation_name][param_key]

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
