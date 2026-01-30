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

    # Define perturbation types (starting with change_dosage and remove_sentences)
    all_perturbations = ['change_dosage', 'remove_sentences', 'add_typos', 'add_confusion']

    # Determine which perturbations to run
    if args.perturbation:
        if args.perturbation not in all_perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}")
        perturbation_names = [args.perturbation]
        print(f"Running single perturbation: {args.perturbation}")
    else:
        perturbation_names = all_perturbations
        print(f"Running all perturbations in order: {' -> '.join(perturbation_names)}")

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

        # Determine if we should use sentence_ids subset for fine level
        sentence_ids_subset_file = None
        if level == 'fine':
            subset_file = os.path.join(paths['project_root'], 'data', 'fine_sentence_ids_subset.json')
            if os.path.exists(subset_file):
                sentence_ids_subset_file = subset_file
                print(f"Using sentence_ids subset: {os.path.basename(subset_file)}")

        # Load data (with optional subset filtering for fine level)
        all_qa_pairs = load_qa_data(data_path, sentence_ids_subset_file=sentence_ids_subset_file)
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

        # Step 1: Get/compute original ratings
        print(f"\n{'='*80}")
        print("STEP 1: ORIGINAL RATINGS")
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
        # This is especially important for fine level where we skip computing missing ratings
        qa_pairs = [qa for qa in qa_pairs if qa[id_key] in original_ratings_dict]
        print(f"✓ Processing {len(qa_pairs)} examples with original ratings")

        # Step 2: Process each perturbation
        print(f"\n{'='*80}")
        print("STEP 2: PERTURBATIONS")
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
                    remaining_qa_pairs = [qa for qa in qa_pairs if qa[id_key] not in processed_ids]

                    if len(remaining_qa_pairs) == 0:
                        print(f"All {len(qa_pairs)} QA pairs already processed. Skipping.")
                        continue

                    print(f"Processing {len(remaining_qa_pairs)} remaining QA pairs (out of {len(qa_pairs)} total)")
                    print(f"Saving results to: {perturbation_name}/{output_filename}")

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
