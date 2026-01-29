#!/usr/bin/env python3
"""
Temporary script to generate ratings for existing perturbations.
"""

import json
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perturbation_pipeline import load_prompt, get_rating_with_averaging


def main():
    # Configuration
    perturbation_file = '/Users/Federica_1/Documents/GitHub/m3-eval/output/cqa_eval/perturbations/change_dosage_fine.jsonl'
    prompt_path = '/Users/Federica_1/Documents/GitHub/m3-eval/code/prompts/fineprompt_system.txt'
    model = 'gpt-4.1-2025-04-14'
    num_runs = 5

    # Output file
    output_file = '/Users/Federica_1/Documents/GitHub/m3-eval/output/cqa_eval/temp_gpt4_1_change_dosage_fine_ratings.jsonl'

    # Load perturbations
    perturbations = []
    with open(perturbation_file, 'r') as f:
        for line in f:
            perturbations.append(json.loads(line))

    print(f"Loaded {len(perturbations)} perturbations from {os.path.basename(perturbation_file)}")

    # Load prompt
    system_prompt, user_template = load_prompt(prompt_path)

    # Check for existing ratings to resume
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(entry['sentence_id'])
        print(f"Found {len(processed_ids)} already processed entries")

    # Process each perturbation
    with open(output_file, 'a') as f:
        for i, perturb in enumerate(perturbations):
            sentence_id = perturb['sentence_id']

            # Skip if already processed
            if sentence_id in processed_ids:
                print(f"[{i+1}/{len(perturbations)}] Skipping {sentence_id} (already processed)")
                continue

            question = perturb['question']
            perturbed_answer = perturb['perturbed_answer']

            print(f"\n[{i+1}/{len(perturbations)}] Rating {sentence_id}...")
            start_time = time.time()

            # Get rating for perturbed answer
            perturbed_rating = get_rating_with_averaging(
                question, perturbed_answer, system_prompt, user_template,
                model, num_runs=num_runs, flush_output=True
            )

            elapsed_time = time.time() - start_time
            print(f'Time taken: {elapsed_time:.2f} seconds')

            # Build result
            result = perturb.copy()
            result['perturbed_rating'] = perturbed_rating
            result['model'] = model

            # Save to file
            json.dump(result, f)
            f.write('\n')
            f.flush()

    print(f"\n{'='*80}")
    print("RATING COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
