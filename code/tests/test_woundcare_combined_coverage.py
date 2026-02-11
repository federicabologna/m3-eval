"""
Test WoundCare answer perturbation coverage on the combined test+valid set.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))
from woundcare_perturbations import apply_woundcare_answer_perturbation


def test_perturbation_coverage(data_path):
    """Test how many answers can be perturbed."""

    # Load data
    print(f"Loading data from {data_path}...")
    qa_pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line))

    print(f"Loaded {len(qa_pairs)} total encounters\n")

    # Count by split
    test_count = sum(1 for qa in qa_pairs if qa.get('split') == 'test')
    valid_count = sum(1 for qa in qa_pairs if qa.get('split') == 'valid')
    print(f"  Test: {test_count}")
    print(f"  Valid: {valid_count}\n")

    # Perturbation types
    perturbation_types = [
        'swap_infection',
        'swap_anatomic_location',
        'swap_time_frequency',
        'swap_urgency',
        'swap_severity'
    ]

    # Track results
    results = {
        pert_type: {
            'successful': 0,
            'failed': 0,
            'test_successful': 0,
            'valid_successful': 0,
            'examples': []
        }
        for pert_type in perturbation_types
    }

    # Test each encounter
    for qa_pair in qa_pairs:
        question_id = qa_pair['question_id']
        original_answer = qa_pair['gpt4o_response']
        split = qa_pair.get('split', 'unknown')

        for pert_type in perturbation_types:
            perturbed_answer, success, metadata = apply_woundcare_answer_perturbation(
                original_answer,
                pert_type,
                seed=42
            )

            if success:
                results[pert_type]['successful'] += 1
                if split == 'test':
                    results[pert_type]['test_successful'] += 1
                elif split == 'valid':
                    results[pert_type]['valid_successful'] += 1

                if len(results[pert_type]['examples']) < 3:
                    results[pert_type]['examples'].append({
                        'question_id': question_id,
                        'split': split,
                        'changed': f"'{metadata['original_term']}' → '{metadata['new_term']}'"
                    })
            else:
                results[pert_type]['failed'] += 1

    # Print results
    print("=" * 80)
    print("PERTURBATION COVERAGE ANALYSIS")
    print("=" * 80)

    for pert_type in perturbation_types:
        successful = results[pert_type]['successful']
        test_succ = results[pert_type]['test_successful']
        valid_succ = results[pert_type]['valid_successful']
        failed = results[pert_type]['failed']
        percentage = (successful / len(qa_pairs)) * 100

        print(f"\n{pert_type.upper()}")
        print("-" * 80)
        print(f"  Total Successful: {successful}/{len(qa_pairs)} ({percentage:.1f}%)")
        print(f"    Test:  {test_succ}/{test_count} ({test_succ/test_count*100:.1f}%)")
        print(f"    Valid: {valid_succ}/{valid_count} ({valid_succ/valid_count*100:.1f}%)")
        print(f"  Failed: {failed}/{len(qa_pairs)} ({failed/len(qa_pairs)*100:.1f}%)")

        if results[pert_type]['examples']:
            print(f"\n  Examples:")
            for i, ex in enumerate(results[pert_type]['examples'], 1):
                print(f"    [{i}] {ex['question_id']} ({ex['split']}): {ex['changed']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    at_least_one = 0
    all_five = 0

    for qa_pair in qa_pairs:
        original_answer = qa_pair['gpt4o_response']
        perturbable_count = 0

        for pert_type in perturbation_types:
            _, success, _ = apply_woundcare_answer_perturbation(original_answer, pert_type, seed=42)
            if success:
                perturbable_count += 1

        if perturbable_count > 0:
            at_least_one += 1
        if perturbable_count == 5:
            all_five += 1

    print(f"Total encounters: {len(qa_pairs)}")
    print(f"  Test: {test_count}")
    print(f"  Valid: {valid_count}")
    print(f"\nPerturbable by at least one method: {at_least_one}/{len(qa_pairs)} ({at_least_one/len(qa_pairs)*100:.1f}%)")
    print(f"Perturbable by all five methods: {all_five}/{len(qa_pairs)} ({all_five/len(qa_pairs)*100:.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    data_path = '/Users/Federica_1/Documents/GitHub/m3-eval/data/woundcare_gpt4o_coarse.jsonl'
    test_perturbation_coverage(data_path)
