"""
Test WoundCare prompt formatting with actual test dataset.
"""

import json
import sys
import os

# Add helpers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))
from woundcare_prompt_formatter import format_woundcare_evaluation_prompt_with_metadata


def test_prompt_with_real_data():
    """Test prompt formatting with real WoundCare test data."""
    # Load first test case
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'woundcare_test_coarse.jsonl')

    with open(data_path, 'r') as f:
        qa_pair = json.loads(f.readline())

    print("=" * 80)
    print("WOUNDCARE PROMPT TEST")
    print("=" * 80)
    print(f"\nTest Case ID: {qa_pair['question_id']}")
    print(f"Number of Reference Responses: {len(qa_pair['reference_responses'])}")
    print(f"Image IDs: {qa_pair['image_ids']}")
    print(f"Wound Type: {qa_pair['clinical_metadata'].get('wound_type')}")
    print(f"Infection Status: {qa_pair['clinical_metadata'].get('infection')}")

    # Test 1: Using first reference as candidate
    print("\n" + "=" * 80)
    print("TEST 1: Evaluating first reference response as candidate")
    print("=" * 80)

    prompt1 = format_woundcare_evaluation_prompt_with_metadata(
        qa_pair,
        candidate_response=qa_pair['reference_responses'][0]['content']
    )
    print(prompt1)

    # Test 2: Using a custom candidate (simulating a perturbed response)
    print("\n\n" + "=" * 80)
    print("TEST 2: Evaluating a custom candidate response")
    print("=" * 80)

    custom_candidate = "Keep the wound clean and dry. Apply a bandage to protect it from further injury."

    prompt2 = format_woundcare_evaluation_prompt_with_metadata(
        qa_pair,
        candidate_response=custom_candidate
    )
    print(prompt2)


if __name__ == '__main__':
    test_prompt_with_real_data()
