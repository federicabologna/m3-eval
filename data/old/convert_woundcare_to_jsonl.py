"""
Convert WoundCare dataset to M3-Eval JSONL format.

This script converts the WoundCare test.json to the JSONL format expected by M3-Eval,
including image references, clinical metadata, and all 3 reference responses for visual-aware evaluation.
"""

import json
import os


def convert_woundcare_to_jsonl(input_path, output_path):
    """
    Convert WoundCare JSON to M3-Eval JSONL format.

    Args:
        input_path: Path to WoundCare test.json
        output_path: Path to output JSONL file
    """
    # Load WoundCare data
    with open(input_path, 'r') as f:
        woundcare_data = json.load(f)

    print(f"Loaded {len(woundcare_data)} encounters from WoundCare dataset")

    qa_pairs = []

    for encounter in woundcare_data:
        encounter_id = encounter['encounter_id']

        # Construct question from title and content (English only)
        question_title = encounter.get('query_title_en', '')
        question_content = encounter.get('query_content_en', '')

        # Combine title and content for the full question
        if question_title and question_content:
            question = f"{question_title}\n\n{question_content}"
        elif question_title:
            question = question_title
        elif question_content:
            question = question_content
        else:
            continue  # Skip if no question text

        # Extract clinical metadata
        clinical_metadata = {
            'anatomic_locations': encounter.get('anatomic_locations', []),
            'wound_type': encounter.get('wound_type', ''),
            'wound_thickness': encounter.get('wound_thickness', ''),
            'tissue_color': encounter.get('tissue_color', ''),
            'drainage_amount': encounter.get('drainage_amount', ''),
            'drainage_type': encounter.get('drainage_type', ''),
            'infection': encounter.get('infection', '')
        }

        # Get image references
        image_ids = encounter.get('image_ids', [])

        # Get all 3 reference responses (English only)
        responses = encounter.get('responses', [])

        if len(responses) != 3:
            print(f"Warning: Encounter {encounter_id} has {len(responses)} responses, expected 3")
            if not responses:
                continue  # Skip if no responses

        # Extract all reference responses
        reference_responses = []
        for response in responses:
            answer_text = response.get('content_en', '')
            author_id = response.get('author_id', '')
            if answer_text:
                reference_responses.append({
                    'author_id': author_id,
                    'content': answer_text
                })

        if not reference_responses:
            continue  # Skip if no valid responses

        # Create one entry per encounter with all reference responses
        qa_pair = {
            'question_id': encounter_id,
            'question': question,
            'reference_responses': reference_responses,
            'image_ids': image_ids,
            'clinical_metadata': clinical_metadata,
            'annotation_type': 'coarse'
        }

        qa_pairs.append(qa_pair)

    # Write to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for qa_pair in qa_pairs:
            f.write(json.dumps(qa_pair) + '\n')

    print(f"Converted {len(qa_pairs)} encounters to {output_path}")
    print(f"Total reference responses: {sum(len(qa['reference_responses']) for qa in qa_pairs)}")
    return len(qa_pairs)


def main():
    """Main conversion function."""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir is /Users/.../m3-eval/data/old
    # We need to go up to /Users/.../m3-eval/data
    data_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(data_dir)

    # Input: WoundCare test.json
    input_path = os.path.join(
        data_dir,
        'old',
        'woundcare',
        'dataset-challenge-mediqa-2025-wv',
        'test.json'
    )

    # Output: WoundCare test coarse JSONL
    output_path = os.path.join(
        data_dir,
        'woundcare_test_coarse.jsonl'
    )

    # Convert
    num_encounters = convert_woundcare_to_jsonl(input_path, output_path)

    print(f"\nConversion complete!")
    print(f"Total encounters: {num_encounters}")
    print(f"Output file: {output_path}")


if __name__ == '__main__':
    main()
