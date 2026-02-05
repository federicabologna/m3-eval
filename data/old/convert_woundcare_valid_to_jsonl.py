"""
Convert WoundCare valid.json to JSONL format.

Similar to test set conversion - keeps all reference responses.
"""

import json
import os


def convert_valid_to_jsonl():
    """Convert valid set to JSONL format."""

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    valid_json_path = os.path.join(
        script_dir,
        'woundcare',
        'dataset-challenge-mediqa-2025-wv',
        'valid.json'
    )

    output_path = os.path.join(
        os.path.dirname(script_dir),
        'woundcare_valid_coarse.jsonl'
    )

    # Load valid data
    print(f"Loading valid data from {valid_json_path}...")
    with open(valid_json_path, 'r') as f:
        valid_data = json.load(f)

    print(f"Loaded {len(valid_data)} valid encounters")

    # Convert to JSONL format
    converted = []

    for encounter in valid_data:
        encounter_id = encounter['encounter_id']

        # Combine query title and content
        query_title = encounter.get('query_title_en', '')
        query_content = encounter.get('query_content_en', '')
        question = f"{query_title}\n{query_content}".strip()

        # Get all reference responses
        responses = encounter.get('responses', [])
        reference_responses = []

        for response in responses:
            answer_text = response.get('content_en', '')
            author_id = response.get('author_id', '')
            if answer_text:
                reference_responses.append({
                    'author_id': author_id,
                    'content': answer_text
                })

        # Clinical metadata
        clinical_metadata = {
            'anatomic_locations': encounter.get('anatomic_locations', []),
            'wound_type': encounter.get('wound_type', ''),
            'wound_thickness': encounter.get('wound_thickness', ''),
            'tissue_color': encounter.get('tissue_color', ''),
            'drainage_amount': encounter.get('drainage_amount', ''),
            'drainage_type': encounter.get('drainage_type', ''),
            'infection': encounter.get('infection', '')
        }

        converted_entry = {
            'question_id': encounter_id,
            'question': question,
            'reference_responses': reference_responses,
            'image_ids': encounter.get('image_ids', []),
            'clinical_metadata': clinical_metadata,
            'annotation_type': 'coarse'
        }

        converted.append(converted_entry)

    # Save as JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        for entry in converted:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(converted)} encounters")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total encounters: {len(converted)}")

    # Count reference responses
    ref_counts = {}
    for entry in converted:
        count = len(entry['reference_responses'])
        ref_counts[count] = ref_counts.get(count, 0) + 1

    print("\nReference responses per encounter:")
    for count in sorted(ref_counts.keys()):
        print(f"  {count} references: {ref_counts[count]} encounters")

    # Sample
    if converted:
        print("\n" + "=" * 80)
        print("SAMPLE ENTRY")
        print("=" * 80)
        sample = converted[0]
        print(f"Question ID: {sample['question_id']}")
        print(f"Question: {sample['question'][:200]}...")
        print(f"\nReference Responses: {len(sample['reference_responses'])}")
        for i, ref in enumerate(sample['reference_responses'], 1):
            print(f"  [{i}] {ref['author_id']}: {ref['content'][:100]}...")
        print(f"\nImages: {sample['image_ids']}")
        print(f"Clinical Metadata: {sample['clinical_metadata']}")

    print("=" * 80)


if __name__ == '__main__':
    convert_valid_to_jsonl()
