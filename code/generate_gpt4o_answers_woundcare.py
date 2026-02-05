"""
Generate GPT-4o answers for WoundCare test/valid set with multimodal five-shot prompting.

This script:
1. Loads the WoundCare test (3 refs) or valid (2 refs) set
2. Uses the five-shot prompt with wound care examples
3. Includes images from both prompt examples and questions
4. Generates GPT-4o responses for each question
5. Saves results with reference responses + GPT-4o response
"""

import json
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Verify API key is loaded
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError(f"OPENAI_API_KEY not found. Please check {env_path}")

# Add helpers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))
from multimodal_inference import load_woundcare_images, get_multimodal_response


def load_five_shot_prompt():
    """Load the five-shot prompt template."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'prompts', 'five_shot_prompt.txt')

    with open(prompt_path, 'r') as f:
        return f.read()


def load_five_shot_images():
    """Load images for the five-shot examples."""
    # Images from the five-shot prompt examples
    example_image_ids = [
        'IMG_ENC0058_0001.jpg',  # Example 1
        'IMG_ENC0013_0001.jpg',  # Example 2
        'IMG_ENC0141_0001.jpg',  # Example 3
        'IMG_ENC0126_0001.jpg', 'IMG_ENC0126_0002.jpg',  # Example 4
        'IMG_ENC0115_0001.jpg', 'IMG_ENC0115_0002.jpg'   # Example 5
    ]

    # Use train images directory for five-shot examples
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_images_dir = os.path.join(
        project_root,
        'data',
        'old',
        'woundcare',
        'dataset-challenge-mediqa-2025-wv',
        'images_train',
        'images_train'
    )

    return load_woundcare_images(example_image_ids, train_images_dir)


def get_images_dir_for_split(split):
    """Get the appropriate images directory for test or valid split."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    base_path = os.path.join(
        project_root,
        'data',
        'old',
        'woundcare',
        'dataset-challenge-mediqa-2025-wv'
    )

    if split == 'test':
        # Test has nested structure: images_test/images_test/
        images_dir = os.path.join(base_path, 'images_test', 'images_test')
    elif split == 'valid':
        # Valid has flat structure: images_valid/
        images_dir = os.path.join(base_path, 'images_valid')
    else:
        raise ValueError(f"Unknown split: {split}")

    return images_dir


def create_prompt_with_question(five_shot_prompt, question):
    """Create full prompt with five-shot examples and the new question."""
    prompt = five_shot_prompt + f"\n\nNow answer the following question:\n\nQuestion: {question}\nAnswer:"
    return prompt


def generate_gpt4o_answers(
    data_path,
    output_path,
    split="test",
    model="gpt-4o",
    resume=True
):
    """
    Generate GPT-4o answers for all questions.

    Args:
        data_path: Path to woundcare_test_coarse.jsonl or woundcare_valid_coarse.jsonl
        output_path: Path to save results
        split: 'test' or 'valid' (for image directory selection)
        model: Model name (default: gpt-4o)
        resume: Whether to resume from existing output file
    """

    # Load data
    print(f"Loading {split} data from {data_path}...")
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Loaded {len(data)} {split} encounters")

    # Get images directory for this split
    images_dir = get_images_dir_for_split(split)

    # Load five-shot prompt
    print("Loading five-shot prompt...")
    five_shot_prompt = load_five_shot_prompt()

    # Load five-shot example images
    print("Loading five-shot example images...")
    five_shot_images = load_five_shot_images()
    print(f"Loaded {len(five_shot_images)} five-shot example images")

    # Check for existing output to resume
    processed_ids = set()
    results = []

    if resume and os.path.exists(output_path):
        print(f"Resuming from {output_path}...")
        with open(output_path, 'r') as f:
            for line in f:
                result = json.loads(line)
                results.append(result)
                processed_ids.add(result['question_id'])
        print(f"Found {len(processed_ids)} already processed encounters")

    # Process each encounter
    print(f"\nGenerating GPT-4o answers using model: {model}")
    print("=" * 80)

    for qa_pair in tqdm(data, desc="Generating answers"):
        question_id = qa_pair['question_id']

        # Skip if already processed
        if question_id in processed_ids:
            continue

        try:
            # Load question images from appropriate split directory
            question_images = []
            if qa_pair.get('image_ids'):
                question_images = load_woundcare_images(qa_pair['image_ids'], images_dir)

            # Combine five-shot images with question images
            all_images = five_shot_images + question_images

            # Create prompt with question
            full_prompt = create_prompt_with_question(
                five_shot_prompt,
                qa_pair['question']
            )

            # Get GPT-4o response
            gpt4o_response = get_multimodal_response(
                text=full_prompt,
                images=all_images,
                model=model,
                system_message=None  # System message is included in prompt
            )

            # Create result entry
            result = {
                'question_id': question_id,
                'question': qa_pair['question'],
                'reference_responses': qa_pair['reference_responses'],
                'gpt4o_response': gpt4o_response,
                'image_ids': qa_pair.get('image_ids', []),
                'clinical_metadata': qa_pair['clinical_metadata'],
                'annotation_type': 'coarse'
            }

            results.append(result)

            # Save incrementally
            with open(output_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')

        except Exception as e:
            print(f"\nError processing {question_id}: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"Complete! Generated {len(results)} answers")
    print(f"Output saved to: {output_path}")
    print("=" * 80)

    # Print sample result
    if results:
        print("\nSample Result:")
        print("-" * 80)
        sample = results[0]
        print(f"Question ID: {sample['question_id']}")
        print(f"Question: {sample['question'][:150]}...")
        print(f"\nReference Responses: {len(sample['reference_responses'])}")
        for i, ref in enumerate(sample['reference_responses'], 1):
            print(f"  [{i}] {ref['author_id']}: {ref['content'][:80]}...")
        print(f"\nGPT-4o Response: {sample['gpt4o_response'][:150]}...")
        print(f"\nImages: {sample['image_ids']}")
        print(f"Clinical Metadata: {sample['clinical_metadata']}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate GPT-4o answers for WoundCare dataset')
    parser.add_argument('--split', choices=['test', 'valid'], default='test',
                       help='Dataset split to process (default: test)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: only process 3 examples')
    parser.add_argument('--model', default='gpt-4o', help='Model to use (default: gpt-4o)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths based on split
    if args.split == 'test':
        data_path = os.path.join(project_root, 'data', 'woundcare_test_coarse.jsonl')
        output_basename = 'woundcare_test_gpt4o_coarse'
    elif args.split == 'valid':
        data_path = os.path.join(project_root, 'data', 'woundcare_valid_coarse.jsonl')
        output_basename = 'woundcare_valid_gpt4o_coarse'
    else:
        raise ValueError(f"Unknown split: {args.split}")

    if args.test_mode:
        output_path = os.path.join(project_root, 'data', f'{output_basename}_sample3.jsonl')
        print("\n" + "=" * 80)
        print(f"TEST MODE: Processing only first 3 examples from {args.split} set")
        print("=" * 80 + "\n")

        # Load only first 3 examples
        data_subset = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                data_subset.append(json.loads(line))

        # Save as temporary file
        temp_path = data_path + '.temp3'
        with open(temp_path, 'w') as f:
            for item in data_subset:
                f.write(json.dumps(item) + '\n')

        data_path = temp_path
    else:
        output_path = os.path.join(project_root, 'data', f'{output_basename}.jsonl')

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate answers
    generate_gpt4o_answers(
        data_path=data_path,
        output_path=output_path,
        split=args.split,
        model=args.model,
        resume=True
    )

    # Clean up temp file if in test mode
    if args.test_mode and os.path.exists(data_path):
        os.remove(data_path)


if __name__ == '__main__':
    main()
