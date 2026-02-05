"""
Sample 5 random examples from WoundCare train set for prompt.
"""

import json
import random

# Load WoundCare train data
with open('/Users/Federica_1/Documents/GitHub/m3-eval/data/old/woundcare/dataset-challenge-mediqa-2025-wv/train.json', 'r') as f:
    data = json.load(f)

# Set seed for reproducibility
random.seed(42)

# Sample 5 random examples
samples = random.sample(data, 5)

# Print formatted examples
for i, sample in enumerate(samples, 1):
    print(f'Example {i}:')

    # Question
    question_title = sample.get('query_title_en', '')
    question_content = sample.get('query_content_en', '')

    if question_title and question_content:
        question = f"{question_title} {question_content}"
    elif question_title:
        question = question_title
    else:
        question = question_content

    print(f'Question: {question}')

    # Images
    images = sample.get('image_ids', [])
    if images:
        print(f"Images: {', '.join(images)}")

    # Answer (first response)
    if sample.get('responses'):
        answer = sample['responses'][0].get('content_en', '')
        print(f'Answer: {answer}')

    print()
