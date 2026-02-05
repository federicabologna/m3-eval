"""
Helper functions for formatting WoundCare evaluation prompts.
"""

import os
from typing import Dict, List


def load_woundcare_prompt_template(prompts_dir: str = None) -> str:
    """
    Load the WoundCare rating prompt template.

    Args:
        prompts_dir: Path to prompts directory (optional)

    Returns:
        Prompt template string with placeholders
    """
    if prompts_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(script_dir, 'prompts')

    template_path = os.path.join(prompts_dir, 'woundcare_rating_system.txt')

    with open(template_path, 'r') as f:
        template = f.read()

    return template


def format_woundcare_evaluation_prompt(
    query: str,
    candidate_response: str,
    prompts_dir: str = None
) -> str:
    """
    Format the complete WoundCare evaluation prompt.

    Args:
        query: Patient query/question
        candidate_response: The response to be evaluated
        prompts_dir: Path to prompts directory (optional)

    Returns:
        Formatted prompt ready to send to LLM
    """
    # Load template
    template = load_woundcare_prompt_template(prompts_dir)

    # Fill in template (no references used)
    prompt = template.format(
        query=query,
        candidate_response=candidate_response
    )

    return prompt


def format_woundcare_evaluation_prompt_with_metadata(
    qa_pair: Dict,
    candidate_response: str = None,
    prompts_dir: str = None
) -> str:
    """
    Format evaluation prompt from a WoundCare QA pair dict.

    Args:
        qa_pair: QA pair dict with 'question', 'clinical_metadata', 'image_ids'
        candidate_response: Candidate response to evaluate (required if not using gpt4o_response)
        prompts_dir: Path to prompts directory (optional)

    Returns:
        Formatted prompt ready to send to LLM
    """
    query = qa_pair.get('question', '')

    # Add clinical metadata and image context to query
    clinical_metadata = qa_pair.get('clinical_metadata', {})
    image_ids = qa_pair.get('image_ids', [])

    # Enhance query with clinical context
    enhanced_query = query

    if image_ids:
        enhanced_query += f"\n\n[Associated Images: {', '.join(image_ids)}]"

    if clinical_metadata:
        metadata_parts = []
        for key, value in clinical_metadata.items():
            if value and value != [] and value != '':
                if isinstance(value, list):
                    value_str = ', '.join(value)
                else:
                    value_str = str(value)
                # Convert snake_case to Title Case
                key_formatted = key.replace('_', ' ').title()
                metadata_parts.append(f"{key_formatted}: {value_str}")

        if metadata_parts:
            enhanced_query += "\n\n[Clinical Information]\n" + "\n".join(metadata_parts)

    # Use provided candidate response (no references needed)
    if candidate_response is None:
        candidate_response = qa_pair.get('gpt4o_response', '')

    return format_woundcare_evaluation_prompt(
        query=enhanced_query,
        candidate_response=candidate_response,
        prompts_dir=prompts_dir
    )


# Example usage
if __name__ == '__main__':
    # Example QA pair
    example_qa = {
        'question': "There's a hole on the sole of my foot. It hurts a lot. What should I do?",
        'reference_responses': [
            {
                'author_id': 'annotator1',
                'content': 'This could be a corn or callus. Try over-the-counter treatments and see a podiatrist if it doesn\'t improve.'
            },
            {
                'author_id': 'annotator2',
                'content': 'Possibly a plantar wart. Use OTC treatments or see a podiatrist for evaluation.'
            },
            {
                'author_id': 'annotator3',
                'content': 'Cover with a dressing to reduce friction. See a provider to rule out infection or foreign body.'
            }
        ],
        'clinical_metadata': {
            'anatomic_locations': ['foot-sole'],
            'wound_type': 'pressure',
            'wound_thickness': 'stage_I',
            'infection': 'not_infected'
        },
        'image_ids': ['IMG_001.jpg']
    }

    # Format prompt
    prompt = format_woundcare_evaluation_prompt_with_metadata(
        qa_pair=example_qa,
        candidate_response="Apply antibiotic cream and keep it clean."
    )

    print(prompt)
