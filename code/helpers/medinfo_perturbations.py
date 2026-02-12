"""
Perturbation functions specific to MedInfo medication QA dataset.

LLM-based perturbations that inject critical and non-critical medication errors.
"""

import json
import os
from typing import Dict, Tuple


# Load MedInfo prompts (cached)
_MEDINFO_PROMPTS = {}


def _load_medinfo_prompt(error_type: str, level: str):
    """Load MedInfo prompt for specific error type and level (cached)."""
    global _MEDINFO_PROMPTS

    key = f"{error_type}_{level}"

    if key not in _MEDINFO_PROMPTS:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        prompt_path = os.path.join(
            script_dir, 'prompts', 'medinfo_prompts',
            f'medinfo_{error_type}_{level}.txt'
        )

        with open(prompt_path, 'r') as f:
            _MEDINFO_PROMPTS[key] = f.read()

    return _MEDINFO_PROMPTS[key]


def _parse_medinfo_json_response(response_text: str, original_answer: str) -> Tuple[str, Dict]:
    """
    Parse JSON response from MedInfo perturbation.

    Expected format:
    {
        "modified_answer": "...",
        "changes_detail": [
            {
                "change_index": 0,
                "original": "...",
                "modified": "...",
                "explanation": "...",
                "severity": "critical",
                "harm_potential": "..."
            }
        ]
    }

    Args:
        response_text: Raw LLM response
        original_answer: Original answer text

    Returns:
        (perturbed_answer, metadata)
    """
    try:
        # Try to extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            return original_answer, {
                'error': 'No JSON found in response',
                'raw_response': response_text[:500],
                'parsed_successfully': False
            }

        json_str = response_text[json_start:json_end]
        data = json.loads(json_str)

        # Extract modified answer
        perturbed_answer = data.get('modified_answer', '').strip()

        # Extract changes_detail (array)
        changes_detail = data.get('changes_detail', [])

        if not perturbed_answer:
            return original_answer, {
                'error': 'Empty modified_answer in JSON',
                'raw_response': response_text[:500],
                'parsed_successfully': False
            }

        # Return perturbed answer and metadata
        metadata = {
            'changes_detail': changes_detail,
            'parsed_successfully': True,
            'num_changes': len(changes_detail) if changes_detail else (1 if perturbed_answer != original_answer else 0)
        }

        return perturbed_answer, metadata

    except json.JSONDecodeError as e:
        return original_answer, {
            'error': f'JSON decode error: {str(e)}',
            'raw_response': response_text[:500],
            'parsed_successfully': False
        }
    except Exception as e:
        return original_answer, {
            'error': f'Unexpected error: {str(e)}',
            'raw_response': response_text[:500],
            'parsed_successfully': False
        }


def llm_inject_medinfo_error(
    answer: str,
    error_type: str,
    level: str,
    model: str = 'gpt-4.1',
    max_retries: int = 5
) -> Tuple[str, Dict]:
    """
    Use LLM to inject medication errors into MedInfo answers.

    Args:
        answer: Original answer text
        error_type: Either 'critical_error' or 'noncritical_error'
        level: Either 'coarse' or 'fine'
        model: LLM model to use (default: gpt-4.1)
        max_retries: Number of retry attempts

    Returns:
        (perturbed_answer, metadata)
    """
    from helpers.multi_llm_inference import get_response

    error_names = {
        'critical_error': 'Critical medication error (10x-50x overdose)',
        'noncritical_error': 'Non-critical medication error (2-3x typical dose)'
    }

    if error_type not in error_names:
        raise ValueError(f"Error type {error_type} not supported. Use 'critical_error' or 'noncritical_error'.")

    # Load prompt
    system_prompt = _load_medinfo_prompt(error_type, level)

    # Build user message with the answer
    user_message = f"Here is the medication answer to modify:\n\n{answer}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Try to get valid response
    for attempt in range(max_retries):
        try:
            response = get_response(messages, model=model)

            # Parse JSON response
            perturbed_answer, metadata = _parse_medinfo_json_response(response, answer)

            # Check if parsing was successful
            if 'error' in metadata:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1}: Parse error, retrying...")
                    # Add feedback message for next attempt
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "The JSON format was invalid. Please provide a valid JSON response with all required fields as specified in the prompt."})
                    continue
                else:
                    metadata['skip_reason'] = 'Failed to parse response after max retries'
                    return answer, metadata

            # Check if perturbation was successful
            if perturbed_answer == answer or metadata.get('num_changes', 0) == 0:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1}: No changes detected, retrying...")
                    # Add feedback message for next attempt
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "You did not make any changes to the answer. Please inject medication errors as instructed. Make sure to modify the dosage or other medication details according to the error type specified."})
                    continue
                else:
                    metadata['skip_reason'] = 'No changes after max retries'

            metadata['level'] = level
            metadata['model'] = model
            return perturbed_answer, metadata

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}: Error - {e}, retrying...")
                continue
            else:
                return answer, {
                    'error': str(e),
                    'level': level,
                    'model': model,
                    'skip_reason': f'Failed after {max_retries} attempts'
                }

    return answer, {
        'level': level,
        'model': model,
        'skip_reason': 'Max retries exceeded'
    }


def inject_critical_error(answer: str, level: str = 'coarse', model: str = 'gpt-4.1') -> Tuple[str, Dict]:
    """
    Inject critical medication errors (10x-50x overdoses, life-threatening).

    Args:
        answer: Original medication answer text
        level: 'coarse' or 'fine' (determines which prompt to use)
        model: LLM model to use

    Returns:
        (perturbed_answer, metadata)
    """
    return llm_inject_medinfo_error(answer, error_type='critical_error', level=level, model=model)


def inject_noncritical_error(answer: str, level: str = 'coarse', model: str = 'gpt-4.1') -> Tuple[str, Dict]:
    """
    Inject non-critical medication errors (2-3x typical dose, unusual but safe).

    Args:
        answer: Original medication answer text
        level: 'coarse' or 'fine' (determines which prompt to use)
        model: LLM model to use

    Returns:
        (perturbed_answer, metadata)
    """
    return llm_inject_medinfo_error(answer, error_type='noncritical_error', level=level, model=model)
