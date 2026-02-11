"""
Perturbation functions specific to radiology reports (RadEval dataset).
"""

import json
import os
import random
import re
import spacy
from typing import Dict, Tuple

# Load spacy model for sentence segmentation
nlp = spacy.load('en_core_web_sm')

# Load rexerr prompts (cached)
_REXERR_COMBINED_PROMPTS = {}


def _load_rexerr_combined_prompt(error_type: int):
    """Load combined rexerr prompt for specific error type (cached)."""
    global _REXERR_COMBINED_PROMPTS

    if error_type not in _REXERR_COMBINED_PROMPTS:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        error_names = {
            5: 'False_Prediction',
            10: 'Add_Contradiction',
            11: 'False_Negation'
        }

        error_name = error_names[error_type]
        prompt_path = os.path.join(
            script_dir, 'prompts', 'combined_rexerr_prompts',
            f'error_type_{error_type}_{error_name}_COMBINED.txt'
        )

        with open(prompt_path, 'r') as f:
            # Skip the first two lines (header and separator)
            lines = f.readlines()
            _REXERR_COMBINED_PROMPTS[error_type] = ''.join(lines[2:])

    return _REXERR_COMBINED_PROMPTS[error_type]


def remove_sentences_by_percentage(text, percentage=0.3):
    """
    Remove a percentage of sentences from radiology report.
    Same as CQA eval implementation.
    """
    doc = nlp(text)
    text_sentences = list(doc.sents)
    total_sentences = len(text_sentences)

    if total_sentences < 2:
        return text

    num_to_remove = max(1, int(total_sentences * percentage))
    num_to_remove = min(num_to_remove, total_sentences - 1)

    indices_to_remove = random.sample(range(total_sentences), num_to_remove)
    indices_to_remove.sort(reverse=True)

    for idx in indices_to_remove:
        text_sentences.pop(idx)

    modified_text = ' '.join([sent.text.strip() for sent in text_sentences])
    return modified_text


def add_typos(text, typo_probability=0.5):
    """
    Add typos to radiology report by swapping adjacent characters.
    Same as CQA eval implementation.
    """
    words = text.split()
    modified_words = []

    for word in words:
        # Only apply typos to words longer than 3 characters
        if len(word) > 3 and random.random() < typo_probability:
            # Pick a random position (not first or last character)
            pos = random.randint(1, len(word) - 2)
            # Swap with next character
            word_list = list(word)
            word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
            modified_words.append(''.join(word_list))
        else:
            modified_words.append(word)

    return ' '.join(modified_words)


def swap_qualifiers(text):
    """
    Swap qualifiers to their opposites in radiology reports.

    Swaps:
    - Left <-> Right
    - Mild <-> Severe
    - Low <-> High
    - Small <-> Large
    - Absent <-> Present
    - Minimal <-> Extensive
    - Decreased <-> Increased
    """
    # Define swap pairs (case-insensitive matching)
    swap_pairs = [
        (r'\b(left)\b', r'\b(right)\b'),
        (r'\b(mild)\b', r'\b(severe)\b'),
        (r'\b(low)\b', r'\b(high)\b'),
        (r'\b(small)\b', r'\b(large)\b'),
        (r'\b(absent)\b', r'\b(present)\b'),
        (r'\b(minimal)\b', r'\b(extensive)\b'),
        (r'\b(decreased)\b', r'\b(increased)\b'),
        (r'\b(no)\b', r'\b(multiple)\b'),
        (r'\b(normal)\b', r'\b(abnormal)\b'),
    ]

    # Create a modified copy
    modified_text = text
    changes = []

    for pattern1, pattern2 in swap_pairs:
        # Find all matches for both patterns
        matches1 = list(re.finditer(pattern1, modified_text, re.IGNORECASE))
        matches2 = list(re.finditer(pattern2, modified_text, re.IGNORECASE))

        # Replace from end to start to preserve indices
        all_matches = []
        for match in matches1:
            all_matches.append((match.start(), match.end(), match.group(1), pattern2.strip(r'\b()').strip()))
        for match in matches2:
            all_matches.append((match.start(), match.end(), match.group(1), pattern1.strip(r'\b()').strip()))

        # Sort by position (reverse order)
        all_matches.sort(reverse=True, key=lambda x: x[0])

        # Apply replacements
        for start, end, original, replacement in all_matches:
            # Preserve original case
            if original.isupper():
                replacement = replacement.upper()
            elif original[0].isupper():
                replacement = replacement.capitalize()

            modified_text = modified_text[:start] + replacement + modified_text[end:]
            changes.append(f"{original} -> {replacement}")

    return modified_text, {'changes': changes, 'num_changes': len(changes)}


def swap_organs(text):
    """
    Swap organs/anatomical locations in radiology reports.

    Examples:
    - Lung <-> Heart
    - Liver <-> Kidney
    - Chest <-> Abdomen
    """
    # Define organ/anatomical swap pairs
    organ_pairs = [
        (r'\b(lung|lungs|pulmonary)\b', r'\b(heart|cardiac)\b'),
        (r'\b(liver|hepatic)\b', r'\b(kidney|kidneys|renal)\b'),
        (r'\b(chest|thorax|thoracic)\b', r'\b(abdomen|abdominal)\b'),
        (r'\b(brain|cerebral)\b', r'\b(spinal cord|spine)\b'),
        (r'\b(stomach|gastric)\b', r'\b(intestine|intestinal|bowel)\b'),
    ]

    modified_text = text
    changes = []

    for pattern1, pattern2 in organ_pairs:
        # Find matches
        matches1 = list(re.finditer(pattern1, modified_text, re.IGNORECASE))
        matches2 = list(re.finditer(pattern2, modified_text, re.IGNORECASE))

        # Collect all swaps
        all_matches = []
        for match in matches1:
            # Use first alternative from pattern2 as replacement
            replacement = pattern2.strip(r'\b()').split('|')[0]
            all_matches.append((match.start(), match.end(), match.group(0), replacement))
        for match in matches2:
            replacement = pattern1.strip(r'\b()').split('|')[0]
            all_matches.append((match.start(), match.end(), match.group(0), replacement))

        # Sort reverse order
        all_matches.sort(reverse=True, key=lambda x: x[0])

        # Apply replacements
        for start, end, original, replacement in all_matches:
            # Preserve case
            if original.isupper():
                replacement = replacement.upper()
            elif original[0].isupper():
                replacement = replacement.capitalize()

            modified_text = modified_text[:start] + replacement + modified_text[end:]
            changes.append(f"{original} -> {replacement}")

    return modified_text, {'changes': changes, 'num_changes': len(changes)}


def _parse_rexerr_json_response(response_text: str, original_report: str) -> Tuple[str, Dict]:
    """
    Parse JSON response from rexerr-style perturbation.

    Expected format (unified with MedInfo):
    {
      "modified_report": "...",
      "changes_detail": [
        {
          "sentence_index": 1,
          "original": "...",
          "modified": "...",
          "explanation": "...",
          "severity": "",
          "harm_potential": ""
        }
      ]
    }

    The changes_detail array only contains changed sentences (not all sentences).

    Returns:
        (perturbed_report, metadata)
    """
    try:
        # Try to extract JSON from response (may have markdown code fences)
        json_match = re.search(r'\{[\s\S]*"modified_report"[\s\S]*\}', response_text)

        if not json_match:
            # Try without the field name requirement
            json_match = re.search(r'\{[\s\S]*\}', response_text)

        if not json_match:
            return original_report, {
                'error': 'No JSON found in response',
                'raw_response': response_text
            }

        # Parse JSON
        result = json.loads(json_match.group(0))

        # Extract modified report
        perturbed_report = result.get('modified_report', original_report)

        # Extract changes_detail (unified field name)
        changes_list = result.get('changes_detail', [])

        # Build summary - all entries in changes_list are actual changes
        explanations = [c.get('explanation', '') for c in changes_list if c.get('explanation')]

        return perturbed_report, {
            'changes_detail': changes_list,
            'num_changes': len(changes_list),
            'parsed_successfully': True,
            'level': 'report'  # RadEval operates at report level
        }

    except json.JSONDecodeError as e:
        return original_report, {
            'error': f'JSON decode error: {str(e)}',
            'raw_response': response_text
        }
    except Exception as e:
        return original_report, {
            'error': f'Failed to parse response: {str(e)}',
            'raw_response': response_text
        }


def llm_inject_error(
    text: str,
    error_type: int,
    model: str = 'gpt-4.1',
    max_retries: int = 5
) -> Tuple[str, Dict]:
    """
    Use LLM to inject errors into radiology report using rexerr prompts with JSON output.

    Args:
        text: Radiology report text
        error_type: Error type ID (5, 10, or 11 supported)
        model: LLM model to use (default: gpt-4.1)
        max_retries: Number of retry attempts

    Returns:
        (perturbed_text, metadata)
    """
    from helpers.multi_llm_inference import get_response

    # Map error types to names
    error_names = {
        5: "False prediction",
        10: "Add contradiction",
        11: "False negation"
    }

    if error_type not in error_names:
        raise ValueError(f"Error type {error_type} not supported. Use 5, 10, or 11.")

    # Load combined prompt (system + error-specific instructions + JSON format)
    system_prompt = _load_rexerr_combined_prompt(error_type)

    # Build user message with the report
    user_message = f"Here is the radiology report to modify:\n\n{text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Try to get valid response
    for attempt in range(max_retries):
        try:
            response = get_response(messages, model=model)

            # Parse JSON response
            perturbed_text, metadata = _parse_rexerr_json_response(response, text)

            # Check if parsing was successful
            if 'error' in metadata:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1}: Parse error, retrying...")
                    # Add feedback message for next attempt
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "The JSON format was invalid. Please provide a valid JSON response with the exact structure specified."})
                    continue
                else:
                    metadata['skip_reason'] = 'Failed to parse response after max retries'
                    return text, metadata

            # Check if perturbation was successful
            if perturbed_text == text or metadata.get('num_changes', 0) == 0:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1}: No changes detected, retrying...")
                    # Add feedback message for next attempt
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "You did not make any changes to the report. Please inject errors into the report as instructed. Make sure to modify sentences according to the error type specified."})
                    continue
                else:
                    metadata['skip_reason'] = 'No changes after max retries'

            metadata['error_type'] = error_type
            metadata['error_name'] = error_names[error_type]
            metadata['model'] = model
            return perturbed_text, metadata

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}: Error - {e}, retrying...")
                continue
            else:
                return text, {
                    'error': str(e),
                    'error_type': error_type,
                    'error_name': error_names[error_type],
                    'model': model,
                    'skip_reason': f'Failed after {max_retries} attempts'
                }

    return text, {
        'error_type': error_type,
        'error_name': error_names[error_type],
        'model': model,
        'skip_reason': 'Max retries exceeded'
    }


def inject_false_prediction(text: str, model: str = 'gpt-4.1') -> Tuple[str, Dict]:
    """
    Error type 5: Add false predictions (findings not present in report).

    Args:
        text: Radiology report text
        model: LLM model to use

    Returns:
        (perturbed_text, metadata)
    """
    return llm_inject_error(text, error_type=5, model=model)


def inject_contradiction(text: str, model: str = 'gpt-4.1') -> Tuple[str, Dict]:
    """
    Error type 10: Add contradictions (opposite statements within report).

    Args:
        text: Radiology report text
        model: LLM model to use

    Returns:
        (perturbed_text, metadata)
    """
    return llm_inject_error(text, error_type=10, model=model)


def inject_false_negation(text: str, model: str = 'gpt-4.1') -> Tuple[str, Dict]:
    """
    Error type 11: Change positive findings to negative (remove findings).

    Args:
        text: Radiology report text
        model: LLM model to use

    Returns:
        (perturbed_text, metadata)
    """
    return llm_inject_error(text, error_type=11, model=model)
