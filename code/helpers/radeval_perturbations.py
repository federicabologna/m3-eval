"""
Perturbation functions specific to radiology reports (RadEval dataset).
"""

import random
import re
import spacy

# Load spacy model for sentence segmentation
nlp = spacy.load('en_core_web_sm')


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
