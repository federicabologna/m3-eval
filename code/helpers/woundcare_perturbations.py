"""
WoundCare answer perturbations - modify clinical terms within answer text.

These functions find and swap clinical terminology in the GPT-4o responses.
"""

import random
import re
from typing import Dict, Tuple, List


# Anatomical locations from data dictionary
ANATOMIC_LOCATIONS = [
    'abdomen', 'ankle', 'arm', 'armpit', 'back', 'backofhead', 'chest', 'chin',
    'ear', 'elbow', 'eyeregion', 'face', 'fingernail', 'fingers', 'foot',
    'foot-sole', 'forearm', 'forehead', 'groin', 'hand', 'hand-back', 'heel',
    'knee', 'lips', 'lowerback', 'lowerleg', 'mouth', 'napeofneck', 'neck',
    'nose', 'palm', 'scalp', 'shoulder', 'thigh', 'toenail', 'toes',
    'tongue', 'wrist'
]

# Normalize multi-word locations for matching
LOCATION_VARIANTS = {
    'finger': ['finger', 'fingers'],
    'toe': ['toe', 'toes'],
    'foot': ['foot'],
    'hand': ['hand'],
    'arm': ['arm'],
    'leg': ['leg', 'lowerleg'],
    'back': ['back', 'lowerback'],
    'forehead': ['forehead'],
    'scalp': ['scalp'],
    'ankle': ['ankle'],
    'knee': ['knee'],
    'elbow': ['elbow'],
    'wrist': ['wrist'],
    'shoulder': ['shoulder'],
    'chest': ['chest'],
    'abdomen': ['abdomen'],
    'neck': ['neck'],
    'face': ['face'],
    'nose': ['nose'],
    'ear': ['ear'],
    'chin': ['chin'],
    'lips': ['lips'],
    'mouth': ['mouth'],
    'palm': ['palm'],
    'heel': ['heel'],
    'thigh': ['thigh'],
    'groin': ['groin'],
}

# Infection term swaps - grammatically correct, medically incorrect
# Ordered from most specific to least specific to avoid partial matches
INFECTION_SWAPS = [
    # Specific phrases - handle these first
    (r'\bwatch for (?:any )?signs of infection\b', 'watch for continued healing'),
    (r'\bmonitor for (?:any )?signs of infection\b', 'continue monitoring the healing'),
    (r'\bif you notice (?:any )?signs of infection\b', 'as there are no signs of infection'),
    (r'\bcould indicate (?:an? )?infection\b', 'do not indicate infection'),
    (r'\bmay indicate (?:an? )?infection\b', 'do not indicate infection'),
    (r'\bsuggests? (?:an? )?infection\b', 'do not suggest infection'),
    (r'\bshows? signs of infection\b', 'shows no signs of infection'),
    (r'\bno signs of infection\b', 'signs of infection'),
    (r'\bsigns of infection\b', 'no signs of infection'),

    # Specific infection states
    (r'\bappears? infected\b', 'does not appear infected'),
    (r'\bdoes not appear infected\b', 'appears infected'),
    (r'\bmay be infected\b', 'is not infected'),
    (r'\bis not infected\b', 'is infected'),
    (r'\bis infected\b', 'is not infected'),
    (r'\bnot infected\b', 'infected'),

    # General infection references with articles
    (r'\ban infection\b', 'no infection'),
    (r'\bthe infection\b', 'no infection'),

    # Bare "infected" as adjective (e.g., "wound infected")
    (r'\binfected\b', 'not infected'),

    # Bare "infection" as noun - only swap if not already handled
    (r'\binfection\b', 'healing'),
]

# Time/frequency swaps - grammatically correct, medically incorrect
TIME_FREQUENCY_SWAPS = [
    # Specific time phrases
    (r'\btwice (?:a )?daily\b', 'once per week'),
    (r'\btwice a day\b', 'once per week'),
    (r'\bdaily\b', 'weekly'),
    (r'\bevery day\b', 'every week'),
    (r'\beach day\b', 'each week'),
    (r'\bonce a day\b', 'once per month'),
    (r'\bonce daily\b', 'once per month'),
    (r'\bevery (\d+)-(\d+) hours\b', r'every \1-\2 weeks'),
    (r'\bwithin 24 hours\b', 'within 2-3 weeks'),
    (r'\bwithin (\d+) hours\b', r'within \1 weeks'),
    (r'\bin (\d+)-(\d+) days\b', r'in \1-\2 months'),
    (r'\bafter (\d+)-(\d+) days\b', r'after \1-\2 months'),
    (r'\bfor (\d+)-(\d+) days\b', r'for \1-\2 months'),
    (r'\b(\d+)-(\d+) weeks\b', r'\1-\2 months'),
    (r'\bimmediately\b', 'in a few days'),
    (r'\bright away\b', 'when convenient'),
    (r'\bas soon as possible\b', 'at your convenience'),
    (r'\bpromptly\b', 'eventually'),
]

# Urgency swaps - urgent → not urgent only (grammatically correct, medically incorrect)
URGENCY_SWAPS = [
    # Emergency/urgent care phrases
    (r'\bseek (?:immediate|urgent|emergency) (?:medical )?(?:care|attention|treatment)\b', 'monitor at home'),
    (r'\bgo to (?:the )?(?:ER|emergency room|emergency department)\b', 'schedule a routine appointment'),
    (r'\bcall (?:your doctor|a doctor|911) (?:immediately|right away|urgently)\b', 'mention to your doctor at your next visit'),
    (r'\brequires? (?:immediate|urgent|emergency) (?:medical )?(?:care|attention|treatment)\b', 'can be monitored at home'),
    (r'\bneeds? (?:immediate|urgent|emergency) (?:medical )?(?:care|attention|treatment)\b', 'can wait for routine care'),
    (r'\bthis is (?:an )?emergency\b', 'this is not urgent'),
    (r'\bmedical emergency\b', 'routine medical matter'),
    (r'\bseek care immediately\b', 'seek care when convenient'),
    (r'\bget medical help (?:immediately|right away|urgently)\b', 'follow up with your doctor'),
]

# Severity swaps - serious → minor only (one direction, grammatically correct, medically incorrect)
SEVERITY_SWAPS = [
    # Severity descriptors - only downgrade severity
    (r'\bserious\b', 'minor'),
    (r'\bsevere\b', 'mild'),
    (r'\bconcerning\b', 'normal'),
    (r'\bworrisome\b', 'typical'),
    (r'\balarming\b', 'common'),
    (r'\bsignificant\b', 'minor'),
    (r'\bdeep (?:wound|cut|laceration)\b', 'superficial wound'),
    (r'\bmajor\b', 'minor'),
]


def swap_anatomic_location_in_answer(answer: str, seed: int = None) -> Tuple[str, bool, Dict]:
    """
    Find and swap anatomical location terms in answer text.
    Uses pattern matching to preserve grammar (articles, prepositions, etc.).

    Args:
        answer: GPT-4o answer text
        seed: Random seed

    Returns:
        (perturbed_answer, success, metadata)
    """
    if seed is not None:
        random.seed(seed)

    # Build patterns for each location
    # Look for: "your [location]", "the [location]", "[location] wound", etc.
    found_matches = []

    for location_key, variants in LOCATION_VARIANTS.items():
        for variant in variants:
            # Create regex pattern with word boundaries
            pattern = r'\b' + re.escape(variant) + r'\b'

            for match in re.finditer(pattern, answer, re.IGNORECASE):
                found_matches.append({
                    'location_key': location_key,
                    'matched_text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

    if not found_matches:
        return answer, False, {'reason': 'No anatomical locations found'}

    # Sort by position and take the first one
    found_matches.sort(key=lambda x: x['start'])
    first_match = found_matches[0]

    # Pick a different location
    other_locations = [k for k in LOCATION_VARIANTS.keys() if k != first_match['location_key']]
    if not other_locations:
        return answer, False, {'reason': 'No alternative locations'}

    new_location_key = random.choice(other_locations)
    new_text = LOCATION_VARIANTS[new_location_key][0]  # Use first variant

    # Preserve capitalization
    if first_match['matched_text'][0].isupper():
        new_text = new_text.capitalize()

    # Handle singular/plural matching if the original was plural
    original_text = first_match['matched_text']
    if original_text.lower().endswith('s') and not new_text.endswith('s'):
        # Check if the new location has a plural form
        plural_variants = [v for v in LOCATION_VARIANTS[new_location_key] if v.endswith('s')]
        if plural_variants:
            new_text = plural_variants[0]
            if original_text[0].isupper():
                new_text = new_text.capitalize()

    perturbed_answer = answer[:first_match['start']] + new_text + answer[first_match['end']:]

    metadata = {
        'original_term': original_text,
        'original_key': first_match['location_key'],
        'new_term': new_text,
        'new_key': new_location_key,
        'position': first_match['start']
    }

    return perturbed_answer, True, metadata


def swap_infection_in_answer(answer: str, seed: int = None) -> Tuple[str, bool, Dict]:
    """
    Find and swap infection status terms in answer text.
    Uses pattern matching for grammatically correct replacements.
    """
    if seed is not None:
        random.seed(seed)

    # Try each pattern
    for pattern, replacement in INFECTION_SWAPS:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            original_text = match.group(0)
            start, end = match.span()

            # Preserve capitalization of first letter
            if original_text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]

            perturbed_answer = answer[:start] + replacement + answer[end:]

            metadata = {
                'original_term': original_text,
                'new_term': replacement,
                'position': start,
                'pattern': pattern
            }

            return perturbed_answer, True, metadata

    return answer, False, {'reason': 'No infection terms found'}


def swap_time_frequency_in_answer(answer: str, seed: int = None) -> Tuple[str, bool, Dict]:
    """
    Find and swap time/frequency terms in answer text.
    Uses pattern matching for grammatically correct replacements.
    """
    if seed is not None:
        random.seed(seed)

    # Try each pattern
    for pattern, replacement in TIME_FREQUENCY_SWAPS:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            original_text = match.group(0)
            start, end = match.span()

            # Preserve capitalization of first letter
            if original_text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]

            perturbed_answer = answer[:start] + replacement + answer[end:]

            metadata = {
                'original_term': original_text,
                'new_term': replacement,
                'position': start,
                'pattern': pattern
            }

            return perturbed_answer, True, metadata

    return answer, False, {'reason': 'No time/frequency terms found'}


def swap_urgency_in_answer(answer: str, seed: int = None) -> Tuple[str, bool, Dict]:
    """
    Find and swap urgency terms in answer text (urgent → not urgent only).
    Uses pattern matching for grammatically correct replacements.
    """
    if seed is not None:
        random.seed(seed)

    # Try each pattern
    for pattern, replacement in URGENCY_SWAPS:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            original_text = match.group(0)
            start, end = match.span()

            # Preserve capitalization of first letter
            if original_text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]

            perturbed_answer = answer[:start] + replacement + answer[end:]

            metadata = {
                'original_term': original_text,
                'new_term': replacement,
                'position': start,
                'pattern': pattern
            }

            return perturbed_answer, True, metadata

    return answer, False, {'reason': 'No urgency terms found'}


def swap_severity_in_answer(answer: str, seed: int = None) -> Tuple[str, bool, Dict]:
    """
    Find and swap severity terms in answer text (serious → minor only).
    Uses pattern matching for grammatically correct replacements.
    Only downgrades severity, never upgrades.
    """
    if seed is not None:
        random.seed(seed)

    # Try each pattern
    for pattern, replacement in SEVERITY_SWAPS:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            original_text = match.group(0)
            start, end = match.span()

            # Preserve capitalization of first letter
            if original_text[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]

            perturbed_answer = answer[:start] + replacement + answer[end:]

            metadata = {
                'original_term': original_text,
                'new_term': replacement,
                'position': start,
                'pattern': pattern
            }

            return perturbed_answer, True, metadata

    return answer, False, {'reason': 'No severity terms found'}


def apply_woundcare_answer_perturbation(
    answer: str,
    perturbation_type: str,
    seed: int = None
) -> Tuple[str, bool, Dict]:
    """
    Apply perturbation to answer text.

    Args:
        answer: Answer text to perturb
        perturbation_type: Type of perturbation
        seed: Random seed

    Returns:
        (perturbed_answer, success, metadata)
    """
    if perturbation_type == 'swap_infection':
        return swap_infection_in_answer(answer, seed)
    elif perturbation_type == 'swap_anatomic_location':
        return swap_anatomic_location_in_answer(answer, seed)
    elif perturbation_type == 'swap_time_frequency':
        return swap_time_frequency_in_answer(answer, seed)
    elif perturbation_type == 'swap_urgency':
        return swap_urgency_in_answer(answer, seed)
    elif perturbation_type == 'swap_severity':
        return swap_severity_in_answer(answer, seed)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


# Test function
if __name__ == '__main__':
    # Test examples
    test_answers = [
        "The wound on your finger appears infected. Clean it daily and watch for signs of infection.",
        "This is a serious wound. Seek immediate medical attention right away.",
        "Clean the cut on your hand twice daily. Monitor for infection over the next 1-2 days.",
        "The deep wound on your forehead is concerning. Go to the ER within 24 hours if it worsens."
    ]

    print("Testing WoundCare Answer Perturbations:")
    print("=" * 80)

    perturbation_types = [
        'swap_infection',
        'swap_anatomic_location',
        'swap_time_frequency',
        'swap_urgency',
        'swap_severity'
    ]

    for i, answer in enumerate(test_answers, 1):
        print(f"\nExample {i}:")
        print(f"Original: {answer}")

        # Try each perturbation
        for pert_type in perturbation_types:
            perturbed, success, metadata = apply_woundcare_answer_perturbation(answer, pert_type, seed=42)

            if success:
                print(f"\n{pert_type}:")
                print(f"  Perturbed: {perturbed}")
                print(f"  Changed: '{metadata['original_term']}' → '{metadata['new_term']}'")
            else:
                print(f"\n{pert_type}: {metadata.get('reason', 'Failed')}")
