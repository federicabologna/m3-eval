import spacy
import random
import re
import Levenshtein

# Load scispacy model for medical entity recognition
# This provides better recognition of medical terms, diseases, drugs, etc.
try:
    nlp = spacy.load("en_ner_bc5cdr_md")  # Medical NER model (recognizes diseases and chemicals/drugs)
except OSError:
    print("Scispacy model not found. Install with: pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")
    print("Falling back to standard spacy model...")
    nlp = spacy.load("en_core_web_sm")


def modify_dosage(match):
    number = float(match.group(1))
    space = ' ' if match.group(2) else ''
    unit = match.group(3)

    # Randomly choose to multiply or divide by 10
    if random.choice([True, False]):
        new_number = number * 10
    else:
        new_number = number / 10

    # Format the number (remove unnecessary decimals)
    if new_number.is_integer():
        new_number_str = str(int(new_number))
    else:
        new_number_str = str(new_number)

    return new_number_str + space + unit


def modify_time_interval(match):
    """Modify time intervals like 'every 4-6 hours'."""
    prefix = match.group(1)  # "every "
    first_num = int(match.group(2))
    separator = match.group(3)  # "-" or " to "
    second_num = int(match.group(4))
    suffix = match.group(5)  # " hours"

    # Randomly choose to multiply or divide by 2
    if random.choice([True, False]):
        new_first = first_num * 2
        new_second = second_num * 2
    else:
        new_first = max(1, first_num // 2)
        new_second = max(1, second_num // 2)

    return f"{prefix}{new_first}{separator}{new_second}{suffix}"


def modify_anatomical_count(match):
    """Modify anatomical references like 'both eyes' or 'each nostril'."""
    quantifier = match.group(1)  # "both", "each", "one", "the"
    body_part = match.group(2)  # "eye", "eyes", "nostril", etc.

    # Map of transformations
    if quantifier.lower() == "both":
        new_quantifier = random.choice(["one", "each"])
    elif quantifier.lower() == "each":
        new_quantifier = random.choice(["both", "one"])
    elif quantifier.lower() == "one":
        new_quantifier = random.choice(["both", "each"])
    elif quantifier.lower() == "the":
        new_quantifier = random.choice(["both", "each"])
    else:
        new_quantifier = quantifier

    # Preserve capitalization
    if quantifier[0].isupper():
        new_quantifier = new_quantifier.capitalize()

    return f"{new_quantifier} {body_part}"


def modify_administration_instruction(match):
    """Flip administration instructions like swallow/chew."""
    negation = match.group(1)  # "don't", "do not", "avoid", etc. (or empty)
    instruction = match.group(2).lower()  # "swallow", "chew", "crush", etc.
    rest = match.group(3) if match.lastindex >= 3 else ""  # the rest like " the tablet"

    # Instruction opposites
    opposites = {
        'swallow': 'chew',
        'chew': 'swallow',
        'crush': 'take whole',
        'dissolve': 'swallow',
        'suck': 'swallow',
    }

    # If there's a negation, remove it (just use the instruction)
    if negation:
        # "don't swallow" → "swallow"
        result = instruction
    else:
        # "swallow" → "chew"
        result = opposites.get(instruction, instruction)

    # Preserve capitalization of first character
    if match.group(0)[0].isupper():
        result = result.capitalize()

    return result + rest


def change_dosage(text):
    # Common dosage units
    dosage_units = r'(mg|mcg|g|mL|ml|L|l|units?|IU|cc|drops?|tablets?|caps?|capsules?|tsp|tbsp|oz)'

    # Pattern to match dosages: number (with optional decimal) followed by optional space and unit
    dosage_pattern = r'\b(\d+(?:\.\d+)?)(\s*)(' + dosage_units + r')\b'

    # Pattern to match time intervals: "every 4-6 hours/minutes/days/weeks"
    time_pattern = r'\b(every\s+)(\d+)([\s-](?:to\s+)?)(\d+)(\s+(?:hours?|minutes?|days?|weeks?|months?))\b'

    # Pattern to match anatomical references
    body_parts = r'(eyes?|ears?|nostrils?|hands?|arms?|legs?|feet?|knees?|elbows?|cheeks?)'
    anatomical_pattern = r'\b(both|each|one|the)\s+(' + body_parts + r')\b'

    # Pattern to match administration instructions with optional negation
    # Matches: "swallow", "chew", "don't swallow", "do not chew", "avoid crushing", etc.
    admin_pattern = r'\b(don\'t\s+|do\s+not\s+|avoid\s+)?(swallow|chew|crush|dissolve|suck)(\s+(?:the\s+)?(?:tablet|pill|capsule|medication|medicine)s?)?\b'

    # Find all matches for all patterns
    all_matches = []

    for match in re.finditer(dosage_pattern, text, flags=re.IGNORECASE):
        all_matches.append(('dosage', match))

    for match in re.finditer(time_pattern, text, flags=re.IGNORECASE):
        all_matches.append(('time', match))

    for match in re.finditer(anatomical_pattern, text, flags=re.IGNORECASE):
        all_matches.append(('anatomical', match))

    for match in re.finditer(admin_pattern, text, flags=re.IGNORECASE):
        all_matches.append(('admin', match))

    # If no matches found, return original text
    if not all_matches:
        return text, {'dosage': 0, 'time': 0, 'anatomical': 0, 'admin': 0}

    # Randomly select up to 3 changes
    num_changes = min(3, len(all_matches))
    selected_matches = random.sample(all_matches, num_changes)

    # Sort by position (descending) to maintain indices when replacing
    selected_matches.sort(key=lambda x: x[1].start(), reverse=True)

    # Apply changes and count by category
    modified_text = text
    change_counts = {'dosage': 0, 'time': 0, 'anatomical': 0, 'admin': 0}

    for category, match in selected_matches:
        # Apply the appropriate modification function
        if category == 'dosage':
            replacement = modify_dosage(match)
        elif category == 'time':
            replacement = modify_time_interval(match)
        elif category == 'anatomical':
            replacement = modify_anatomical_count(match)
        elif category == 'admin':
            replacement = modify_administration_instruction(match)

        # Replace the text
        modified_text = modified_text[:match.start()] + replacement + modified_text[match.end():]
        change_counts[category] += 1

    return modified_text, change_counts
    


def add_confusion(text):
    # Process the text with spacy
    doc = nlp(text)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    # Randomly shuffle the sentences
    random.shuffle(sentences)

    # Join sentences back together
    shuffled_text = " ".join(sentences)

    return shuffled_text


def add_typos(text, typo_probability=0.5):
    # Keyboard proximity map for realistic typos (QWERTY layout)
    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcsx', 'e': 'wrds',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yugjbn', 'i': 'uojk', 'j': 'uikmnh',
        'k': 'iolmj', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iplk',
        'p': 'ol', 'q': 'wa', 'r': 'etfd', 's': 'wedxza', 't': 'ryfg',
        'u': 'yihj', 'v': 'cfgb', 'w': 'qesa', 'x': 'zsdc', 'y': 'tugh',
        'z': 'asx'
    }

    def apply_typo(word):
        if len(word) <= 2:
            return word

        typo_type = random.choice(['swap', 'delete', 'duplicate', 'substitute'])
        word_list = list(word)

        if typo_type == 'swap' and len(word) > 2:
            # Swap two adjacent characters
            pos = random.randint(0, len(word) - 2)
            word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]

        elif typo_type == 'delete':
            # Delete a random character
            pos = random.randint(0, len(word) - 1)
            word_list.pop(pos)

        elif typo_type == 'duplicate':
            # Duplicate a random character
            pos = random.randint(0, len(word) - 1)
            word_list.insert(pos, word_list[pos])

        elif typo_type == 'substitute':
            # Substitute with a nearby key
            pos = random.randint(0, len(word) - 1)
            char = word_list[pos].lower()
            if char in keyboard_neighbors and keyboard_neighbors[char]:
                new_char = random.choice(keyboard_neighbors[char])
                # Preserve case
                if word_list[pos].isupper():
                    new_char = new_char.upper()
                word_list[pos] = new_char

        return ''.join(word_list)

    # Tokenize text using spacy/scispacy
    doc = nlp(text)

    # Identify medical entities (drugs, diseases, conditions)
    # Scispacy model recognizes DISEASE and CHEMICAL entities
    medical_entities = set()

    # Add medical entities from scispacy NER
    for ent in doc.ents:
        # DISEASE: medical conditions, diseases
        # CHEMICAL: drugs, chemicals
        if ent.label_ in ['DISEASE', 'CHEMICAL']:
            for token in ent:
                medical_entities.add(token.i)
        # Also include standard entities that might capture medical terms
        elif ent.label_ in ['PRODUCT', 'ORG']:
            for token in ent:
                medical_entities.add(token.i)

    # Medical suffixes that indicate conditions/diseases (fallback for terms not caught by NER)
    medical_suffixes = [
        'itis', 'osis', 'emia', 'pathy', 'algia', 'oma', 'iasis',
        'trophy', 'penia', 'plasia', 'dynia', 'plegia', 'rrhea'
    ]

    # Check each token for additional medical patterns not caught by NER
    for token in doc:
        # Skip if already identified
        if token.i in medical_entities:
            continue

        # Skip very short words, punctuation, and stop words
        if len(token.text) <= 3 or not token.is_alpha or token.is_stop:
            continue

        token_lower = token.text.lower()

        # 1. Capitalized medical terms (likely drug names)
        if token.text[0].isupper() and len(token.text) > 3 and not token.is_sent_start:
            medical_entities.add(token.i)

        # 2. Words with medical suffixes (diseases/conditions)
        elif any(token_lower.endswith(suffix) for suffix in medical_suffixes):
            medical_entities.add(token.i)

        # 3. Medical prefixes (for nouns)
        elif token.pos_ == 'NOUN' and any([
            token_lower.startswith('hyper'),
            token_lower.startswith('hypo'),
            token_lower.startswith('dys'),
            token_lower.startswith('poly'),
        ]):
            medical_entities.add(token.i)

    # If no medical entities found, return original text
    if not medical_entities:
        return text

    # Build result with typos applied to medical entities
    result = []
    for token in doc:
        if token.i in medical_entities:
            # Apply typo with specified probability
            if random.random() < typo_probability:
                result.append(apply_typo(token.text) + token.whitespace_)
            else:
                result.append(token.text_with_ws)
        else:
            result.append(token.text_with_ws)

    return ''.join(result)
    
    
def find_best_matching_sentence(text_sentences, target_sentence):
    """Find the sentence in text_sentences that best matches target_sentence using Levenshtein distance."""
    best_match = None
    best_distance = float('inf')
    best_index = -1

    target_lower = target_sentence.lower().strip()

    for idx, sent in enumerate(text_sentences):
        sent_lower = sent.text.lower().strip()
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(sent_lower, target_lower)

        if distance < best_distance:
            best_distance = distance
            best_match = sent
            best_index = idx

    return best_match, best_distance, best_index


def find_all_matching_sentences(text_sentences, target_sentence, threshold):
    """Find all sentences that match target_sentence within threshold, sorted by distance."""
    target_lower = target_sentence.lower().strip()
    matches = []

    for idx, sent in enumerate(text_sentences):
        sent_lower = sent.text.lower().strip()
        distance = Levenshtein.distance(sent_lower, target_lower)

        if distance < threshold:
            matches.append({
                'sentence': sent,
                'distance': distance,
                'index': idx
            })

    # Sort by distance (best matches first)
    matches.sort(key=lambda x: x['distance'])
    return matches


def remove_sentences_by_percentage(text, percentage=0.3):
    """Remove a percentage of sentences from the text.

    Args:
        text: The text to modify
        percentage: Percentage of sentences to remove (0.0-1.0), default 0.3 (30%)

    Returns:
        Modified text with sentences removed, or original text if too few sentences
    """
    # Process text into sentences
    doc = nlp(text)
    text_sentences = list(doc.sents)

    total_sentences = len(text_sentences)

    # Need at least 2 sentences to remove anything
    if total_sentences < 2:
        return text

    # Calculate number of sentences to remove
    num_to_remove = max(1, int(total_sentences * percentage))

    # Don't remove all sentences - keep at least 1
    num_to_remove = min(num_to_remove, total_sentences - 1)

    # Randomly select indices to remove
    import random
    indices_to_remove = random.sample(range(total_sentences), num_to_remove)

    # Sort in descending order to maintain correct indices when removing
    indices_to_remove.sort(reverse=True)

    # Remove the selected sentences
    for idx in indices_to_remove:
        text_sentences.pop(idx)

    # Reconstruct the text
    modified_text = ' '.join([sent.text.strip() for sent in text_sentences])
    return modified_text 