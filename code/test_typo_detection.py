import sys
import json
sys.path.insert(0, '/Users/Federica_1/Documents/GitHub/m3-eval/code')

import spacy

nlp = spacy.load('en_core_web_sm')

# Load some sample answers
with open('/Users/Federica_1/Documents/GitHub/m3-eval/data/coarse_5pt_expert+llm_consolidated.jsonl', 'r') as f:
    samples = []
    for i, line in enumerate(f):
        if i >= 5:  # Get first 5 answers
            break
        data = json.loads(line)
        samples.append(data['answer'])

# Medical suffixes
medical_suffixes = [
    'itis', 'osis', 'emia', 'pathy', 'algia', 'oma', 'iasis',
    'trophy', 'penia', 'plasia', 'dynia', 'plegia', 'rrhea'
]

for idx, text in enumerate(samples[:3], 1):
    print("="*80)
    print(f"SAMPLE {idx}:")
    print("="*80)
    print(f"Text: {text[:200]}...\n")

    doc = nlp(text)

    # Identify medical entities using the same logic
    medical_entities = {}

    # Named entities
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG', 'PERSON']:
            for token in ent:
                medical_entities[token.i] = f"{token.text} (NER: {ent.label_})"

    # Check each token
    for token in doc:
        if token.i in medical_entities:
            continue

        if len(token.text) <= 3 or not token.is_alpha or token.is_stop:
            continue

        token_lower = token.text.lower()

        # 1. Capitalized medical terms
        if token.text[0].isupper() and len(token.text) > 3 and not token.is_sent_start:
            medical_entities[token.i] = f"{token.text} (Capitalized)"

        # 2. Medical suffixes
        elif any(token_lower.endswith(suffix) for suffix in medical_suffixes):
            medical_entities[token.i] = f"{token.text} (Suffix: {[s for s in medical_suffixes if token_lower.endswith(s)][0]})"

        # 3. Medical prefixes
        elif token.pos_ == 'NOUN' and any([
            token_lower.startswith('hyper'),
            token_lower.startswith('hypo'),
            token_lower.startswith('dys'),
            token_lower.startswith('poly'),
            token_lower.startswith('neo'),
            token_lower.startswith('pseudo'),
        ]):
            prefix = [p for p in ['hyper', 'hypo', 'dys', 'poly', 'neo', 'pseudo'] if token_lower.startswith(p)][0]
            medical_entities[token.i] = f"{token.text} (Prefix: {prefix})"

        # 4. Medical condition markers
        elif token_lower in ['syndrome', 'disease', 'disorder', 'infection', 'inflammation', 'condition']:
            medical_entities[token.i] = f"{token.text} (Marker word)"
            if token.i > 0 and doc[token.i - 1].pos_ == 'NOUN':
                medical_entities[token.i - 1] = f"{doc[token.i - 1].text} (Before marker)"

    if medical_entities:
        print(f"Identified {len(medical_entities)} medical entities:")
        for token_idx in sorted(medical_entities.keys()):
            print(f"  - {medical_entities[token_idx]}")
    else:
        print("No medical entities identified")

    print()
