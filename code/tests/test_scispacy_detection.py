import spacy

# Test with both models
print("="*80)
print("TESTING MEDICAL ENTITY DETECTION")
print("="*80)

# Load both models
try:
    nlp_sci = spacy.load("en_ner_bc5cdr_md")
    has_scispacy = True
    print("\n✓ Scispacy model loaded: en_ner_bc5cdr_md")
except:
    has_scispacy = False
    print("\n✗ Scispacy model NOT found")
    print("Install with:")
    print("  pip install scispacy")
    print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")

nlp_standard = spacy.load("en_core_web_sm")
print("✓ Standard spacy model loaded: en_core_web_sm")

# Test texts with medical terms
test_texts = [
    "Saxenda (liraglutide) is a GLP-1 receptor agonist used for diabetes treatment.",
    "Patients with hypertension, diabetes, or arthritis should consult their doctor.",
    "Common side effects include nausea, diarrhea, and hypoglycemia.",
    "Metformin is used to treat type 2 diabetes and helps control blood glucose levels.",
    "The patient has chronic obstructive pulmonary disease (COPD) and asthma."
]

for idx, text in enumerate(test_texts, 1):
    print(f"\n{'='*80}")
    print(f"TEST {idx}: {text}")
    print("="*80)

    # Standard spacy
    doc_standard = nlp_standard(text)
    standard_entities = [(ent.text, ent.label_) for ent in doc_standard.ents]
    print(f"\nStandard Spacy entities: {len(standard_entities)}")
    for ent_text, label in standard_entities:
        print(f"  - {ent_text} ({label})")

    if has_scispacy:
        # Scispacy
        doc_sci = nlp_sci(text)
        sci_entities = [(ent.text, ent.label_) for ent in doc_sci.ents]
        print(f"\nScispacy entities: {len(sci_entities)}")
        for ent_text, label in sci_entities:
            print(f"  - {ent_text} ({label})")

        # Show what scispacy caught that standard didn't
        standard_texts = set([e[0].lower() for e in standard_entities])
        sci_texts = set([e[0].lower() for e in sci_entities])
        new_catches = sci_texts - standard_texts

        if new_catches:
            print(f"\n✓ Scispacy caught {len(new_catches)} additional medical terms:")
            for term in new_catches:
                print(f"  + {term}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
