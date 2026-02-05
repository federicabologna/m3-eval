#!/usr/bin/env python3
"""
Create fine-level version of MedInfo2019 dataset.

Takes the coarse-level MedInfo2019 dataset and creates sentence-level entries
by marking each sentence individually.
"""

import json
import os
import re
import spacy
from typing import List, Tuple


def merge_short_sentences(sentences, min_length=30):
    """Merge sentences shorter than min_length with the next sentence."""
    merged = []
    i = 0

    while i < len(sentences):
        current = sentences[i]
        if len(current) < min_length and '\n\n' not in current and i + 1 < len(sentences):
            # Merge with the next sentence
            merged_sentence = current.strip() + " " + sentences[i + 1].strip()
            merged.append(merged_sentence)
            i += 2  # Skip the next one since it's been merged
        else:
            merged.append(current)
            i += 1

    return merged


def clean_fine_sentence(text):
    """Clean formatting issues in marked sentences."""
    # Replace "\n\n]" with "]\n\n"
    text = text.replace("\n\n]", "]\n\n")
    # Replace "\n]" with "]\n"
    text = text.replace("\n]", "]\n")
    # Replace "\n\nX.]" with "]\n\nX." where X is 1–7
    text = re.sub(r'\n\n([1-7])\.\]', r']\n\n\1.', text)
    # Replace "<mark>[X." with "X. <mark>[" where X is a number between 1 and 7
    text = re.sub(r'<mark>\[(?P<num>[1-7])\. (.*?)\]', r'\g<num>. <mark>[\2]', text)

    return text


def bold_sentences(text, nlp):
    """
    Create multiple versions of text, each with one sentence marked.

    Returns:
        List of tuples: (sentence_index, marked_text)
    """
    # Process the text with spaCy to segment into sentences
    doc = nlp(text)
    split_sentences = [sentence.text for sentence in doc.sents if len(sentence.text) > 5]

    # Merge short sentences
    sentences = merge_short_sentences(split_sentences)

    fine_sentences = []
    for bold_index in range(len(sentences)):
        bold_sentence = f'<mark>[{sentences[bold_index]}]</mark>'
        new_sentences = sentences[:bold_index] + [bold_sentence] + sentences[bold_index + 1:]
        fine_sentence = ' '.join(new_sentences)
        clean_sentence = clean_fine_sentence(fine_sentence)
        fine_sentences.append((bold_index, clean_sentence))

    return fine_sentences


def create_fine_dataset(input_path: str, output_path: str):
    """
    Create fine-level dataset from coarse-level MedInfo2019 dataset.

    Args:
        input_path: Path to coarse-level JSONL file
        output_path: Path to output fine-level JSONL file
    """
    print(f"Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Loading dataset from: {input_path}")

    # Load coarse dataset
    coarse_data = []
    with open(input_path, 'r') as f:
        for line in f:
            coarse_data.append(json.loads(line))

    print(f"Loaded {len(coarse_data)} coarse-level entries")

    # Create fine-level entries
    fine_data = []
    total_sentences = 0

    for entry in coarse_data:
        answer_id = entry['answer_id']
        answer = entry['answer']

        # Skip very short answers
        if len(answer.strip()) < 10:
            print(f"  Skipping {answer_id}: Answer too short")
            continue

        # Create marked versions for each sentence
        try:
            marked_sentences = bold_sentences(answer, nlp)
        except Exception as e:
            print(f"  Error processing {answer_id}: {e}")
            continue

        # Create a fine entry for each sentence
        for sentence_idx, marked_answer in marked_sentences:
            fine_entry = entry.copy()

            # Update fields for fine-level
            fine_entry['sentence_id'] = f"{answer_id}_{sentence_idx}"
            fine_entry['answer'] = marked_answer
            fine_entry['sentence_index'] = sentence_idx
            fine_entry['annotation_type'] = 'fine'

            fine_data.append(fine_entry)

        total_sentences += len(marked_sentences)

        if len(fine_data) % 100 == 0:
            print(f"  Processed {len(coarse_data)} answers -> {len(fine_data)} sentences")

    # Save fine-level dataset
    print(f"\nSaving fine-level dataset to: {output_path}")
    with open(output_path, 'w') as f:
        for entry in fine_data:
            json.dump(entry, f)
            f.write('\n')

    print(f"\n{'='*80}")
    print(f"Fine-level dataset created!")
    print(f"{'='*80}")
    print(f"Input:  {len(coarse_data)} coarse-level answers")
    print(f"Output: {len(fine_data)} fine-level sentences")
    print(f"Average sentences per answer: {len(fine_data) / len(coarse_data):.1f}")
    print(f"\nSaved to: {output_path}")


def main():
    print("="*80)
    print("MedInfo2019 Fine-Level Dataset Creator")
    print("="*80)

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'medinfo2019_medications_qa.jsonl')
    output_path = os.path.join(script_dir, 'medinfo2019_medications_qa_fine.jsonl')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    create_fine_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
