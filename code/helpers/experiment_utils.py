"""
Shared utilities for M3-Eval experiments.
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple


def setup_paths(output_dir=None, dataset='cqa_eval'):
    """Setup project paths."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)

    # Set default output directory based on dataset
    if output_dir is None:
        if dataset == 'medinfo':
            output_dir = os.path.join(project_root, 'output', 'medinfo')
        elif dataset == 'woundcare':
            output_dir = os.path.join(project_root, 'output', 'woundcare')
        elif dataset == 'dermavqa':
            output_dir = os.path.join(project_root, 'output', 'dermavqa')
        else:
            output_dir = os.path.join(project_root, 'output', 'cqa_eval')

    # Select dataset paths
    if dataset == 'medinfo':
        # MedInfo2019 dataset (using 300-sample subsets)
        coarse_data_path = os.path.join(project_root, 'data', 'medinfo_coarse_subset300.jsonl')
        fine_data_path = os.path.join(project_root, 'data', 'medinfo_fine_subset300.jsonl')
    elif dataset == 'woundcare':
        # WoundCare dataset (coarse-level only, train set)
        coarse_data_path = os.path.join(project_root, 'data', 'woundcare_coarse.jsonl')
        fine_data_path = None  # Not using fine-level for WoundCare
    elif dataset == 'dermavqa':
        # DermaVQA dataset (300-sample subset, coarse-level only)
        coarse_data_path = os.path.join(project_root, 'data', 'dermavqa_coarse_subset300.jsonl')
        fine_data_path = None  # Not using fine-level for DermaVQA
    else:  # cqa_eval (default)
        coarse_data_path = os.path.join(project_root, 'data', 'coarse_5pt_expert+llm_consolidated.jsonl')
        fine_data_path = os.path.join(project_root, 'data', 'fine_5pt_expert+llm_consolidated.jsonl')

    fine_subset_path = os.path.join(project_root, 'data', 'fine_sentence_ids_subset.json')
    fine_balanced_subset_path = os.path.join(project_root, 'data', 'fine_sentence_ids_balanced_subset.json')

    os.makedirs(output_dir, exist_ok=True)

    return {
        'project_root': project_root,
        'output_dir': output_dir,
        'coarse_data_path': coarse_data_path,
        'fine_data_path': fine_data_path,
        'fine_subset_path': fine_subset_path,
        'fine_balanced_subset_path': fine_balanced_subset_path,
        'prompts_dir': os.path.join(script_dir, 'prompts'),
        'dataset': dataset
    }


def load_qa_data(data_path: str, sentence_ids_subset_file: str = None) -> List[Dict]:
    """
    Load QA pairs from JSONL file.

    Args:
        data_path: Path to the JSONL data file
        sentence_ids_subset_file: Optional path to JSON file containing list of sentence_ids to filter by

    Returns:
        List of QA pair dictionaries
    """
    qa_pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line))

    # Filter by sentence_ids if subset file is provided
    if sentence_ids_subset_file and os.path.exists(sentence_ids_subset_file):
        with open(sentence_ids_subset_file, 'r') as f:
            allowed_ids = set(json.load(f))

        # Determine ID key
        id_key = 'sentence_id' if 'sentence_id' in qa_pairs[0] else 'answer_id'

        original_count = len(qa_pairs)
        qa_pairs = [qa for qa in qa_pairs if qa[id_key] in allowed_ids]
        filtered_count = len(qa_pairs)

        print(f"Filtered data using {os.path.basename(sentence_ids_subset_file)}: {original_count} -> {filtered_count} examples")

    return qa_pairs


def get_processed_ids(output_path: str) -> Set[str]:
    """Get set of already processed answer IDs from output file."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Use answer_id for coarse/fine, or sentence_id if present
                    id_key = 'sentence_id' if 'sentence_id' in entry else 'answer_id'
                    processed_ids.add(entry[id_key])
            print(f"Found {len(processed_ids)} already processed entries in output file")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    return processed_ids


def clean_model_name(model: str) -> str:
    """Clean model name for use in filenames."""
    return model.replace('/', '-').replace('.', '_')


def get_id_key(qa_pairs: List[Dict]) -> str:
    """Determine ID key (sentence_id or answer_id) from QA pairs."""
    return 'sentence_id' if 'sentence_id' in qa_pairs[0] else 'answer_id'


def get_successful_perturbation_ids(
    perturbation_names: List[str],
    level: str,
    output_dir: str,
    typo_probs: List[float] = None,
    remove_pcts: List[float] = None
) -> Set[str]:
    """
    Load all perturbation files and collect IDs that have successful perturbations.

    Args:
        perturbation_names: List of perturbation types
        level: 'coarse' or 'fine'
        output_dir: Output directory path
        typo_probs: List of typo probabilities to check (for add_typos)
        remove_pcts: List of removal percentages to check (for remove_sentences)

    Returns:
        Set of IDs that have successful perturbations
    """
    perturbations_dir = os.path.join(output_dir, 'perturbations')
    successful_ids = set()

    # Default parameters
    if typo_probs is None:
        typo_probs = [0.3, 0.5, 0.7]
    if remove_pcts is None:
        remove_pcts = [0.3, 0.5, 0.7]

    for perturbation_name in perturbation_names:
        # Determine parameter values for this perturbation
        if perturbation_name == 'remove_sentences':
            param_values = [(pct, 0.5) for pct in remove_pcts]
        elif perturbation_name == 'add_typos':
            param_values = [(0.3, prob) for prob in typo_probs]
        else:
            param_values = [(0.3, 0.5)]

        for remove_pct, typo_prob in param_values:
            # Build filename
            if perturbation_name == 'remove_sentences':
                pct_str = str(int(remove_pct * 100))
                filename = f"{perturbation_name}_{pct_str}pct_{level}.jsonl"
            elif perturbation_name == 'add_typos':
                prob_str = str(typo_prob).replace('.', '')
                filename = f"{perturbation_name}_{prob_str}prob_{level}.jsonl"
            else:
                filename = f"{perturbation_name}_{level}.jsonl"

            filepath = os.path.join(perturbations_dir, filename)

            # Load perturbations if file exists
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        # Skip entries without successful perturbations
                        if 'skip_reason' not in entry:
                            id_key = 'sentence_id' if 'sentence_id' in entry else 'answer_id'
                            successful_ids.add(entry[id_key])

    return successful_ids


def extract_marked_text(text: str) -> Tuple[str, str, str]:
    """
    Extract marked text from answer with <mark> tags.

    Returns:
        (before_mark, marked_text, after_mark) where:
        - before_mark: text before <mark> tag
        - marked_text: text inside <mark> tags (without the tags)
        - after_mark: text after </mark> tag

    If no <mark> tags found, returns ('', full_text, '')
    """
    import re

    match = re.search(r'(.*?)<mark>(.*?)</mark>(.*)', text, re.DOTALL)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        # No mark tags found, return entire text as marked portion
        return '', text, ''


def apply_perturbation(
    perturbation_name: str,
    original_answer: str,
    qa_pair: Dict,
    typo_prob: float = 0.5,
    remove_pct: float = 0.3
) -> Tuple[str, Dict]:
    """
    Apply perturbation to an answer.

    For fine-level data (annotation_type='fine'), perturbations are applied
    only to the marked sentence within <mark> tags.
    For coarse-level data, perturbations are applied to the entire answer.

    Returns:
        (perturbed_answer, metadata) where metadata contains perturbation-specific info
    """
    from helpers.perturbation_functions import (
        add_confusion,
        add_typos,
        change_dosage,
        remove_sentences_by_percentage
    )

    perturbed_answer = None
    metadata = {}

    # Check if this is fine-level data (has marked text)
    is_fine_level = qa_pair.get('annotation_type') == 'fine' and '<mark>' in original_answer

    if is_fine_level:
        # Extract the marked sentence
        before_mark, marked_text, after_mark = extract_marked_text(original_answer)
        text_to_perturb = marked_text
    else:
        # Use entire answer for coarse-level
        text_to_perturb = original_answer

    # Apply perturbation to the text
    if perturbation_name == 'add_confusion':
        perturbed_text = add_confusion(text_to_perturb)

    elif perturbation_name == 'add_typos':
        perturbed_text = add_typos(text_to_perturb, typo_probability=typo_prob)
        metadata['typo_probability'] = typo_prob

    elif perturbation_name == 'change_dosage':
        perturbed_text, change_counts = change_dosage(text_to_perturb)
        metadata['change_counts'] = change_counts

    elif perturbation_name == 'remove_sentences':
        perturbed_text = remove_sentences_by_percentage(text_to_perturb, percentage=remove_pct)
        metadata['removal_percentage'] = remove_pct
    else:
        perturbed_text = text_to_perturb

    # Reconstruct the full answer
    if is_fine_level:
        # Put the perturbed text back inside <mark> tags
        perturbed_answer = f"{before_mark}<mark>{perturbed_text}</mark>{after_mark}"
    else:
        perturbed_answer = perturbed_text

    if perturbed_answer is None:
        perturbed_answer = original_answer

    return perturbed_answer, metadata


def save_result(output_path: str, result: Dict):
    """Save a single result to JSONL file."""
    try:
        with open(output_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')
            f.flush()  # Ensure data is written to disk immediately
    except IOError as e:
        print(f"ERROR: Failed to save result to {output_path}: {e}")
        raise


def get_or_create_perturbations(
    perturbation_name: str,
    level: str,
    qa_pairs: List[Dict],
    typo_prob: float = 0.5,
    remove_pct: float = 0.3,
    seed: int = 42,
    output_dir: str = None
) -> Dict[str, Dict]:
    """
    Get perturbations - load from file if exists, generate missing ones if partial.

    Returns:
        Dictionary mapping answer_id/sentence_id to perturbation data
    """
    if output_dir is None:
        paths = setup_paths()
        output_dir = paths['output_dir']

    perturbations_dir = os.path.join(output_dir, 'perturbations')
    os.makedirs(perturbations_dir, exist_ok=True)

    # Determine filename based on perturbation type (no seed in name)
    if perturbation_name == 'remove_sentences':
        pct_str = str(int(remove_pct * 100))
        filename = f"{perturbation_name}_{pct_str}pct_{level}.jsonl"
    elif perturbation_name == 'add_typos':
        prob_str = str(typo_prob).replace('.', '')
        filename = f"{perturbation_name}_{prob_str}prob_{level}.jsonl"
    else:
        filename = f"{perturbation_name}_{level}.jsonl"

    filepath = os.path.join(perturbations_dir, filename)
    id_key = get_id_key(qa_pairs)

    # Load existing perturbations if file exists
    perturbations = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                entry = json.loads(line)
                perturbations[entry[id_key]] = entry
        print(f"✓ Loaded {len(perturbations)} existing perturbations from {filename}")

    # Check which qa_pairs are missing
    missing_qa_pairs = [qa for qa in qa_pairs if qa[id_key] not in perturbations]

    if len(missing_qa_pairs) == 0:
        print(f"✓ All {len(qa_pairs)} perturbations complete!")
        return perturbations

    # Generate missing perturbations
    print(f"⚠ {len(missing_qa_pairs)} perturbations missing (out of {len(qa_pairs)} total)")
    print(f"  Generating missing perturbations...")

    # Set random seed for reproducibility
    random.seed(seed)

    with open(filepath, 'a') as f:  # Append mode
        for qa_pair in missing_qa_pairs:
            original_answer = qa_pair['answer']

            # Apply perturbation
            perturbed_answer, metadata = apply_perturbation(
                perturbation_name,
                original_answer,
                qa_pair,
                typo_prob=typo_prob,
                remove_pct=remove_pct
            )

            # Skip if no perturbation applied
            if perturbed_answer == original_answer:
                continue

            # Build entry
            entry = qa_pair.copy()
            entry['perturbation'] = perturbation_name
            entry['perturbed_answer'] = perturbed_answer
            entry['random_seed'] = seed
            entry.update(metadata)

            # Save to file
            json.dump(entry, f)
            f.write('\n')
            f.flush()  # Ensure data is written to disk immediately

            # Add to dictionary
            perturbations[qa_pair[id_key]] = entry

    print(f"✓ Generated {len(missing_qa_pairs)} missing perturbations. Total: {len(perturbations)}")
    return perturbations


def get_or_create_original_ratings(
    qa_pairs: List[Dict],
    level: str,
    prompt_path: str,
    model: str,
    output_dir: str,
    model_name_clean: str,
    num_runs: int = 5,
    skip_missing: bool = False
) -> Dict[str, Dict]:
    """
    Get original ratings - load from file if complete, otherwise generate missing ones.

    Args:
        skip_missing: If True, skip computing missing ratings and only return existing ones

    Returns:
        Dictionary mapping answer_id/sentence_id to original rating
    """
    from perturbation_pipeline import load_prompt, get_rating_with_averaging
    import time

    # Setup paths
    original_ratings_dir = os.path.join(output_dir, 'original_ratings')
    os.makedirs(original_ratings_dir, exist_ok=True)

    original_ratings_filename = f"original_{level}_{model_name_clean}_rating.jsonl"
    original_ratings_path = os.path.join(original_ratings_dir, original_ratings_filename)

    id_key = get_id_key(qa_pairs)

    # Load existing ratings if file exists
    ratings_dict = {}
    if os.path.exists(original_ratings_path):
        with open(original_ratings_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                ratings_dict[entry[id_key]] = entry['original_rating']
        print(f"✓ Loaded {len(ratings_dict)} existing original ratings from {original_ratings_filename}")

    # Check which qa_pairs are missing
    missing_qa_pairs = [qa for qa in qa_pairs if qa[id_key] not in ratings_dict]

    if len(missing_qa_pairs) == 0:
        print(f"✓ All {len(qa_pairs)} original ratings complete!")
        return ratings_dict

    # If skip_missing is True, return only existing ratings without computing missing ones
    if skip_missing:
        print(f"⚠ {len(missing_qa_pairs)} original ratings missing (out of {len(qa_pairs)} total)")
        print(f"✓ Skipping missing ratings (using only {len(ratings_dict)} existing ratings)")
        return ratings_dict

    # Generate missing ratings
    print(f"⚠ {len(missing_qa_pairs)} original ratings missing (out of {len(qa_pairs)} total)")
    print(f"  Computing missing ratings...")

    # Load prompt
    system_prompt, user_template = load_prompt(prompt_path)

    with open(original_ratings_path, 'a') as f:  # Append mode
        for qa_pair in missing_qa_pairs:
            question = qa_pair['question']
            original_answer = qa_pair['answer']

            print(f"\nGetting rating for {qa_pair[id_key]}...")
            start_time = time.time()
            original_rating = get_rating_with_averaging(
                question, original_answer, system_prompt, user_template, model, num_runs=num_runs, flush_output=True
            )
            elapsed_time = time.time() - start_time
            print(f'Time taken: {elapsed_time:.2f} seconds')

            # Build entry
            result = qa_pair.copy()
            result['original_rating'] = original_rating

            # Save to file
            json.dump(result, f)
            f.write('\n')
            f.flush()  # Ensure data is written to disk immediately

            # Add to dictionary
            ratings_dict[qa_pair[id_key]] = original_rating

    print(f"✓ Generated {len(missing_qa_pairs)} missing ratings. Total: {len(ratings_dict)}")
    return ratings_dict


def get_perturbation_prompt_mapping(level='coarse'):
    """
    Get mapping of perturbation names to detection prompts.

    Args:
        level: 'coarse' (answer-level) or 'fine' (sentence-level)
    """
    # Determine text based on level
    if level == 'fine':
        text_type = 'sentence'
    else:
        text_type = 'answer'

    return {
        'change_dosage': {
            'detection_type': 'error',
            'question': f'Is there an error in this {text_type}?',
            'priming_text': f'Note: This {text_type} contains an error.'
        },
        'add_typos': {
            'detection_type': 'error',
            'question': f'Is there an error in this {text_type}?',
            'priming_text': f'Note: This {text_type} contains an error.'
        },
        'remove_sentences': {
            'detection_type': 'missing',
            'question': f'Is important information missing from this {text_type}?',
            'priming_text': f'Note: This {text_type} is missing important information.'
        },
        'add_confusion': {
            'detection_type': 'error',
            'question': f'Is there an error in this {text_type}?',
            'priming_text': f'Note: This {text_type} contains an error.'
        }
    }
