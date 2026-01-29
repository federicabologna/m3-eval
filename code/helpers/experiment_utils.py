"""
Shared utilities for M3-Eval experiments.
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple


def setup_paths(output_dir=None):
    """Setup project paths."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'cqa_eval')

    coarse_data_path = os.path.join(project_root, 'data', 'coarse_5pt_expert+llm_consolidated.jsonl')
    fine_data_path = os.path.join(project_root, 'data', 'fine_5pt_expert+llm_consolidated.jsonl')

    os.makedirs(output_dir, exist_ok=True)

    return {
        'project_root': project_root,
        'output_dir': output_dir,
        'coarse_data_path': coarse_data_path,
        'fine_data_path': fine_data_path,
        'prompts_dir': os.path.join(script_dir, 'prompts')
    }


def load_qa_data(data_path: str) -> List[Dict]:
    """Load QA pairs from JSONL file."""
    qa_pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line))
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


def apply_perturbation(
    perturbation_name: str,
    original_answer: str,
    qa_pair: Dict,
    typo_prob: float = 0.5,
    remove_pct: float = 0.3
) -> Tuple[str, Dict]:
    """
    Apply perturbation to an answer.

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

    if perturbation_name == 'add_confusion':
        perturbed_answer = add_confusion(original_answer)

    elif perturbation_name == 'add_typos':
        perturbed_answer = add_typos(original_answer, typo_probability=typo_prob)
        metadata['typo_probability'] = typo_prob

    elif perturbation_name == 'change_dosage':
        perturbed_answer, change_counts = change_dosage(original_answer)
        metadata['change_counts'] = change_counts

    elif perturbation_name == 'remove_sentences':
        perturbed_answer = remove_sentences_by_percentage(original_answer, percentage=remove_pct)
        metadata['removal_percentage'] = remove_pct

    if perturbed_answer is None:
        perturbed_answer = original_answer

    return perturbed_answer, metadata


def save_result(output_path: str, result: Dict):
    """Save a single result to JSONL file."""
    try:
        with open(output_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')
    except IOError as e:
        print(f"ERROR: Failed to save result: {e}")
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
    num_runs: int = 5
) -> Dict[str, Dict]:
    """
    Get original ratings - load from file if complete, otherwise generate missing ones.

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
        'remove_must_have': {
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
