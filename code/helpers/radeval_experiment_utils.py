"""
Experiment utilities for RadEval dataset.
Similar structure to experiment_utils.py for CQA eval.
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple


def setup_radeval_paths(output_dir=None, data_path=None):
    """Setup RadEval project paths."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'radeval')

    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'radeval_expert_dataset.jsonl')

    os.makedirs(output_dir, exist_ok=True)

    return {
        'project_root': project_root,
        'output_dir': output_dir,
        'data_path': data_path,
    }


def load_radeval_data(data_path: str) -> List[Dict]:
    """Load RadEval data from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_processed_ids(output_path: str) -> Set[str]:
    """Get set of already processed IDs from output file."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    processed_ids.add(entry['id'])
            print(f"Found {len(processed_ids)} already processed entries in output file")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    return processed_ids


def clean_model_name(model: str) -> str:
    """Clean model name for use in filenames."""
    return model.replace('/', '-').replace('.', '_')


def apply_radeval_perturbation(
    perturbation_name: str,
    original_text: str,
    typo_prob: float = 0.5,
    remove_pct: float = 0.3,
    llm_model: str = 'gpt-4.1'
) -> Tuple[str, Dict]:
    """
    Apply perturbation to a radiology report.

    Args:
        perturbation_name: Name of perturbation to apply
        original_text: Original report text
        typo_prob: Probability for typo perturbation
        remove_pct: Percentage for sentence removal
        llm_model: Model to use for LLM-based perturbations (default: gpt-4.1)

    Returns:
        (perturbed_text, metadata) where metadata contains perturbation-specific info
    """
    from helpers.radeval_perturbations import (
        add_typos,
        remove_sentences_by_percentage,
        swap_qualifiers,
        swap_organs,
        inject_false_prediction,
        inject_contradiction,
        inject_false_negation
    )

    perturbed_text = None
    metadata = {}

    if perturbation_name == 'add_typos':
        perturbed_text = add_typos(original_text, typo_probability=typo_prob)
        metadata['typo_probability'] = typo_prob

    elif perturbation_name == 'remove_sentences':
        perturbed_text = remove_sentences_by_percentage(original_text, percentage=remove_pct)
        metadata['removal_percentage'] = remove_pct

    elif perturbation_name == 'swap_qualifiers':
        perturbed_text, change_info = swap_qualifiers(original_text)
        metadata['qualifier_changes'] = change_info

    elif perturbation_name == 'swap_organs':
        perturbed_text, change_info = swap_organs(original_text)
        metadata['organ_changes'] = change_info

    # LLM-based perturbations using rexerr prompts
    elif perturbation_name == 'inject_false_prediction':
        perturbed_text, metadata = inject_false_prediction(original_text, model=llm_model)

    elif perturbation_name == 'inject_contradiction':
        perturbed_text, metadata = inject_contradiction(original_text, model=llm_model)

    elif perturbation_name == 'inject_false_negation':
        perturbed_text, metadata = inject_false_negation(original_text, model=llm_model)

    if perturbed_text is None:
        perturbed_text = original_text

    return perturbed_text, metadata


def save_result(output_path: str, result: Dict):
    """Save a single result to JSONL file."""
    try:
        with open(output_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')
    except IOError as e:
        print(f"ERROR: Failed to save result: {e}")
        raise


def get_or_create_radeval_perturbations(
    perturbation_name: str,
    data: List[Dict],
    text_field: str,
    typo_prob: float = 0.5,
    remove_pct: float = 0.3,
    seed: int = 42,
    output_dir: str = None,
    llm_model: str = 'gpt-4.1'
) -> Dict[str, Dict]:
    """
    Get perturbations - load from file if exists, generate missing ones if partial.

    Args:
        perturbation_name: Name of perturbation
        data: List of data entries
        text_field: Field name containing text to perturb (e.g., 'prediction', 'report')
        typo_prob: Typo probability
        remove_pct: Removal percentage
        seed: Random seed
        output_dir: Output directory
        llm_model: Model for LLM-based perturbations (default: gpt-4.1)

    Returns:
        Dictionary mapping ID to perturbation data
    """
    if output_dir is None:
        paths = setup_radeval_paths()
        output_dir = paths['output_dir']

    perturbations_dir = os.path.join(output_dir, 'perturbations')
    os.makedirs(perturbations_dir, exist_ok=True)

    # Determine filename based on perturbation type
    if perturbation_name == 'remove_sentences':
        pct_str = str(int(remove_pct * 100))
        filename = f"{perturbation_name}_{pct_str}pct.jsonl"
    elif perturbation_name == 'add_typos':
        prob_str = str(typo_prob).replace('.', '')
        filename = f"{perturbation_name}_{prob_str}prob.jsonl"
    else:
        filename = f"{perturbation_name}.jsonl"

    filepath = os.path.join(perturbations_dir, filename)

    # Load existing perturbations if file exists
    perturbations = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                entry = json.loads(line)
                perturbations[entry['id']] = entry
        print(f"✓ Loaded {len(perturbations)} existing perturbations from {filename}")

    # Check which entries are missing
    missing_data = [item for item in data if item['id'] not in perturbations]

    if len(missing_data) == 0:
        print(f"✓ All {len(data)} perturbations complete!")
        return perturbations

    # Generate missing perturbations
    print(f"⚠ {len(missing_data)} perturbations missing (out of {len(data)} total)")
    print(f"  Generating missing perturbations...")

    # Set random seed for reproducibility
    random.seed(seed)

    with open(filepath, 'a') as f:
        for item in missing_data:
            original_text = item[text_field]

            # Apply perturbation
            perturbed_text, metadata = apply_radeval_perturbation(
                perturbation_name,
                original_text,
                typo_prob=typo_prob,
                remove_pct=remove_pct,
                llm_model=llm_model
            )

            # Skip if no perturbation applied or if skip_reason is present
            if 'skip_reason' in metadata:
                print(f"  Skipping {item['id']}: {metadata['skip_reason']}")
                continue

            # Skip if no actual changes were made
            num_changes = metadata.get('num_changes', 0)
            if num_changes == 0 or perturbed_text == original_text:
                print(f"  Skipping {item['id']}: No changes made (num_changes={num_changes})")
                continue

            # Build entry
            entry = item.copy()
            entry['perturbation'] = perturbation_name
            entry[f'perturbed_{text_field}'] = perturbed_text
            entry['random_seed'] = seed
            entry.update(metadata)

            # Save to file
            json.dump(entry, f)
            f.write('\n')

            # Add to dictionary
            perturbations[item['id']] = entry

    print(f"✓ Generated {len(missing_data)} missing perturbations. Total: {len(perturbations)}")
    return perturbations


def get_or_create_radeval_original_ratings(
    data: List[Dict],
    text_field: str,
    reference_field: str,
    output_dir: str,
    model: str = None,
    cpu: bool = False,
    num_runs: int = 1
) -> Dict[str, Dict]:
    """
    Get original GREEN ratings - load from file if complete, otherwise generate missing ones.

    Args:
        data: List of data entries
        text_field: Field containing text to evaluate (e.g., 'prediction')
        reference_field: Field containing reference text
        output_dir: Output directory
        model: Model to use for GREEN evaluation
        cpu: If True, run on CPU (for GREEN model only)
        num_runs: Number of evaluation runs (GREEN is deterministic)

    Returns:
        Dictionary mapping ID to original rating
    """
    from helpers.green_eval import get_green_rating
    import time

    # Setup paths
    original_ratings_dir = os.path.join(output_dir, 'original_ratings')
    os.makedirs(original_ratings_dir, exist_ok=True)

    # Include model name in filename for clarity
    model_name_clean = clean_model_name(model) if model else "GREEN-radllama2-7b"
    original_ratings_filename = f"original_{model_name_clean}_green_rating.jsonl"
    original_ratings_path = os.path.join(original_ratings_dir, original_ratings_filename)

    # Load existing ratings if file exists
    ratings_dict = {}
    if os.path.exists(original_ratings_path):
        with open(original_ratings_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                ratings_dict[entry['id']] = entry['original_rating']
        print(f"✓ Loaded {len(ratings_dict)} existing original ratings from {original_ratings_filename}")

    # Check which entries are missing
    missing_data = [item for item in data if item['id'] not in ratings_dict]

    if len(missing_data) == 0:
        print(f"✓ All {len(data)} original ratings complete!")
        return ratings_dict

    # Generate missing ratings
    print(f"⚠ {len(missing_data)} original ratings missing (out of {len(data)} total)")
    print(f"  Computing missing ratings...")

    with open(original_ratings_path, 'a') as f:
        for item in missing_data:
            prediction = item[text_field]
            reference = item[reference_field]

            print(f"\nGetting GREEN rating for {item['id']}...")
            start_time = time.time()
            original_rating = get_green_rating(
                prediction, reference,
                model_name=model,
                cpu=cpu,
                num_runs=num_runs
            )
            elapsed_time = time.time() - start_time
            print(f'Time taken: {elapsed_time:.2f} seconds')

            # Build entry
            result = item.copy()
            result['original_rating'] = original_rating

            # Save to file
            json.dump(result, f)
            f.write('\n')

            # Add to dictionary
            ratings_dict[item['id']] = original_rating

    print(f"✓ Generated {len(missing_data)} missing ratings. Total: {len(ratings_dict)}")
    return ratings_dict


def get_or_create_radeval_chexbert_ratings(
    data: List[Dict],
    text_field: str,
    reference_field: str,
    output_dir: str,
    device: str = 'mps'
) -> Dict[str, Dict]:
    """
    Get original CheXbert ratings - load from file if complete, otherwise generate missing ones.

    Args:
        data: List of data entries
        text_field: Field containing text to evaluate (e.g., 'prediction')
        reference_field: Field containing reference text
        output_dir: Output directory
        device: Device to use ('mps', 'cuda', or 'cpu')

    Returns:
        Dictionary mapping ID to original CheXbert rating
    """
    from helpers.chexbert_eval import get_chexbert_rating
    import time

    # Setup paths
    original_ratings_dir = os.path.join(output_dir, 'original_ratings')
    os.makedirs(original_ratings_dir, exist_ok=True)

    # CheXbert doesn't need model name (uses fixed BERT model)
    original_ratings_filename = f"original_chexbert_rating.jsonl"
    original_ratings_path = os.path.join(original_ratings_dir, original_ratings_filename)

    # Load existing ratings if file exists
    ratings_dict = {}
    if os.path.exists(original_ratings_path):
        with open(original_ratings_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                ratings_dict[entry['id']] = entry['original_chexbert_rating']
        print(f"✓ Loaded {len(ratings_dict)} existing original CheXbert ratings from {original_ratings_filename}")

    # Check which entries are missing
    missing_data = [item for item in data if item['id'] not in ratings_dict]

    if len(missing_data) == 0:
        print(f"✓ All {len(data)} original CheXbert ratings complete!")
        return ratings_dict

    # Generate missing ratings
    print(f"⚠ {len(missing_data)} original CheXbert ratings missing (out of {len(data)} total)")
    print(f"  Computing missing ratings...")

    with open(original_ratings_path, 'a') as f:
        for item in missing_data:
            prediction = item[text_field]
            reference = item[reference_field]

            print(f"\nGetting CheXbert rating for {item['id']}...")
            start_time = time.time()
            original_rating = get_chexbert_rating(
                prediction, reference,
                device=device
            )
            elapsed_time = time.time() - start_time
            print(f'Time taken: {elapsed_time:.2f} seconds')

            # Build entry
            result = item.copy()
            result['original_chexbert_rating'] = original_rating

            # Save to file
            json.dump(result, f)
            f.write('\n')

            # Add to dictionary
            ratings_dict[item['id']] = original_rating

    print(f"✓ Generated {len(missing_data)} missing ratings. Total: {len(ratings_dict)}")
    return ratings_dict
