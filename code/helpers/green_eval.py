"""
GREEN (Generation, Ranking, and Evaluation using Expert Notes) evaluation wrapper.

Based on: https://github.com/jbdel/RadEval

GREEN evaluates radiology report generation by comparing generated reports
against reference reports using clinical entity matching.
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path


def compute_green_score(prediction, reference, green_path=None):
    """
    Compute GREEN score for a single prediction-reference pair.

    Args:
        prediction: Generated radiology report text
        reference: Reference radiology report text
        green_path: Path to GREEN evaluation script (optional)

    Returns:
        Dictionary with GREEN metrics
    """
    # TODO: Integrate actual GREEN evaluation code
    # For now, return placeholder structure

    # GREEN typically returns metrics like:
    # - Precision: How many predicted entities are correct
    # - Recall: How many reference entities were found
    # - F1: Harmonic mean of precision and recall

    return {
        'green_score': 0.0,  # Placeholder
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'note': 'GREEN evaluation not yet integrated'
    }


def batch_compute_green_scores(predictions, references, green_path=None):
    """
    Compute GREEN scores for multiple prediction-reference pairs.

    Args:
        predictions: List of generated report texts
        references: List of reference report texts
        green_path: Path to GREEN evaluation script

    Returns:
        List of score dictionaries
    """
    scores = []
    for pred, ref in zip(predictions, references):
        score = compute_green_score(pred, ref, green_path)
        scores.append(score)

    return scores


def get_green_rating(prediction, reference, num_runs=1):
    """
    Get GREEN rating with optional averaging over multiple runs.

    For consistency with CQA eval pipeline structure.

    Args:
        prediction: Generated report text
        reference: Reference report text
        num_runs: Number of evaluation runs (GREEN is deterministic, so this is mainly for API consistency)

    Returns:
        Dictionary with GREEN metrics
    """
    # GREEN is deterministic, so multiple runs return the same score
    # But we maintain this interface for consistency with CQA pipeline
    score = compute_green_score(prediction, reference)

    if num_runs > 1:
        score['_meta'] = {
            'num_runs': num_runs,
            'note': 'GREEN is deterministic - multiple runs return identical scores'
        }

    return score


# Placeholder for when GREEN repo is integrated
def setup_green_evaluator(repo_path=None):
    """
    Setup GREEN evaluator by cloning/loading the RadEval repository.

    Args:
        repo_path: Path to RadEval repository

    Returns:
        Path to GREEN evaluation module
    """
    if repo_path is None:
        # Default to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        repo_path = project_root / 'external' / 'RadEval'

    if not repo_path.exists():
        print(f"RadEval repository not found at {repo_path}")
        print("Please clone it with:")
        print(f"  git clone https://github.com/jbdel/RadEval.git {repo_path}")
        return None

    return repo_path


# TODO: Integration steps for actual GREEN evaluation:
# 1. Clone RadEval repo: git clone https://github.com/jbdel/RadEval.git
# 2. Install dependencies from RadEval repo
# 3. Import GREEN evaluation functions
# 4. Implement compute_green_score() to call actual GREEN code
# 5. Handle entity extraction and matching
