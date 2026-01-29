"""
CheXbert evaluation wrapper for radiology report evaluation.

Based on: https://github.com/jbdel/RadEval

CheXbert extracts 14 clinical conditions from radiology reports using a BERT-based model.
Computes precision, recall, and F1 scores by comparing predicted vs reference conditions.

Supports device specification: 'cuda', 'mps', or 'cpu'
"""

import torch
from typing import Dict, List

# Cache evaluators to avoid reloading models
_evaluator_cache = {}


def get_chexbert_evaluator(device='mps'):
    """
    Get or create a CheXbert evaluator with the specified device.

    Args:
        device: Device to use. Options:
            - 'mps': Apple Silicon GPU (default)
            - 'cuda': NVIDIA GPU
            - 'cpu': CPU

    Returns:
        F1CheXbert evaluator instance
    """
    # Validate device availability
    if device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'

    if device in _evaluator_cache:
        return _evaluator_cache[device]

    print(f"Initializing CheXbert evaluator on device: {device}")
    from RadEval.factual.f1chexbert import F1CheXbert

    evaluator = F1CheXbert(device=device)
    _evaluator_cache[device] = evaluator
    return evaluator


def compute_chexbert_score(prediction, reference, device='mps'):
    """
    Compute CheXbert scores for a single prediction-reference pair.

    Args:
        prediction: Generated radiology report text
        reference: Reference radiology report text
        device: Device to use ('mps', 'cuda', or 'cpu')

    Returns:
        Dictionary with CheXbert metrics including:
        - accuracy: Overall accuracy
        - chexbert_5_micro_f1: Micro F1 for 5 most common conditions
        - chexbert_all_micro_f1: Micro F1 for all 14 conditions
        - chexbert_5_macro_f1: Macro F1 for 5 most common conditions
        - chexbert_all_macro_f1: Macro F1 for all 14 conditions
    """
    evaluator = get_chexbert_evaluator(device)

    # Run evaluation - returns (accuracy, accuracy_per_sample, chexbert_all, chexbert_5)
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = evaluator([prediction], [reference])

    return {
        'accuracy': float(accuracy),
        'chexbert_5_micro_f1': float(chexbert_5["micro avg"]["f1-score"]),
        'chexbert_all_micro_f1': float(chexbert_all["micro avg"]["f1-score"]),
        'chexbert_5_macro_f1': float(chexbert_5["macro avg"]["f1-score"]),
        'chexbert_all_macro_f1': float(chexbert_all["macro avg"]["f1-score"]),
        'chexbert_5_weighted_f1': float(chexbert_5["weighted avg"]["f1-score"]),
        'chexbert_all_weighted_f1': float(chexbert_all["weighted avg"]["f1-score"]),
        'device': device
    }


def batch_compute_chexbert_scores(predictions, references, device='mps'):
    """
    Compute CheXbert scores for multiple prediction-reference pairs.

    Args:
        predictions: List of generated report texts
        references: List of reference report texts
        device: Device to use ('mps', 'cuda', or 'cpu')

    Returns:
        Tuple of (overall_metrics, per_sample_accuracies) where:
        - overall_metrics: Dictionary with aggregate F1 scores
        - per_sample_accuracies: List of accuracy scores per sample
    """
    evaluator = get_chexbert_evaluator(device)

    # Run batch evaluation
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = evaluator(predictions, references)

    overall_metrics = {
        'accuracy': float(accuracy),
        'chexbert_5_micro_f1': float(chexbert_5["micro avg"]["f1-score"]),
        'chexbert_all_micro_f1': float(chexbert_all["micro avg"]["f1-score"]),
        'chexbert_5_macro_f1': float(chexbert_5["macro avg"]["f1-score"]),
        'chexbert_all_macro_f1': float(chexbert_all["macro avg"]["f1-score"]),
        'chexbert_5_weighted_f1': float(chexbert_5["weighted avg"]["f1-score"]),
        'chexbert_all_weighted_f1': float(chexbert_all["weighted avg"]["f1-score"]),
        'device': device
    }

    return overall_metrics, accuracy_per_sample


def get_chexbert_rating(prediction, reference, device='mps'):
    """
    Get CheXbert rating for consistency with GREEN eval pipeline structure.

    Args:
        prediction: Generated report text
        reference: Reference report text
        device: Device to use ('mps', 'cuda', or 'cpu')

    Returns:
        Dictionary with CheXbert metrics
    """
    return compute_chexbert_score(prediction, reference, device)
