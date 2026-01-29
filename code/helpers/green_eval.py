"""
GREEN (Generation, Ranking, and Evaluation using Expert Notes) evaluation wrapper.

Based on: https://github.com/jbdel/RadEval

GREEN evaluates radiology report generation by comparing generated reports
against reference reports using clinical entity matching.

Supports:
- GREEN's radllama2-7b model (default)
- API models (gpt-4.1-2025-04-14, gpt-4o, gpt-4o-mini, etc.)
"""

import os
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from RadEval import RadEval

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Cache evaluators to avoid reloading models
_evaluator_cache = {}


def get_model_type(model_name):
    """Determine if model is GREEN's LLM or an API model."""
    if model_name is None or model_name == "StanfordAIMI/GREEN-radllama2-7b":
        return "green"
    elif model_name.startswith("gpt-"):
        return "openai"
    elif model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith("gemini-"):
        return "google"
    else:
        # Assume it's a HuggingFace model for GREEN
        return "green"


def get_green_evaluator(model_name=None, cpu=False):
    """
    Get or create a GREEN evaluator with the specified model.

    Args:
        model_name: Model to use. Options:
            - None or "StanfordAIMI/GREEN-radllama2-7b": Use GREEN's radllama2-7b (default)
            - "gpt-4.1-2025-04-14", "gpt-4o", "gpt-4o-mini": Use OpenAI API models
        cpu: If True, run on CPU (for GREEN model only)

    Returns:
        RadEval evaluator instance
    """
    model_type = get_model_type(model_name)
    cache_key = f"{model_name}_{cpu}"

    if cache_key in _evaluator_cache:
        return _evaluator_cache[cache_key]

    if model_type == "green":
        # Use GREEN's radllama2-7b model
        green_model = model_name if model_name else "StanfordAIMI/GREEN-radllama2-7b"
        print(f"Initializing GREEN evaluator with model: {green_model}")
        evaluator = RadEval(
            do_green=True,
            do_radgraph=False,
            do_chexbert=False,
        )
        # Override the model if custom one specified
        if model_name:
            from RadEval.factual.green_score import GREEN
            evaluator.green_scorer = GREEN(green_model, cpu=cpu)
    elif model_type == "openai":
        # Use OpenAI API via MammoGREEN with rate limit handling
        print(f"Initializing GREEN evaluator with OpenAI model: {model_name}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Import MammoGREEN directly
        from RadEval.factual.green_score import MammoGREEN
        evaluator = MammoGREEN(
            model_name=model_name,
            api_key=api_key,
            output_dir="."
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    _evaluator_cache[cache_key] = evaluator
    return evaluator


def compute_green_score(prediction, reference, model_name=None, cpu=False):
    """
    Compute GREEN score for a single prediction-reference pair.

    Args:
        prediction: Generated radiology report text
        reference: Reference radiology report text
        model_name: Model to use for evaluation (default: StanfordAIMI/GREEN-radllama2-7b)
        cpu: If True, run on CPU (for GREEN model only)

    Returns:
        Dictionary with GREEN metrics
    """
    evaluator = get_green_evaluator(model_name, cpu)

    # Run evaluation - returns (mean, std, scores_list, results_df)
    results = evaluator(refs=[reference], hyps=[prediction])
    mean_score, std_score, scores_list, _ = results

    green_score = scores_list[0] if scores_list else mean_score

    return {
        'green_score': float(green_score),
        'model': model_name or "StanfordAIMI/GREEN-radllama2-7b"
    }


def batch_compute_green_scores(predictions, references, model_name=None, cpu=False):
    """
    Compute GREEN scores for multiple prediction-reference pairs.

    Args:
        predictions: List of generated report texts
        references: List of reference report texts
        model_name: Model to use for evaluation
        cpu: If True, run on CPU (for GREEN model only)

    Returns:
        List of score dictionaries
    """
    evaluator = get_green_evaluator(model_name, cpu)

    # Run batch evaluation - returns (mean, std, scores_list, results_df)
    mean_score, std_score, scores_list, _ = evaluator(refs=references, hyps=predictions)

    return [
        {
            'green_score': float(score),
            'model': model_name or "StanfordAIMI/GREEN-radllama2-7b"
        }
        for score in scores_list
    ]


def get_green_rating(prediction, reference, model_name=None, cpu=False, num_runs=1):
    """
    Get GREEN rating with optional averaging over multiple runs.

    For consistency with CQA eval pipeline structure.

    Args:
        prediction: Generated report text
        reference: Reference report text
        model_name: Model to use for evaluation
        cpu: If True, run on CPU (for GREEN model only)
        num_runs: Number of evaluation runs (GREEN is deterministic, so this is mainly for API consistency)

    Returns:
        Dictionary with GREEN metrics
    """
    # GREEN is deterministic, so multiple runs return the same score
    # But we maintain this interface for consistency with CQA pipeline
    score = compute_green_score(prediction, reference, model_name, cpu)

    if num_runs > 1:
        score['_meta'] = {
            'num_runs': num_runs,
            'note': 'GREEN is deterministic - multiple runs return identical scores'
        }

    return score
