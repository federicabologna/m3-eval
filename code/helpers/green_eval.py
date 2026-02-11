"""
GREEN (Generative Radiology Error Evaluation) evaluation wrapper.

Uses the GREEN framework from green.py to evaluate radiology report generation
by comparing generated reports against reference reports using LLM-based error analysis.

Supports:
- OpenAI API models (gpt-4.1-2025-04-14, gpt-4o, gpt-4o-mini, etc.)
- Azure OpenAI models

Requirements:
- pip install openai numpy pandas datasets tqdm azure-identity
"""

import os
import sys
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import green.py
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

try:
    from green import GREEN, GreenClientConfig, AzureGreenClientConfig, GreenGenerationConfig
except ImportError as e:
    raise ImportError(
        f"Failed to import GREEN modules: {e}\n"
        "Please install required packages:\n"
        "pip install openai numpy pandas datasets tqdm azure-identity"
    )

# Load environment variables from .env file in project root
project_root = code_dir.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Cache evaluators to avoid recreating clients
_evaluator_cache = {}


def get_green_evaluator(
    model_name: str = "gpt-4o",
    is_azure: bool = False,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: float = 1.0,
    max_completion_tokens: int = 2048,
    n: int = 5,
    error_priming: bool = False
) -> GREEN:
    """
    Get or create a GREEN evaluator with the specified model.

    Args:
        model_name: Model to use (e.g., "gpt-4o", "gpt-4.1-2025-04-14")
        is_azure: Whether to use Azure OpenAI
        azure_endpoint: Azure OpenAI endpoint URL (if using Azure)
        api_version: Azure API version (if using Azure)
        temperature: Sampling temperature (default: 1.0)
        max_completion_tokens: Maximum tokens in response (default: 2048)
        n: Number of completions per API call for averaging (default: 5)
        error_priming: Add note that candidate report contains errors (default: False)

    Returns:
        GREEN evaluator instance
    """
    cache_key = f"{model_name}_{is_azure}_{azure_endpoint}_{temperature}_{n}_{error_priming}"

    if cache_key in _evaluator_cache:
        return _evaluator_cache[cache_key]

    print(f"Initializing GREEN evaluator with model: {model_name}")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not is_azure:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Configure client
    if is_azure:
        if not azure_endpoint:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_version:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        client_config = AzureGreenClientConfig(
            model=model_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            is_async=False,
            is_azure=True,
            max_retries=30
        )
    else:
        client_config = GreenClientConfig(
            model=model_name,
            api_key=api_key,
            is_async=False,
            is_azure=False,
            max_retries=30
        )

    # Configure generation
    generation_config = GreenGenerationConfig(
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        n=n  # Number of completions to average (default: 5)
    )

    # Create evaluator
    evaluator = GREEN(
        client_config=client_config,
        generation_config=generation_config,
        output_dir=".",
        compute_summary_stats=False,
        error_priming=error_priming
    )

    _evaluator_cache[cache_key] = evaluator
    return evaluator


def compute_green_score(
    prediction: str,
    reference: str,
    model_name: str = "gpt-4o",
    is_azure: bool = False,
    temperature: float = 1.0,
    n: int = 5,
    error_priming: bool = False
) -> Dict:
    """
    Compute GREEN score for a single prediction-reference pair.

    Args:
        prediction: Generated radiology report text
        reference: Reference radiology report text
        model_name: Model to use for evaluation (default: gpt-4o)
        is_azure: Whether to use Azure OpenAI
        temperature: Sampling temperature
        n: Number of completions to average (default: 5)
        error_priming: Add note that candidate report contains errors (default: False)

    Returns:
        Dictionary with GREEN score and error analysis (averaged across n completions)
    """
    evaluator = get_green_evaluator(
        model_name=model_name,
        is_azure=is_azure,
        temperature=temperature,
        n=n,
        error_priming=error_priming
    )

    # Use update method for single evaluation
    evaluator.update(hyp=prediction, ref=reference, compute_completion=True)

    # Compute results (returns mean, std, scores, summary, df)
    _mean, _std, scores, _summary, _df = evaluator.compute()

    # Extract score for the single example (averaged)
    green_score = scores[-1] if scores else 0.0
    error_counts = evaluator.error_counts.iloc[-1].to_dict() if len(evaluator.error_counts) > 0 else {}
    completion = evaluator.completions[-1][0] if evaluator.completions else ""

    # Extract individual scores from all n completions before averaging
    # evaluator.error_counts is already averaged, but we need to access the pre-averaged data
    # The GREEN class stores individual error_counts in a list during process_results()
    # We'll extract individual scores from results_df if available
    individual_scores = []
    if hasattr(evaluator, 'results_df') and len(evaluator.results_df) > 0:
        # Get the last entry's green_analysis (list of n completions)
        last_completions = evaluator.completions[-1] if evaluator.completions else []
        # Recompute individual GREEN scores from each completion
        for completion_text in last_completions:
            indiv_score_data = evaluator.compute_green_and_error_counts(completion_text)
            individual_scores.append(float(indiv_score_data[0]))  # First element is the green score

    return {
        'score': float(green_score),
        'model': model_name,
        'error_counts': error_counts,
        'analysis': completion,
        'individual_scores': individual_scores  # Add list of n individual scores
    }


def batch_compute_green_scores(
    predictions: List[str],
    references: List[str],
    model_name: str = "gpt-4o",
    is_azure: bool = False,
    temperature: float = 1.0,
    n: int = 5
) -> List[Dict]:
    """
    Compute GREEN scores for multiple prediction-reference pairs.

    Args:
        predictions: List of generated report texts
        references: List of reference report texts
        model_name: Model to use for evaluation
        is_azure: Whether to use Azure OpenAI
        temperature: Sampling temperature
        n: Number of completions to average per example (default: 5)

    Returns:
        List of score dictionaries (each averaged across n completions)
    """
    evaluator = get_green_evaluator(
        model_name=model_name,
        is_azure=is_azure,
        temperature=temperature,
        n=n
    )

    # Run batch evaluation - now returns (mean, std, scores_list, summary, results_df)
    mean_score, std_score, scores_list, summary, results_df = evaluator(refs=references, hyps=predictions)

    # Extract individual scores
    results = []
    for i, green_score in enumerate(scores_list):
        error_counts = evaluator.error_counts.iloc[i].to_dict()
        completion = evaluator.completions[i][0] if i < len(evaluator.completions) else ""

        # Extract individual scores from all n completions for this example
        individual_scores = []
        if i < len(evaluator.completions):
            completions_for_example = evaluator.completions[i]
            for completion_text in completions_for_example:
                indiv_score_data = evaluator.compute_green_and_error_counts(completion_text)
                individual_scores.append(float(indiv_score_data[0]))

        results.append({
            'score': float(green_score),
            'model': model_name,
            'error_counts': error_counts,
            'analysis': completion,
            'individual_scores': individual_scores  # Add list of n individual scores
        })

    return results


def get_green_rating(
    prediction: str,
    reference: str,
    model_name: str = "gpt-4o",
    cpu: bool = False,  # Kept for API compatibility, not used
    num_runs: int = 5,  # Now uses n parameter for single API call with 5 completions
    is_azure: bool = False,
    temperature: float = 1.0,
    error_priming: bool = False
) -> Dict:
    """
    Get GREEN rating with averaging over multiple completions.

    Uses OpenAI's n parameter to get multiple completions in a single API call.
    The green.py script automatically averages them.

    For consistency with CQA eval pipeline structure.

    Args:
        prediction: Generated report text
        reference: Reference report text
        model_name: Model to use for evaluation (default: gpt-4o)
        cpu: Unused (kept for API compatibility)
        num_runs: Number of completions to average via n parameter (default: 5)
        is_azure: Whether to use Azure OpenAI
        temperature: Sampling temperature
        error_priming: Add note that candidate report contains errors (default: False)

    Returns:
        Dictionary with GREEN metrics (automatically averaged across n completions)
    """
    # Use n parameter for efficient single-call averaging
    result = compute_green_score(
        prediction=prediction,
        reference=reference,
        model_name=model_name,
        is_azure=is_azure,
        temperature=temperature,
        n=num_runs,  # Pass num_runs as n parameter
        error_priming=error_priming
    )

    # Add metadata about averaging
    result['num_runs'] = num_runs
    result['_note'] = f'Averaged across {num_runs} completions from single API call (n={num_runs})'

    return result
