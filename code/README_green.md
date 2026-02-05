# GREEN: Generative Radiology Error Evaluation with LLMs

## Overview

GREEN (Generative Radiology Error Evaluation) is a Python evaluation framework that uses Large Language Models (LLMs) to automatically assess the accuracy of radiology reports by comparing candidate reports against reference reports composed by expert radiologists.

## Features

- **LLM-based Evaluation**: Leverages OpenAI or Azure OpenAI models to perform detailed error analysis
- **Comprehensive Error Classification**: Categorizes errors into clinically significant and insignificant categories
- **Detailed Error Analysis**: Identifies six types of errors including false findings, missing findings, misidentification, severity misassessment, and comparison issues
- **Batch and Streaming Processing**: Supports both synchronous and asynchronous processing modes
- **Flexible Client Configuration**: Works with both standard OpenAI and Azure OpenAI endpoints
- **Statistical Analysis**: Computes mean scores, standard deviations, and detailed error count statistics

## Installation

### Requirements

```bash
pip install openai numpy pandas datasets tqdm azure-identity
```

### Dependencies

- `openai` - OpenAI Python client
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `datasets` - Hugging Face datasets library
- `tqdm` - Progress bars
- `azure-identity` - Azure authentication (for Azure OpenAI)

## Usage

### Basic Example

```python
from green import GREEN, GreenClientConfig, GreenGenerationConfig

# Configure the OpenAI client
client_config = GreenClientConfig(
    model="gpt-4",
    api_key="your-api-key",
    is_async=False,
    is_azure=False
)

# Configure generation parameters
generation_config = GreenGenerationConfig(
    max_completion_tokens=2048,
    temperature=1.0,
    n=1
)

# Initialize GREEN
green = GREEN(
    client_config=client_config,
    generation_config=generation_config,
    output_dir="./results"
)

# Evaluate reports
reference_reports = ["Reference report text..."]
candidate_reports = ["Candidate report text..."]

mean, std, green_scores, summary, results_df = green(
    refs=reference_reports,
    hyps=candidate_reports
)

print(f"Mean GREEN Score: {mean:.4f} ± {std:.4f}")
print(summary)
```

### Azure OpenAI Example

```python
from green import GREEN, AzureGreenClientConfig, GreenGenerationConfig

# Configure Azure OpenAI client
client_config = AzureGreenClientConfig(
    model="gpt-4",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-15-preview",
    is_async=False,
    is_azure=True
)

generation_config = GreenGenerationConfig(
    max_completion_tokens=2048,
    temperature=1.0,
    n=1
)

green = GREEN(client_config, generation_config)
```

### Incremental Updates

```python
# Add reports one at a time
green.update(
    hyp="Candidate report text",
    ref="Reference report text",
    compute_completion=True
)

# Compute results
results = green.compute()
```

### Async Processing

```python
import asyncio

# Configure for async mode
client_config = GreenClientConfig(
    model="gpt-4",
    api_key="your-api-key",
    is_async=True
)

green = GREEN(client_config, generation_config)

# Use async update
async def process_reports():
    await green.async_update(
        hyp="Candidate report",
        ref="Reference report"
    )

asyncio.run(process_reports())
```

## Error Categories

GREEN evaluates reports across the following dimensions:

### Clinically Significant Errors
1. **(a)** False report of a finding in the candidate
2. **(b)** Missing a finding present in the reference
3. **(c)** Misidentification of a finding's anatomic location/position
4. **(d)** Misassessment of the severity of a finding
5. **(e)** Mentioning a comparison that isn't in the reference
6. **(f)** Omitting a comparison detailing a change from a prior study

### Clinically Insignificant Errors
Same six categories as above, but for errors with minimal clinical impact.

### Matched Findings
Number of findings correctly identified in both reports.

## GREEN Score Calculation

The GREEN score is computed as:

```
GREEN = matched_findings / (matched_findings + sum(clinically_significant_errors))
```

A score of 1.0 indicates perfect accuracy with no clinically significant errors.

## Output

The framework produces:

1. **Mean GREEN Score**: Average accuracy across all evaluated reports
2. **Standard Deviation**: Variability in scores
3. **Error Counts DataFrame**: Detailed breakdown of errors by category
4. **Summary Statistics**: Mean error counts per category
5. **Results DataFrame**: Complete results with references, predictions, and GREEN analysis

### Results DataFrame Columns

- `reference`: Reference report text
- `predictions`: Candidate report text
- `green_analysis`: LLM-generated analysis
- `green_score`: Individual GREEN score
- Error count columns for each subcategory
- `matched_findings`: Count of matched findings

## Configuration

### GreenClientConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | Model name (e.g., "gpt-4") |
| `api_key` | str | None | OpenAI API key |
| `user` | str | None | User identifier for tracking |
| `max_retries` | int | 30 | Maximum retry attempts |
| `is_async` | bool | True | Enable async mode |
| `is_azure` | bool | False | Use Azure OpenAI |
| `default_headers` | dict | None | Custom HTTP headers |

### AzureGreenClientConfig

Inherits from `GreenClientConfig` with additional parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `azure_endpoint` | str | None | Azure OpenAI endpoint URL |
| `api_version` | str | None | Azure API version |

### GreenGenerationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_completion_tokens` | int | 2048 | Maximum tokens in response |
| `temperature` | float | 1.0 | Sampling temperature |
| `n` | int | 1 | Number of completions |

## Methods

### `__call__(hyps, refs)`
Main evaluation method. Processes lists of candidate and reference reports.

### `update(hyp, ref, compute_completion=True)`
Add a single report pair for evaluation.

### `async_update(hyp, ref, compute_completion=True)`
Async version of update method.

### `compute()`
Compute results from accumulated reports.

### `process_results()`
Process and analyze LLM responses to extract error counts and GREEN scores.

## Notes

- Reports are truncated to 1024 tokens by default in prompt generation
- Azure OpenAI authentication uses DefaultAzureCredential
- The framework focuses on clinical findings rather than writing style
- Multiple completions (n > 1) are averaged for more robust scoring
