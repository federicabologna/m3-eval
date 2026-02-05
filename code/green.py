import re
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI


def clean_responses(response):
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if (
            "[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>"
        ) in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def make_prompt(text1, text2, max_len: int = 1024):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt


@dataclass
class GreenClientConfig:
    model: str
    api_key: Optional[str] = None
    user: Optional[str] = None
    max_retries: int = 30
    is_async: bool = True
    is_azure: bool = False
    default_headers: Optional[dict] = None

    def to_dict(self, include_user: bool = False, include_model: bool = False):
        output =  {"api_key": self.api_key, "max_retries": self.max_retries}
        if self.user is not None and include_user:
            output["user"] = self.user
        if self.default_headers is not None:
            output["default_headers"] = self.default_headers
        if include_model:
            output["model"] = self.model
        return output


@dataclass
class AzureGreenClientConfig(GreenClientConfig):
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    is_azure: bool = True

    def to_dict(self, **kwargs):
        output = super().to_dict(**kwargs)
        if self.azure_endpoint is not None:
            output["azure_endpoint"] = self.azure_endpoint
        if self.api_version is not None:
            output["api_version"] = self.api_version
        if self.azure_endpoint is not None and self.api_version is not None:
            del output["api_key"]
        return output


@dataclass
class GreenGenerationConfig:
    max_completion_tokens: int = 2048
    temperature: float = 1.0
    n: int = 1


class GREEN:
    def __init__(
        self,
        client_config: GreenClientConfig,
        generation_config: GreenGenerationConfig,
        output_dir: str = ".",
        compute_summary_stats: bool = True
    ):
        super().__init__()

        self.output_dir = output_dir
    
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

        self.generation_config = generation_config
        self.client_config = client_config
        self._init_client()

        self.dataset = None
        self.refs = []
        self.hyps = []

        self.prompts = []
        self.completions = []
        self.green_scores = None
        self.error_counts = None

        self.compute_summary_stats = compute_summary_stats

    def _init_client(self):
        client_config = self.client_config
        is_azure = client_config.is_azure
        is_async = client_config.is_async

        if is_async and is_azure:
            client_cls = AsyncAzureOpenAI
        elif is_async and not is_azure:
            client_cls = AsyncOpenAI
        elif not is_async and is_azure:
            client_cls = AzureOpenAI
        else:  # not is_async and not is_azure
            client_cls = OpenAI

        client_args = client_config.to_dict()

        if is_azure:
            from azure.identity import get_bearer_token_provider, DefaultAzureCredential
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client_args["azure_ad_token_provider"] = token_provider

        self.client = client_cls(**client_args)

    def __call__(self, hyps=None, refs=None):
        print("Processing data...making prompts")

        if refs is None:
            if len(self.refs) > 0:
                refs = self.refs
            else:
                raise ValueError("refs is empty.")
        else:
            self.refs = refs
        if hyps is None:
            if len(self.hyps) > 0:
                hyps = self.hyps
            else:
                raise ValueError("hyps is empty.")
        else:
            self.hyps = hyps
    
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})
        dataset = self.process_data(dataset)
        self.dataset = dataset
        print("Done.")

        for d in tqdm(self.dataset, total=len(self.dataset)):
            local_completions = self.get_response(d["prompt"])
            self.completions.append(local_completions)
            self.prompts.append(d["prompt"])
        mean, std, green_scores, summary, results_df = self.process_results()

        return mean, std, green_scores, summary, results_df

    def update(self, hyp, ref, compute_completion: bool = True):
        self.refs.append(ref)
        self.hyps.append(hyp)

        if compute_completion:
            prompt = make_prompt(ref, hyp)
            local_completions = self.get_response(prompt)
            self.completions.append(local_completions)
            self.prompts.append(prompt)

    async def async_update(self, hyp, ref, compute_completion: bool = True):
        self.refs.append(ref)
        self.hyps.append(hyp)

        if compute_completion:
            prompt = make_prompt(ref, hyp)
            local_completions = await self.async_get_response(prompt)
            self.completions.append(local_completions)
            self.prompts.append(prompt)


    def process_data(self, dataset):
        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }
        dataset = dataset.map(prompting, batched=True)
        return dataset

    async def async_get_response(self, prompt):
        args = {}
        if self.client_config.user is not None:
            args["user"] = self.client_config.user

        outputs = await self.client.chat.completions.create(
            model=self.client_config.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.generation_config.max_completion_tokens,
            temperature=self.generation_config.temperature,
            n=self.generation_config.n,
            **args
        )

        return [clean_responses(r.message.content) for r in outputs.choices]

    def get_response(self, prompt):
        args = {}
        if self.client_config.user is not None:
            args["user"] = self.client_config.user

        outputs = self.client.chat.completions.create(
            model=self.client_config.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.generation_config.max_completion_tokens,
            temperature=self.generation_config.temperature,
            n=self.generation_config.n,
            **args
        )

        return [clean_responses(r.message.content) for r in outputs.choices]

    def compute(self):
        """Compute and return GREEN evaluation results."""
        return self.process_results()

    def process_results(self):
        cats = self.sub_categories + ["Matched Findings"]
        self.error_counts = []
        for i in range(self.generation_config.n):
            self.green_error_count_scores = [
                self.compute_green_and_error_counts(response[i]) for response in self.completions
            ]

            error_counts = pd.DataFrame(
                self.green_error_count_scores,
                columns=["green_score"] + cats,
            )

            self.error_counts.append(error_counts)

        self.error_counts = sum(self.error_counts) / len(self.error_counts)
        self.error_counts[cats] = self.error_counts[cats].apply(round)

        subcats = ["a", "b", "c", "d", "e", "f"]
        colname_map = {
            "green_score": "score",
            **{i: j for i, j in zip(self.sub_categories, subcats)},
            "Matched Findings": "matched_findings",
        }
        output = self.error_counts.rename(colname_map, axis=1).mean(axis=0).to_dict()

        self.results_df = pd.DataFrame(
            {
                "reference": self.refs,
                "predictions": self.hyps,
                "green_analysis": self.completions,
                **self.error_counts,
            }
        )

        self.green_scores = self.error_counts["green_score"].values.tolist()
        self.error_counts = self.error_counts[cats]

        # Compute summary statistics
        mean = np.mean(self.green_scores) if self.green_scores else 0.0
        std = np.std(self.green_scores) if self.green_scores else 0.0

        summary = ""
        if self.compute_summary_stats:
            summary = f"\n-------------{self.client_config.model}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]:\n\n"
            for idx, sub_category in enumerate(self.sub_categories):
                accuracy = output.get(subcats[idx], 0)
                summary += f"{sub_category}: {accuracy}. \n\n"
            summary += "----------------------------------\n"

        return mean, std, self.green_scores, summary, self.results_df

    def compute_green_and_error_counts(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            output = 0
        elif sig_present is None or matched_findings is None:
            output = 0 # TODO: originally None, but would crash the code ...
        else:
            output = matched_findings / (matched_findings + sum(sig_errors))

        return [output] + sig_errors + [matched_findings]

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def compute_summary(self):
        print("Computing summary ...")
        accuracies = self.error_counts.mean(axis=0).to_dict()
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f"\n-------------{self.client_config.model}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
        for idx, sub_category in enumerate(self.sub_categories):
            accuracy = accuracies[sub_category]
            summary += f"{sub_category}: {accuracy}. \n\n"
        summary += "----------------------------------\n"

        return mean, std, summary, self.results_df
