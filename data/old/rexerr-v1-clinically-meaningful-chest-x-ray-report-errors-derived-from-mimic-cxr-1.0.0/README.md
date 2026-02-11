# ReXErr-V1: Clinically Meaningful Chest X-Ray Report Errors Derived from MIMIC-CXR

Chest X-Ray Report Errors (ReXErr-v1) is a new dataset based on MIMIC-CXR and constructed using large language models (LLMs). ReXErr-v1 contains synthetic error reports for the vast majority of MIMIC-CXR (200k+ reports). For more details on the error insertion process, see full paper here: https://www.worldscientific.com/doi/abs/10.1142/9789819807024_0006

## Error Categories

We insert errors across the following categories, designed in collaboration with clinicians and board-certified radiologists, which encompass both common human and AI-model errors.

AI-Model Errors:
  - Content Addition
    - Add Medical Device
    - False Prediction
    - False Negation
  - Linguistic Quality
    - Add Repetitions
    - Add Contradictions
  - Context-Dependent
    - Change Name of Device
    - Change Position of Device
    - Change Severity
    - Change Location
    - Change Measurement

Human Errors:
  - Content Addition and Context-Dependent
    - Human errors—similar to the AI-generated errors listed above
  - Linguistic Quality
    - Change to Homophone
    - Add Typo

## Error Injection Pipeline

We collaborated with clinicians in an iterative fashion, constructing our prompts for GPT-4o to most accurately and plausibly synthesize errors across the three separate categories sampled. After generating error reports from the MIMIC-CXR dataset, reports were then spliced into individual sentences with post-hoc error labeling performed using Llama 3.1. See the full paper for more information regarding the error sampling strategy, sentence-splicing protocol, and particular prompts used.

## Dataset Contents and Instructions

The dataset consists of two sets of csv files contained within separate folders, along with a separate clinician review file as described below. The first folder contains the train, validation, and test csv files for the report-level errors, while the second folder contains the train, validation, and test csv files for the sentence-level errors. To access, simply load the csv files into python or other csv-viewing interfaces. The data dictionary file contains definitions for all of the column variables, which are also shown below.

**ReXErr-report-level**
  - *ReXErr-report-level_{train/val/test}.csv* contains the original and error reports from a filtered version of the MIMIC-CXR dataset corresponding to the train, val, or test set respectively. Each row contains a unique radiology report, which corresponds to multiple images present within MIMIC-CXR. Reports are listed in ascending subject ID. Each row of the CSV corresponds to the following:
    - dicom_id: Dicom ID(s) for the associated report
    - study_id: Study ID taken from MIMIC-CXR
    - subject_id: Subject ID taken from MIMIC-CXR
    - original_report: Original report taken from MIMIC-CXR
    - error_report: Report with errors injected using GPT-4o
    - errors_sampled: Errors that were sampled to create the error report. Note that the error report may not contain all of the errors sampled, and for more accurate labeling, see the sentence level labeling.

**ReXErr-sentence-level**
  - *ReXErr-sentence-level_{train/val/test}.csv* contains the original and error sentences based on the  ReXErr-report-level.csv file corresponding to the train, val, or test set respectively. Each row contains a sentence present within a radiology report, with spliced sentences presented in the same consecutive order that they appear within the original reports. Groups of sentences corresponding to a particular report are listed in ascending subject ID. Each row of the CSV corresponds to the following:
    - dicom_id: Dicom ID(s) for the associated report
    - study_id: Study ID taken from MIMIC-CXR
    - subject_id: Subject ID taken from MIMIC-CXR
    - original_sentence: Original sentence from the given MIMIC-CXR report
    - error_sentence: Sentence from the error-injected report. Note that the sentence itself may not necessarily contain an error, but it originates from the error-injected report
    - error_present: Indicator for whether an error is present in the sentence, where 0 corresponds to unchanged sentence, 1 corresponds to error sentence, and 2 corresponds to neutral sentence (references a prior or does not contain any clinically relevant indications/findings)
    - error_type: If an error is present within the error_sentence, the specific type of error it is

*clinician-review.csv* contains the results of the manual clinician review conducted on 100 randomly sampled original and error-injected reports from ReXErr. Each row of the csv corresponds to the following:
  - original_report: Original report taken from MIMIC-CXR
  - error_report: Report with errors injected using GPT-4o
  - errors_sampled: Errors that were sampled to create the error report
  - acceptable: Whether the synthetic error report was determined as plausible by the clinician, indicated by a Yes or No
  - comments: Relevant comments when the report is not plausible

## File Organization
```
./
├── ReXErr-report-level
│   ├── ReXErr-report-level_train.csv
│   ├── ReXErr-report-level_val.csv
│   ├── ReXErr-report-level_test.csv
├── ReXErr-sentence-level
│   ├── ReXErr-sentence-level_train.csv
│   ├── ReXErr-sentence-level_val.csv
│   ├── ReXErr-sentence-level_test.csv
│   ├── utils.py
├── README.md
├── clinician-review.csv
├── data-dictionary.txt
```