#!/usr/bin/env python3
"""
Convert ReXErr clinician-review CSV to JSONL format, compatible with RadEval pipeline.
"""

import csv
import json
import os
import sys
import hashlib

def generate_report_id(original_report, error_report):
    """Generate a unique ID based on report content."""
    # Use first 100 chars of both reports to create a hash
    combined = (original_report[:100] + error_report[:100]).encode('utf-8')
    hash_id = hashlib.md5(combined).hexdigest()[:8]
    return hash_id

def convert_rexerr_review_to_jsonl(review_path, output_path):
    """Convert clinician-review.csv to JSONL format compatible with RadEval pipeline."""
    data = []

    with open(review_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Only include reports marked as acceptable
            if row['acceptable'].strip().lower() == 'yes':
                # Generate unique ID
                report_id = generate_report_id(row['original_report'], row['error_report'])

                # Format compatible with RadEval pipeline
                entry = {
                    'id': f"rexerr_{report_id}",
                    'prediction': row['error_report'],  # Report with errors
                    'reference': row['original_report'],  # Ground truth report
                    'errors_sampled': eval(row['errors_sampled']),  # List of error types
                    'comments': row.get('comments', ''),
                    'source': 'rexerr_clinician_review',
                    'acceptable': 'yes'
                }
                data.append(entry)

    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

    print(f"✓ Converted {len(data)} acceptable examples")
    print(f"✓ Format: compatible with RadEval pipeline")
    print(f"  - 'prediction': error_report (report with injected errors)")
    print(f"  - 'reference': original_report (ground truth)")
    print(f"✓ Saved to: {output_path}")
    return data


if __name__ == '__main__':
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rexerr_dir = os.path.join(
        base_dir,
        'rexerr-v1-clinically-meaningful-chest-x-ray-report-errors-derived-from-mimic-cxr-1.0.0'
    )

    review_path = os.path.join(rexerr_dir, 'clinician-review.csv')
    output_path = os.path.join(os.path.dirname(base_dir), 'rexerr_acceptable_dataset.jsonl')

    print("Converting ReXErr clinician-review to JSONL...")
    print(f"Input: {review_path}")
    print(f"Output: {output_path}")
    print()

    convert_rexerr_review_to_jsonl(review_path, output_path)
