#!/usr/bin/env python3
"""
Validate dosages in MedInfo perturbation files.
Checks if recommended_dosage exceeds maximum_safe_dosage using regex patterns.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple


def extract_dosage_mg_per_day(dosage_str: str) -> Optional[float]:
    """
    Extract total daily dosage in mg from a dosage string using regex.

    Handles formats like:
    - "10 mg per 24 hours"
    - "60 mg per day"
    - "10 mg every 4 hours x 6 doses" (calculates 10 * 6 = 60)
    - "500-1000 mg daily" (takes max value)
    - "4000 mg/day"

    Returns:
        Total mg per day as float, or None if can't parse
    """
    if not dosage_str or dosage_str == "N/A":
        return None

    dosage_str = dosage_str.lower()

    # Pattern 1: Direct daily dose (e.g., "60 mg per day", "10 mg per 24 hours", "4000 mg/day")
    pattern_daily = r'(\d+(?:\.\d+)?)\s*mg\s*(?:per|/)\s*(?:day|24\s*hours?)'
    match = re.search(pattern_daily, dosage_str)
    if match:
        return float(match.group(1))

    # Pattern 2: "X mg daily" or "X mg once daily"
    pattern_simple = r'(\d+(?:\.\d+)?)\s*mg\s*(?:once\s*)?daily'
    match = re.search(pattern_simple, dosage_str)
    if match:
        return float(match.group(1))

    # Pattern 3: Range (e.g., "500-1000 mg daily") - take maximum
    pattern_range = r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*mg'
    match = re.search(pattern_range, dosage_str)
    if match:
        return float(match.group(2))  # Take max of range

    # Pattern 4: Frequency-based (e.g., "10 mg every 4 hours x 6 doses")
    pattern_freq = r'(\d+(?:\.\d+)?)\s*mg\s*every.*?(?:x|×)\s*(\d+)\s*doses?'
    match = re.search(pattern_freq, dosage_str)
    if match:
        dose_per_admin = float(match.group(1))
        num_doses = float(match.group(2))
        return dose_per_admin * num_doses

    # Pattern 5: Simple "X mg" with implied daily
    pattern_basic = r'(\d+(?:\.\d+)?)\s*mg'
    match = re.search(pattern_basic, dosage_str)
    if match:
        return float(match.group(1))

    return None


def compare_dosages(recommended: str, maximum_safe: str) -> Tuple[bool, Optional[float], Optional[float], str]:
    """
    Compare recommended dosage against maximum safe dosage.

    Args:
        recommended: Recommended dosage string
        maximum_safe: Maximum safe dosage string

    Returns:
        (is_unsafe, recommended_mg, max_safe_mg, explanation)
    """
    rec_mg = extract_dosage_mg_per_day(recommended)
    max_mg = extract_dosage_mg_per_day(maximum_safe)

    if rec_mg is None or max_mg is None:
        return (False, rec_mg, max_mg, "Could not parse dosages for comparison")

    # Handle zero values (non-dosage errors)
    if max_mg == 0:
        if rec_mg == 0:
            return (False, rec_mg, max_mg, "Both dosages are 0 mg/day (non-dosage error)")
        else:
            return (True, rec_mg, max_mg, f"UNSAFE: Recommended {rec_mg} mg/day with 0 mg/day maximum")

    if rec_mg > max_mg:
        ratio = rec_mg / max_mg
        return (
            True,
            rec_mg,
            max_mg,
            f"UNSAFE: Recommended {rec_mg} mg/day exceeds maximum safe {max_mg} mg/day ({ratio:.1f}x overdose)"
        )
    else:
        return (
            False,
            rec_mg,
            max_mg,
            f"SAFE: Recommended {rec_mg} mg/day is within maximum safe {max_mg} mg/day"
        )


def validate_medinfo_entry(entry: dict) -> dict:
    """
    Validate a MedInfo entry's dosage safety.

    Args:
        entry: MedInfo entry dict with changes_detail

    Returns:
        Validation result dict with safety flags
    """
    results = []

    changes_detail = entry.get('changes_detail', [])

    for change in changes_detail:
        recommended = change.get('recommended_dosage', 'N/A')
        maximum_safe = change.get('maximum_safe_dosage', 'N/A')

        is_unsafe, rec_mg, max_mg, explanation = compare_dosages(recommended, maximum_safe)

        results.append({
            'change_index': change.get('change_index'),
            'is_unsafe': is_unsafe,
            'recommended_mg_per_day': rec_mg,
            'maximum_safe_mg_per_day': max_mg,
            'explanation': explanation,
            'raw_recommended': recommended,
            'raw_maximum_safe': maximum_safe
        })

    return {
        'answer_id': entry.get('answer_id'),
        'focus_drug': entry.get('focus_drug'),
        'validation_results': results,
        'has_unsafe_dosage': any(r['is_unsafe'] for r in results)
    }


def validate_medinfo_file(file_path: Path):
    """Validate all dosages in a MedInfo file."""

    results = {
        'total': 0,
        'unsafe': 0,
        'safe': 0,
        'unparseable': 0,
        'na_dosages': 0,
        'non_dosage_errors': 0,
        'unsafe_details': []
    }

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            results['total'] += 1

            # Validate this entry
            validation = validate_medinfo_entry(entry)

            # Check each change
            for change_result in validation['validation_results']:
                raw_rec = change_result.get('raw_recommended', '')
                raw_max = change_result.get('raw_maximum_safe', '')

                # Check for N/A
                if raw_rec == 'N/A' or raw_max == 'N/A':
                    results['na_dosages'] += 1
                    continue

                # Check for non-dosage errors (both 0 mg/day)
                if change_result['recommended_mg_per_day'] == 0 and change_result['maximum_safe_mg_per_day'] == 0:
                    results['non_dosage_errors'] += 1
                    continue

                # Check if parseable
                if change_result['recommended_mg_per_day'] is None or change_result['maximum_safe_mg_per_day'] is None:
                    results['unparseable'] += 1
                    continue

                # Check safety
                if change_result['is_unsafe']:
                    results['unsafe'] += 1
                    results['unsafe_details'].append({
                        'answer_id': validation['answer_id'],
                        'focus_drug': validation['focus_drug'],
                        'change_index': change_result['change_index'],
                        'recommended': raw_rec,
                        'maximum_safe': raw_max,
                        'recommended_mg': change_result['recommended_mg_per_day'],
                        'maximum_safe_mg': change_result['maximum_safe_mg_per_day'],
                        'explanation': change_result['explanation']
                    })
                else:
                    results['safe'] += 1

    return results


def main():
    """Validate dosages in critical and non-critical files."""

    medinfo_dir = Path('/Users/Federica_1/Documents/GitHub/m3-eval/output/medinfo/perturbations')
    output_dir = Path('/Users/Federica_1/Documents/GitHub/m3-eval/output/medinfo/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_check = [
        'inject_critical_error_coarse.jsonl',
        'inject_noncritical_error_coarse.jsonl'
    ]

    print("=" * 80)
    print("MEDINFO DOSAGE VALIDATION")
    print("=" * 80)

    all_results = {}

    for filename in files_to_check:
        file_path = medinfo_dir / filename

        if not file_path.exists():
            print(f"\n❌ File not found: {filename}")
            continue

        print(f"\n{filename}:")
        print("-" * 80)

        results = validate_medinfo_file(file_path)

        # Store results for JSON output
        perturbation_name = filename.replace('.jsonl', '')
        all_results[perturbation_name] = results

        print(f"Total entries: {results['total']}")
        print(f"Total changes analyzed: {results['unsafe'] + results['safe'] + results['unparseable'] + results['na_dosages'] + results['non_dosage_errors']}")
        print(f"\nDosage-related errors:")
        print(f"  ❌ UNSAFE: {results['unsafe']} (recommended > maximum safe)")
        print(f"  ✓  SAFE: {results['safe']} (recommended ≤ maximum safe)")
        print(f"  ⚠️  Unparseable: {results['unparseable']} (could not extract mg/day)")
        print(f"\nNon-dosage errors:")
        print(f"  N/A dosages: {results['na_dosages']}")
        print(f"  Other (0 mg/day): {results['non_dosage_errors']}")

        if results['unsafe'] > 0:
            print(f"\nFirst 5 UNSAFE examples:")
            for i, detail in enumerate(results['unsafe_details'][:5], 1):
                print(f"\n  {i}. {detail['focus_drug']} (answer_id: {detail['answer_id']}, change {detail['change_index']})")
                print(f"     Recommended: {detail['recommended']}")
                print(f"     Maximum Safe: {detail['maximum_safe']}")
                print(f"     {detail['explanation']}")

    # Save results to JSON file
    output_file = output_dir / 'dosage_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
