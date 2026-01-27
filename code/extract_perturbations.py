#!/usr/bin/env python3
"""
Extract perturbations from existing rating files.

This script converts old rating files (which include both perturbations and ratings)
into new perturbation-only files that can be reused across experiments.

Usage:
    # Extract from specific file
    python code/extract_perturbations.py \
      --input output/add_typos/add_typos_5prob_coarse_Qwen3-8B_rating.jsonl \
      --output output/perturbations/add_typos_05prob_coarse_seed42.jsonl

    # Extract from entire directory
    python code/extract_perturbations.py \
      --input-dir output/add_typos/ \
      --seed 42
"""

import argparse
import json
import os
import sys
import re


def extract_perturbation_from_entry(entry):
    """Extract perturbation data from a rating entry."""
    # Keys to keep (everything except ratings)
    keys_to_remove = [
        'original_rating',
        'perturbed_rating',
        'control_rating',
        'primed_rating',
        'detection_result'
    ]

    # Create clean entry
    clean_entry = {}
    for key, value in entry.items():
        if key not in keys_to_remove:
            clean_entry[key] = value

    # Ensure required fields exist
    if 'perturbed_answer' not in clean_entry:
        return None

    return clean_entry


def parse_filename(filename):
    """
    Parse filename to extract perturbation info.

    Examples:
        add_typos_5prob_coarse_Qwen3-8B_rating.jsonl → (add_typos, 0.5, None, coarse)
        change_dosage_coarse_Qwen3-8B_rating.jsonl → (change_dosage, None, None, coarse)
        remove_must_have_2removed_fine_Qwen3-8B_rating.jsonl → (remove_must_have, None, 2, fine)
    """
    # Remove _rating.jsonl suffix
    name = filename.replace('_rating.jsonl', '').replace('.jsonl', '')

    # Extract level (coarse or fine)
    level_match = re.search(r'_(coarse|fine)_', name)
    level = level_match.group(1) if level_match else 'coarse'

    # Extract perturbation name
    perturbation = name.split('_')[0] + ('_' + name.split('_')[1] if '_' in name and name.split('_')[1] in ['typos', 'dosage', 'must', 'confusion'] else '')
    if 'must_have' in name:
        perturbation = 'remove_must_have'
    elif 'add_typos' in name:
        perturbation = 'add_typos'
    elif 'change_dosage' in name:
        perturbation = 'change_dosage'
    elif 'add_confusion' in name:
        perturbation = 'add_confusion'
    else:
        perturbation = name.split('_')[0]

    # Extract typo probability
    typo_prob = None
    prob_match = re.search(r'_(\d)prob_', name)
    if prob_match:
        typo_prob = float(f"0.{prob_match.group(1)}")

    # Extract num_removed
    num_removed = None
    removed_match = re.search(r'_(\d)removed_', name)
    if removed_match:
        num_removed = int(removed_match.group(1))

    return perturbation, typo_prob, num_removed, level


def extract_from_file(input_path, output_path):
    """Extract perturbations from a single file."""
    if not os.path.exists(input_path):
        print(f"✗ Input file not found: {input_path}")
        return False

    # Parse filename to get info
    filename = os.path.basename(input_path)
    perturbation, typo_prob, num_removed, level = parse_filename(filename)

    print(f"\nProcessing: {filename}")
    print(f"  Perturbation: {perturbation}")
    print(f"  Level: {level}")
    if typo_prob:
        print(f"  Typo probability: {typo_prob}")
    if num_removed:
        print(f"  Num removed: {num_removed}")

    extracted = 0
    skipped = 0

    with open(output_path, 'w') as outfile:
        with open(input_path, 'r') as infile:
            for line in infile:
                entry = json.loads(line)

                # Extract perturbation data
                clean_entry = extract_perturbation_from_entry(entry)

                if clean_entry is None:
                    skipped += 1
                    continue

                # Write to output (preserve seed if it exists, otherwise don't add it)
                json.dump(clean_entry, outfile)
                outfile.write('\n')
                extracted += 1

    print(f"  ✓ Extracted {extracted} perturbations")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} entries")

    return True


def extract_from_directory(input_dir, output_dir):
    """Extract perturbations from all files in a directory."""
    if not os.path.exists(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSONL files
    files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]

    if not files:
        print(f"✗ No JSONL files found in {input_dir}")
        return

    print(f"Found {len(files)} files to process")

    for filename in sorted(files):
        input_path = os.path.join(input_dir, filename)

        # Generate output filename (no seed)
        perturbation, typo_prob, num_removed, level = parse_filename(filename)

        if perturbation == 'remove_must_have' and num_removed:
            output_filename = f"{perturbation}_{num_removed}removed_{level}.jsonl"
        elif perturbation == 'add_typos' and typo_prob:
            prob_str = str(typo_prob).replace('.', '')
            output_filename = f"{perturbation}_{prob_str}prob_{level}.jsonl"
        else:
            output_filename = f"{perturbation}_{level}.jsonl"

        output_path = os.path.join(output_dir, output_filename)

        extract_from_file(input_path, output_path)

    print(f"\n✓ All files processed")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract perturbations from existing rating files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (single file mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (single file mode)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory path (batch mode)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/perturbations',
        help='Output directory (default: output/perturbations)'
    )

    args = parser.parse_args()

    # Single file mode
    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        extract_from_file(args.input, args.output)

    # Directory mode
    elif args.input_dir:
        extract_from_directory(args.input_dir, args.output_dir)

    else:
        parser.error("Either --input or --input-dir must be specified")


if __name__ == "__main__":
    main()
