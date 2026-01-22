import json
import os
import re
import sys
sys.path.insert(0, '/Users/Federica_1/Documents/GitHub/m3-eval/code')

from helpers.perturbation_functions import nlp

output_dir = '/Users/Federica_1/Documents/GitHub/m3-eval/output'

# Find change_dosage output files
change_dosage_files = []
for subdir in os.listdir(output_dir):
    if subdir == 'change_dosage':
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.endswith('_rating.jsonl'):
                    change_dosage_files.append(os.path.join(subdir_path, f))

if not change_dosage_files:
    # Try root directory as fallback
    for f in os.listdir(output_dir):
        if 'change_dosage' in f and f.endswith('_rating.jsonl'):
            change_dosage_files.append(os.path.join(output_dir, f))

print("="*80)
print("ANALYZING CHANGE_DOSAGE POTENTIAL")
print("="*80)

for filepath in change_dosage_files:
    print(f"\n{os.path.basename(filepath)}")
    print("-"*80)

    with open(filepath, 'r') as f:
        results = [json.loads(line) for line in f]

    # Analyze each entry
    potential_counts = []
    actual_counts = []

    for result in results:
        original_answer = result.get('answer', '')
        change_counts = result.get('change_counts', {})

        if not change_counts:
            continue

        # Count actual changes made
        actual_total = sum(change_counts.values())
        actual_counts.append(actual_total)

        # Count potential changes by finding all matches in original text
        # Dosage pattern
        dosage_units = r'(mg|mcg|g|mL|ml|L|l|units?|IU|cc|drops?|tablets?|caps?|capsules?|tsp|tbsp|oz)'
        dosage_pattern = r'\b(\d+(?:\.\d+)?)(\s*)(' + dosage_units + r')\b'
        dosage_matches = len(re.findall(dosage_pattern, original_answer, re.IGNORECASE))

        # Time interval pattern
        time_pattern = r'\b(every\s+)(\d+)([\s-](?:to\s+)?)(\d+)(\s+(?:hours?|minutes?|days?|weeks?|months?))\b'
        time_matches = len(re.findall(time_pattern, original_answer, re.IGNORECASE))

        # Anatomical pattern
        body_parts = r'(eyes?|ears?|nostrils?|hands?|arms?|legs?|feet?|knees?|elbows?|cheeks?)'
        anatomical_pattern = r'\b(both|each|one|the)\s+(' + body_parts + r')\b'
        anatomical_matches = len(re.findall(anatomical_pattern, original_answer, re.IGNORECASE))

        # Administration pattern
        admin_pattern = r'\b(don\'t\s+|do\s+not\s+|avoid\s+)?(swallow|chew|crush|dissolve|suck)(\s+(?:the\s+)?(?:tablet|pill|capsule|medication|medicine)s?)?\b'
        admin_matches = len(re.findall(admin_pattern, original_answer, re.IGNORECASE))

        potential_total = dosage_matches + time_matches + anatomical_matches + admin_matches
        potential_counts.append(potential_total)

    if potential_counts:
        print(f"\nTotal entries analyzed: {len(potential_counts)}")
        print(f"\nActual changes made (limited to 3):")
        print(f"  Average: {sum(actual_counts)/len(actual_counts):.2f}")
        print(f"  Min: {min(actual_counts)}")
        print(f"  Max: {max(actual_counts)}")

        print(f"\nPotential changes available:")
        print(f"  Average: {sum(potential_counts)/len(potential_counts):.2f}")
        print(f"  Min: {min(potential_counts)}")
        print(f"  Max: {max(potential_counts)}")

        # Count how many had more than 3 potential changes
        more_than_3 = sum(1 for p in potential_counts if p > 3)
        print(f"\nAnswers with >3 potential changes: {more_than_3} ({more_than_3/len(potential_counts)*100:.1f}%)")

        # Show distribution
        print(f"\nDistribution of potential changes:")
        from collections import Counter
        dist = Counter(potential_counts)
        for count in sorted(dist.keys()):
            print(f"  {count} changes: {dist[count]} answers")

        # Show examples with many potential changes
        print(f"\nExamples with many potential changes:")
        combined = list(zip(potential_counts, actual_counts, results))
        combined.sort(key=lambda x: x[0], reverse=True)

        for i, (potential, actual, result) in enumerate(combined[:3], 1):
            print(f"\n  Example {i}: {potential} potential, {actual} actual")
            print(f"    Answer ID: {result.get('answer_id')}")
            print(f"    Change counts: {result.get('change_counts')}")
            print(f"    Answer preview: {result.get('answer', '')[:150]}...")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
