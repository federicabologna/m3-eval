import json
import os
import sys
sys.path.insert(0, '/Users/Federica_1/Documents/GitHub/m3-eval/code')

from helpers.perturbation_functions import nlp, find_best_matching_sentence

output_dir = '/Users/Federica_1/Documents/GitHub/m3-eval/output'

# Find remove_must_have output files
remove_must_have_files = []
for subdir in os.listdir(output_dir):
    if subdir == 'remove_must_have':
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.endswith('_rating.jsonl'):
                    remove_must_have_files.append(os.path.join(subdir_path, f))

if not remove_must_have_files:
    # Try root directory as fallback
    for f in os.listdir(output_dir):
        if 'remove_must_have' in f and f.endswith('_rating.jsonl'):
            remove_must_have_files.append(os.path.join(output_dir, f))

print("="*80)
print("ANALYZING REMOVE_MUST_HAVE POTENTIAL")
print("="*80)

for filepath in remove_must_have_files:
    print(f"\n{os.path.basename(filepath)}")
    print("-"*80)

    with open(filepath, 'r') as f:
        results = [json.loads(line) for line in f]

    # Analyze each entry
    total_must_have = []
    removable_must_have = []

    for result in results:
        original_answer = result.get('answer', '')
        must_have_list = result.get('Must_have', [])

        if not must_have_list:
            continue

        total_must_have.append(len(must_have_list))

        # Check how many of these could actually be removed
        doc = nlp(original_answer)
        text_sentences = list(doc.sents)

        removable_count = 0
        matches_info = []

        for target_must_have in must_have_list:
            best_match, best_distance, best_index = find_best_matching_sentence(text_sentences, target_must_have)
            threshold = len(target_must_have) * 0.8  # Current threshold

            if best_match and best_distance < threshold:
                removable_count += 1
                matches_info.append({
                    'must_have': target_must_have[:60] + '...' if len(target_must_have) > 60 else target_must_have,
                    'distance': best_distance,
                    'threshold': threshold,
                    'ratio': best_distance / len(target_must_have)
                })

        removable_must_have.append(removable_count)

    if total_must_have:
        print(f"\nTotal entries analyzed: {len(total_must_have)}")

        print(f"\nTotal must_have sentences per answer:")
        print(f"  Average: {sum(total_must_have)/len(total_must_have):.2f}")
        print(f"  Min: {min(total_must_have)}")
        print(f"  Max: {max(total_must_have)}")

        print(f"\nRemovable must_have sentences per answer (with 0.8 threshold):")
        print(f"  Average: {sum(removable_must_have)/len(removable_must_have):.2f}")
        print(f"  Min: {min(removable_must_have)}")
        print(f"  Max: {max(removable_must_have)}")

        # Calculate how many could be removed
        at_least_1 = sum(1 for r in removable_must_have if r >= 1)
        at_least_2 = sum(1 for r in removable_must_have if r >= 2)
        at_least_3 = sum(1 for r in removable_must_have if r >= 3)

        print(f"\nAnswers with at least N removable must_have sentences:")
        print(f"  ≥1: {at_least_1} ({at_least_1/len(removable_must_have)*100:.1f}%)")
        print(f"  ≥2: {at_least_2} ({at_least_2/len(removable_must_have)*100:.1f}%)")
        print(f"  ≥3: {at_least_3} ({at_least_3/len(removable_must_have)*100:.1f}%)")

        # Show distribution
        print(f"\nDistribution of total must_have sentences:")
        from collections import Counter
        dist_total = Counter(total_must_have)
        for count in sorted(dist_total.keys()):
            print(f"  {count} sentences: {dist_total[count]} answers")

        print(f"\nDistribution of removable must_have sentences:")
        dist_removable = Counter(removable_must_have)
        for count in sorted(dist_removable.keys()):
            print(f"  {count} removable: {dist_removable[count]} answers")

        # Show examples with many removable sentences
        print(f"\nExamples with multiple removable must_have sentences:")
        combined = list(zip(total_must_have, removable_must_have, results))
        combined.sort(key=lambda x: x[1], reverse=True)

        for i, (total, removable, result) in enumerate(combined[:3], 1):
            print(f"\n  Example {i}: {total} total, {removable} removable")
            print(f"    Answer ID: {result.get('answer_id')}")
            print(f"    Question: {result.get('question', '')[:100]}...")

            # Re-analyze to show match details
            original_answer = result.get('answer', '')
            must_have_list = result.get('Must_have', [])
            doc = nlp(original_answer)
            text_sentences = list(doc.sents)

            for j, target_must_have in enumerate(must_have_list[:3], 1):  # Show first 3
                best_match, best_distance, best_index = find_best_matching_sentence(text_sentences, target_must_have)
                threshold = len(target_must_have) * 0.8
                removable = best_match and best_distance < threshold

                print(f"    Must-have {j}: {'✓ Removable' if removable else '✗ Not removable'}")
                print(f"      Distance: {best_distance:.0f} / Threshold: {threshold:.0f} ({best_distance/len(target_must_have)*100:.0f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
