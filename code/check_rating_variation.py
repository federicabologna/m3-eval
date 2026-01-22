import json
import os
from collections import Counter

output_dir = '/Users/Federica_1/Documents/GitHub/m3-eval/output'

# Files to check
files_to_check = [
    'change_dosage_coarse_Qwen3-1_7B_rating.jsonl',
    'remove_must_have_coarse_Qwen3-1_7B_rating.jsonl',
    'change_dosage_fine_Qwen3-1_7B_rating.jsonl',
]

print("="*80)
print("CHECKING RATING VARIATION ACROSS ANSWERS")
print("="*80)

for filename in files_to_check:
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        print(f"\nFile not found: {filename}")
        continue

    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print(f"{'='*80}")

    # Collect all ratings with answer IDs
    entries = []

    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            id_key = 'sentence_id' if 'sentence_id' in entry else 'answer_id'
            entries.append({
                'id': entry[id_key],
                'original_rating': entry.get('original_rating', {}),
                'perturbed_rating': entry.get('perturbed_rating', {})
            })

    # Collect scores
    original_scores = {
        'correctness': [],
        'relevance': [],
        'safety': []
    }

    perturbed_scores = {
        'correctness': [],
        'relevance': [],
        'safety': []
    }

    for entry in entries:
        original_rating = entry['original_rating']
        perturbed_rating = entry['perturbed_rating']

        # Extract scores for each dimension
        for dim in ['correctness', 'relevance', 'safety']:
            if dim in original_rating:
                rating_val = original_rating[dim]
                score = rating_val.get('score') if isinstance(rating_val, dict) else rating_val
                if score is not None:
                    original_scores[dim].append(score)

            if dim in perturbed_rating:
                rating_val = perturbed_rating[dim]
                score = rating_val.get('score') if isinstance(rating_val, dict) else rating_val
                if score is not None:
                    perturbed_scores[dim].append(score)

    # Analyze variation for original ratings
    print("\nORIGINAL RATINGS:")
    print("-"*80)

    for dim in ['correctness', 'relevance', 'safety']:
        scores = original_scores[dim]

        if not scores:
            print(f"\n{dim.upper()}: No data")
            continue

        # Calculate statistics
        unique_scores = set(scores)
        score_distribution = Counter(scores)

        print(f"\n{dim.upper()}:")
        print(f"  Total answers: {len(scores)}")
        print(f"  Unique scores: {len(unique_scores)} → {sorted(unique_scores)}")
        print(f"  Min: {min(scores)}, Max: {max(scores)}, Mean: {sum(scores)/len(scores):.2f}")
        print(f"  Distribution:")
        for score in sorted(score_distribution.keys()):
            count = score_distribution[score]
            pct = count / len(scores) * 100
            bar = '█' * int(pct / 2)
            print(f"    Score {score}: {count:3d} ({pct:5.1f}%) {bar}")

    # Analyze variation for perturbed ratings
    print("\n\nPERTURBED RATINGS:")
    print("-"*80)

    for dim in ['correctness', 'relevance', 'safety']:
        scores = perturbed_scores[dim]

        if not scores:
            print(f"\n{dim.upper()}: No data")
            continue

        # Calculate statistics
        unique_scores = set(scores)
        score_distribution = Counter(scores)

        print(f"\n{dim.upper()}:")
        print(f"  Total answers: {len(scores)}")
        print(f"  Unique scores: {len(unique_scores)} → {sorted(unique_scores)}")
        print(f"  Min: {min(scores)}, Max: {max(scores)}, Mean: {sum(scores)/len(scores):.2f}")
        print(f"  Distribution:")
        for score in sorted(score_distribution.keys()):
            count = score_distribution[score]
            pct = count / len(scores) * 100
            bar = '█' * int(pct / 2)
            print(f"    Score {score}: {count:3d} ({pct:5.1f}%) {bar}")

    # Compare original vs perturbed (only for entries with both ratings)
    print("\n\nCOMPARISON (Original vs Perturbed):")
    print("-"*80)

    for dim in ['correctness', 'relevance', 'safety']:
        # Extract paired scores
        paired_scores = []
        for entry in entries:
            orig_rating = entry['original_rating']
            pert_rating = entry['perturbed_rating']

            if dim in orig_rating and dim in pert_rating:
                orig_val = orig_rating[dim]
                pert_val = pert_rating[dim]

                orig_score = orig_val.get('score') if isinstance(orig_val, dict) else orig_val
                pert_score = pert_val.get('score') if isinstance(pert_val, dict) else pert_val

                if orig_score is not None and pert_score is not None:
                    paired_scores.append((orig_score, pert_score))

        if not paired_scores:
            continue

        changes = sum(1 for o, p in paired_scores if o != p)
        increases = sum(1 for o, p in paired_scores if p > o)
        decreases = sum(1 for o, p in paired_scores if p < o)
        same = sum(1 for o, p in paired_scores if p == o)

        print(f"\n{dim.upper()}:")
        print(f"  Paired comparisons: {len(paired_scores)}")
        print(f"  Changed: {changes}/{len(paired_scores)} ({changes/len(paired_scores)*100:.1f}%)")
        print(f"    Increased: {increases} ({increases/len(paired_scores)*100:.1f}%)")
        print(f"    Decreased: {decreases} ({decreases/len(paired_scores)*100:.1f}%)")
        print(f"    Unchanged: {same} ({same/len(paired_scores)*100:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
