import json
import os

output_dir = '/Users/Federica_1/Documents/GitHub/m3-eval/output'

# Files to check
files_to_check = [
    'change_dosage_coarse_Qwen3-1_7B_rating.jsonl',
    'remove_must_have_coarse_Qwen3-1_7B_rating.jsonl',
]

print("="*80)
print("CHECKING ORIGINAL RATING CONSISTENCY")
print("="*80)

# Load original ratings from each file
original_ratings_by_file = {}

for filename in files_to_check:
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        print(f"\nFile not found: {filename}")
        continue

    original_ratings = {}

    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Use answer_id or sentence_id
            id_key = 'sentence_id' if 'sentence_id' in entry else 'answer_id'
            answer_id = entry[id_key]
            original_rating = entry.get('original_rating', {})

            original_ratings[answer_id] = original_rating

    original_ratings_by_file[filename] = original_ratings
    print(f"\n{filename}")
    print(f"  Loaded {len(original_ratings)} original ratings")

# Find common answer_ids across files
print(f"\n{'='*80}")
print("COMPARING RATINGS FOR COMMON ANSWERS")
print(f"{'='*80}")

file_pairs = [
    ('change_dosage_coarse_Qwen3-1_7B_rating.jsonl',
     'remove_must_have_coarse_Qwen3-1_7B_rating.jsonl'),
]

for file1, file2 in file_pairs:
    if file1 not in original_ratings_by_file or file2 not in original_ratings_by_file:
        continue

    print(f"\nComparing: {os.path.basename(file1)} vs {os.path.basename(file2)}")
    print("-"*80)

    ratings1 = original_ratings_by_file[file1]
    ratings2 = original_ratings_by_file[file2]

    # Find common answer IDs
    common_ids = set(ratings1.keys()) & set(ratings2.keys())

    print(f"Common answer IDs: {len(common_ids)}")

    if len(common_ids) == 0:
        print("  No overlapping answers found")
        continue

    # Compare ratings for common IDs
    differences = []

    for answer_id in sorted(common_ids):
        rating1 = ratings1[answer_id]
        rating2 = ratings2[answer_id]

        # Compare each dimension
        dimensions = ['correctness', 'relevance', 'safety']

        for dim in dimensions:
            score1 = rating1.get(dim, {}).get('score') if isinstance(rating1.get(dim), dict) else rating1.get(dim)
            score2 = rating2.get(dim, {}).get('score') if isinstance(rating2.get(dim), dict) else rating2.get(dim)

            if score1 != score2:
                differences.append({
                    'answer_id': answer_id,
                    'dimension': dim,
                    'file1_score': score1,
                    'file2_score': score2
                })

    if len(differences) == 0:
        print("✓ All original ratings are CONSISTENT across files!")
    else:
        print(f"✗ Found {len(differences)} DIFFERENCES in original ratings:")
        print()

        for i, diff in enumerate(differences[:10], 1):  # Show first 10
            print(f"  {i}. Answer: {diff['answer_id']}")
            print(f"     Dimension: {diff['dimension']}")
            print(f"     {os.path.basename(file1)}: {diff['file1_score']}")
            print(f"     {os.path.basename(file2)}: {diff['file2_score']}")
            print()

        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more differences")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
