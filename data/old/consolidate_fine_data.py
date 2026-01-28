import json
from collections import defaultdict

# File paths
input_file = '/Users/Federica_1/Documents/GitHub/m3-eval/data/old/fine_5pt_expert+llm.json'
output_file = '/Users/Federica_1/Documents/GitHub/m3-eval/data/fine_5pt_expert+llm_consolidated.jsonl'

# Load all entries
print("Loading fine ratings data...")
data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} total entries")

# Group entries by sentence_id
print("\nGrouping entries by sentence_id...")
grouped = defaultdict(list)

for entry in data:
    sentence_id = entry['sentence_id']
    grouped[sentence_id].append(entry)

print(f"Found {len(grouped)} unique sentences")

# Filter for sentences with at least 3 annotations
print("\nFiltering for sentences with at least 3 annotations...")
filtered_grouped = {sid: entries for sid, entries in grouped.items() if len(entries) >= 3}
print(f"Kept {len(filtered_grouped)} sentences with >= 3 annotations")
print(f"Removed {len(grouped) - len(filtered_grouped)} sentences with < 3 annotations")

# Update grouped to use filtered version
grouped = filtered_grouped

# Consolidate annotator ratings
print("\nConsolidating annotator ratings...")
consolidated = []

for sentence_id, entries in grouped.items():
    # Use the first entry as base
    base_entry = entries[0].copy()

    # Remove single annotator fields
    base_entry.pop('annotator', None)
    base_entry.pop('correctness', None)
    base_entry.pop('relevance', None)
    base_entry.pop('safety', None)
    base_entry.pop('confidence', None)
    base_entry.pop('time', None)

    # Remove additional unwanted fields
    base_entry.pop('_id', None)
    base_entry.pop('rated', None)
    base_entry.pop('batch_id', None)

    # Collect all annotator ratings
    annotator_ratings = {}

    for entry in entries:
        annotator_num = entry['annotator']
        annotator_ratings[f'annotator_{annotator_num}'] = {
            'correctness': entry['correctness'],
            'relevance': entry['relevance'],
            'safety': entry['safety'],
            'confidence': entry.get('confidence'),
            'time': entry.get('time')
        }

    # Add consolidated ratings to base entry
    base_entry['annotator_ratings'] = annotator_ratings

    consolidated.append(base_entry)

# Sort by question_id, answer_id, and sentence_id for consistency
consolidated.sort(key=lambda x: (x['question_id'], x['answer_id'], x['sentence_id']))

# Save consolidated data
print(f"\nSaving {len(consolidated)} consolidated entries...")
with open(output_file, 'w') as f:
    for entry in consolidated:
        json.dump(entry, f)
        f.write('\n')

print(f"\nConsolidation complete!")
print(f"Total consolidated entries: {len(consolidated)}")
print(f"Output saved to: {output_file}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Count annotations per sentence
annotation_counts = defaultdict(int)
for entry in consolidated:
    num_annotations = len(entry['annotator_ratings'])
    annotation_counts[num_annotations] += 1

print(f"\nAnnotations per sentence:")
for count in sorted(annotation_counts.keys()):
    print(f"  {count} annotations: {annotation_counts[count]} sentences")

# Print summary by answer_type
answer_types = defaultdict(int)
for entry in consolidated:
    answer_types[entry.get('answer_type', 'unknown')] += 1

print("\nBreakdown by answer_type:")
for answer_type, count in sorted(answer_types.items()):
    print(f"  {answer_type}: {count}")
