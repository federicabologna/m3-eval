import json
from collections import defaultdict

# File paths
input_file = '/Users/Federica_1/Documents/GitHub/m3-eval/data/old/coarse_5pt_expert+llm.jsonl'
kqa_file = '/Users/Federica_1/Documents/GitHub/m3-eval/data/old/kqa_qa_pairs.jsonl'
output_file = '/Users/Federica_1/Documents/GitHub/m3-eval/data/coarse_5pt_expert+llm_consolidated.jsonl'

# Load Must_have statements
print("Loading Must_have statements...")
must_have_dict = {}
with open(kqa_file, 'r') as f:
    for line in f:
        qa_pair = json.loads(line)
        qa_id = qa_pair['id']
        must_have_dict[qa_id] = qa_pair['Must_have']

print(f"Loaded Must_have for {len(must_have_dict)} questions")

# Group entries by answer_id
print("\nGrouping entries by answer_id...")
grouped = defaultdict(list)

with open(input_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        answer_id = entry['answer_id']
        grouped[answer_id].append(entry)

print(f"Found {len(grouped)} unique answers")

# Consolidate annotator ratings
print("\nConsolidating annotator ratings...")
consolidated = []

for answer_id, entries in grouped.items():
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

    # Add Must_have if answer_type is physician
    if base_entry.get('answer_type') == 'physician':
        # Extract numeric ID from question_id
        question_id = base_entry['question_id']
        try:
            numeric_id = int(question_id.replace('question_', ''))
            if numeric_id in must_have_dict:
                base_entry['Must_have'] = must_have_dict[numeric_id]
            else:
                print(f"Warning: No Must_have found for {question_id}")
                base_entry['Must_have'] = []
        except ValueError:
            print(f"Warning: Could not parse question_id: {question_id}")
            base_entry['Must_have'] = []

    consolidated.append(base_entry)

# Sort by question_id and answer_type for consistency
consolidated.sort(key=lambda x: (x['question_id'], x.get('answer_type', '')))

# Save consolidated data
print(f"\nSaving {len(consolidated)} consolidated entries...")
with open(output_file, 'w') as f:
    for entry in consolidated:
        json.dump(entry, f)
        f.write('\n')

print(f"\nConsolidation complete!")
print(f"Total consolidated entries: {len(consolidated)}")
print(f"Output saved to: {output_file}")

# Print summary by answer_type
answer_types = defaultdict(int)
for entry in consolidated:
    answer_types[entry.get('answer_type', 'unknown')] += 1

print("\nBreakdown by answer_type:")
for answer_type, count in sorted(answer_types.items()):
    print(f"  {answer_type}: {count}")
