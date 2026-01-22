import json

# Define the mapping from text to integers
likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

# Files to process
files_to_process = [
    {
        'input': '/Users/Federica_1/Documents/GitHub/m3-eval/data/coarse_5pt_expert+llm.json',
        'output': '/Users/Federica_1/Documents/GitHub/m3-eval/data/coarse_5pt_expert+llm_converted.jsonl'
    },
    {
        'input': '/Users/Federica_1/Documents/GitHub/m3-eval/data/fine_5pt_expert+llm.json',
        'output': '/Users/Federica_1/Documents/GitHub/m3-eval/data/fine_5pt_expert+llm_converted.jsonl'
    }
]

# Process each file
for file_info in files_to_process:
    input_file = file_info['input']
    output_file = file_info['output']

    print(f"\nProcessing {input_file}...")

    # Read the JSONL file and convert ratings
    converted_count = 0
    total_entries = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            total_entries += 1

            # Convert correctness_llm
            if 'correctness_llm' in entry and entry['correctness_llm'] in likert_map:
                entry['correctness_llm'] = likert_map[entry['correctness_llm']]
                converted_count += 1

            # Convert relevance_llm
            if 'relevance_llm' in entry and entry['relevance_llm'] in likert_map:
                entry['relevance_llm'] = likert_map[entry['relevance_llm']]
                converted_count += 1

            # Convert safety_llm
            if 'safety_llm' in entry and entry['safety_llm'] in likert_map:
                entry['safety_llm'] = likert_map[entry['safety_llm']]
                converted_count += 1

            # Write the converted entry to output file
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f"Conversion complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Total ratings converted: {converted_count}")
    print(f"Output saved to: {output_file}")

print("\n" + "="*80)
print("ALL FILES PROCESSED SUCCESSFULLY")
