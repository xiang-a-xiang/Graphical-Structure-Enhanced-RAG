import json
import random
import glob

# Hardcoded values
input_folder = "NLP_project\data\QA_set"
output_file = "NLP_project\data\QA_set\shuffled_QA.json"
num_questions = 20

# Find all JSON files in the specified folder
files = glob.glob(f"{input_folder}/*.json")
all_items = []

# Load items from each JSON file
for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_items.extend(data)
            else:
                print(f"Warning: File {file} does not contain a list. Skipping.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not all_items:
    print("No valid items found in the provided JSON files.")
    exit()

# Shuffle the combined list
random.shuffle(all_items)

# Control the total number of questions in the output
if num_questions > len(all_items):
    print(f"Requested {num_questions} questions, but only found {len(all_items)} items. Using all available items.")
    num_questions = len(all_items)

selected_items = all_items[:num_questions]

# Write the selected items to the output JSON file
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(selected_items, out_f, indent=4, ensure_ascii=False)

print(f"Output file '{output_file}' created with {len(selected_items)} questions.")
