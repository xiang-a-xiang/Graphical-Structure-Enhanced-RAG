import json
import os

def adding_label(file_path, output_file=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    letters = ['A', 'B', 'C', 'D']
    for item in data:
        for i in range(len(item['options'])):
            if item['correct_answer'] == item['options'][i]:
                item['correct_answer_label'] = letters[i]
    
    # Generate a new filename if not provided
    if output_file is None:
        base_name = os.path.splitext(file_path)[0]  # Get file name without extension
        output_file = f"{base_name}_updated.json"  # Append '_updated' to the filename
    
    # Save the modified data to a new JSON file
    with open(output_file, "w") as out_f:
        json.dump(data, out_f, indent=4)
    
    print(f"File saved successfully as {output_file}")

            
        

if __name__ == "__main__":
    file_path = 'data/Harry_Potter_Data.json'
    new_data = adding_label(file_path)
