import json
import os
import random

def shuffle_options(file_path, output_file=None, avoid_label= 'A'):
    """
    Shuffles the options for each question in a JSON file and updates the correct answer label.
    
    Parameters:
        file_path (str): Path to the input JSON file.
        output_file (str, optional): Path for saving the updated file. If not provided,
                                     a new filename with a '_shuffled' suffix will be created.
        avoid_label (str, optional): A label (e.g., "A") to avoid assigning to the correct answer.
                                     If provided, and the correct answer is placed at that label,
                                     it will be swapped with another option.
    """
    # Load the JSON data
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Process each question item in the data
    for item in data:
        options = item['options']
        # Shuffle the options randomly
        random.shuffle(options)
        
        # Create a list of labels based on the number of options (A, B, C, ...)
        letters = [chr(65 + i) for i in range(len(options))]
        
        # If avoid_label is set and valid, ensure the correct answer is not in that position
        if avoid_label:
            try:
                avoid_index = letters.index(avoid_label.upper())
            except ValueError:
                # If avoid_label is not a valid label for the options, ignore it
                avoid_index = None

            if avoid_index is not None and options[avoid_index] == item['correct_answer']:
                # Choose a random index (other than the avoid_index) to swap with
                possible_indices = [i for i in range(len(options)) if i != avoid_index]
                swap_index = random.choice(possible_indices)
                options[avoid_index], options[swap_index] = options[swap_index], options[avoid_index]
        
        # Recreate labels (in case the number of options is not always 4)
        letters = [chr(65 + i) for i in range(len(options))]
        # Find the new index of the correct answer and update its label
        for i, opt in enumerate(options):
            if opt == item['correct_answer']:
                item['correct_answer_label'] = letters[i]
                break
        
        # Update the options in the item with the newly shuffled order
        item['options'] = options
    
    # Generate a new filename if output_file is not provided
    if output_file is None:
        base_name = os.path.splitext(file_path)[0]
        output_file = f"{base_name}_shuffled.json"
    
    # Save the modified data to the new JSON file
    with open(output_file, "w") as out_f:
        json.dump(data, out_f, indent=4)
    
    print(f"File saved successfully as {output_file}")

# Example usage:
# This will shuffle the options and ensure the correct answer is never in the "A" position.
# shuffle_options("questions.json", avoid_label="A")
if __name__ == "__main__":
    file_path = 'data/Harry_Potter_Data_updated.json'
    new_data = shuffle_options(file_path)
