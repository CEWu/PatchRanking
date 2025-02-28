import json

# Load the JSON data from a file
with open('/home/cwu/Workspace/GoldenPruningCLIP/test/caltech101_tokens_rank_10token_6iter_train.json', 'r') as f:
    data = json.load(f)

# Convert the data to a string and perform the replacement
data_str = json.dumps(data)
new_data_str = data_str.replace('/multimodal-prompt-learning/', '/GoldenPruningCLIP/')

# Convert the modified string back to a JSON object
new_data = json.loads(new_data_str)

# Save the modified data to a new JSON file
with open('output_file.json', 'w') as f:
    json.dump(new_data, f)

print("Replacement done and saved to output_file.json")
