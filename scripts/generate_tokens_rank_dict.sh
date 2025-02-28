# #!/bin/bash

# DATASET=$1

# # Generate tokens_rank_dict and save it as a JSON file using Python
# python -c "
# import json
# import os
# import sys

# dataset = sys.argv[1]

# # Create the directory if it does not exist
# directory_path = f'/home/cwu/Workspace/multimodal-prompt-learning/output/ZeroshotCLIP/vit_b16/{dataset}/tokens_rank'
# if not os.path.exists(directory_path):
#     os.makedirs(directory_path)

# # Create the dictionary
# tokens_rank_dict = {i: 0.0 for i in range(1, 197)}

# # Save the dictionary as a JSON file
# json_path = f'/home/cwu/Workspace/multimodal-prompt-learning/output/ZeroshotCLIP/vit_b16/{dataset}/tokens_rank/{dataset}_tokens_rank_v2.json'
# with open(json_path, 'w') as f:
#     json.dump(tokens_rank_dict, f)

# print(f'Dictionary saved as {json_path}')
# " $DATASET


DATASET=$1
TOKENS_PER_GROUP=$2

# Validate that TOKENS_PER_GROUP is a factor of 196
if [ $((196 % $TOKENS_PER_GROUP)) -ne 0 ]; then
  echo "Error: TOKENS_PER_GROUP must be a factor of 196."
  exit 1
fi

# Generate tokens_rank_dict and save it as a JSON file using Python
python -c "
import json
import os
import sys

dataset = sys.argv[1]
tokens_per_group = int(sys.argv[2])

# Create the directory if it does not exist
directory_path = f'/home/cwu/Workspace/multimodal-prompt-learning/output/ZeroshotCLIP/vit_b16/{dataset}/tokens_rank'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Calculate number of groups based on tokens_per_group
num_groups = 196 // tokens_per_group

tokens_rank_dict = {}
for i in range(1, num_groups + 1):
    start_token = (i - 1) * tokens_per_group + 1
    end_token = i * tokens_per_group
    tokens_rank_dict[i] = {
        'token_ids': list(range(start_token, end_token + 1)),
        'accuracy': 0.0
    }

# Save the dictionary as a JSON file
json_path = f'/home/cwu/Workspace/multimodal-prompt-learning/output/ZeroshotCLIP/vit_b16/{dataset}/tokens_rank/{dataset}_tokens_rank_{tokens_per_group}group.json'
with open(json_path, 'w') as f:
    json.dump(tokens_rank_dict, f, indent=4)

print(f'Dictionary saved as {json_path}')
" $DATASET $TOKENS_PER_GROUP
