
import json

file_name = "output/ZeroshotCLIP_rank/vit_b16/oxford_pets_test/tokens_rank/oxford_pets_tokens_rank_10token_6iter.json"
# Load the contents of the JSON file
with open(file_name, "r") as file:
    oxford_pets_data = json.load(file)


def rank_patches(patches_data):
    """Function to rank patches based on the final logic"""
    # Separate the patches based on "is_keep" value
    kept_patches = []
    discarded_patches = []

    for patch_index, patch_data in patches_data.items():
        if patch_data['is_keep'] == 1:
            kept_patches.append((patch_index, patch_data))
        else:
            discarded_patches.append((patch_index, patch_data))

    # Rank patches with "is_keep" = 1
    kept_patches.sort(key=lambda x: x[1]['score'], reverse=True)

    # Rank patches with "is_keep" = 0
    discarded_patches.sort(key=lambda x: (x[1]['iter'], x[1]['score']), reverse=True)

    # Combine the two ranked lists
    ranked_patches = kept_patches + discarded_patches

    # Extract the desired lists
    is_keep_dict = {int(patch_index): patch_data['is_keep'] for patch_index, patch_data in patches_data.items()}
    is_keep_list_corrected = [is_keep_dict[i] for i in range(len(is_keep_dict))]
    
    # Generate the score list based on the original order of the patches
    score_dict = {int(patch_index): patch_data['score'] for patch_index, patch_data in patches_data.items()}
    score_list_corrected = [score_dict[i] for i in range(len(score_dict))]
    
    patch_index_list = [int(patch_index) for patch_index, _ in ranked_patches]

    return is_keep_list_corrected, score_list_corrected, patch_index_list




# Apply the ranking logic to each key in the JSON file
ranked_data = {}
for key, value in oxford_pets_data.items():
    is_keep_list, score_list, patch_index_list = rank_patches(value)
    ranked_data[key] = {
        'is_keep': is_keep_list,
        'score': score_list,
        'patch_index': patch_index_list
    }

# Write the ranked data to a new JSON file
output_file_path = 'test/oxford_pets_tokens_rank_10token_6iter_avg_test.json'
with open(output_file_path, 'w') as outfile:
    json.dump(ranked_data, outfile)

output_file_path
