import torch
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import InterpolationMode
import json
import pdb
import random
import os

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

random.seed(10)


def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices

# build transforms


t_resize_crop = transforms.Compose([
    transforms.Resize(size=224, interpolation=INTERPOLATION_MODES['bicubic'], max_size=None, antialias=None),
    transforms.CenterCrop(size=(224, 224)),
])

t_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])



def recover_image(tokens):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(14, 14, 16, 16, 3).swapaxes(1, 2).reshape(224, 224, 3)
    return image

def gen_masked_tokens(tokens, indices, alpha=0.2):
    indices = [i for i in range(196) if i in indices]
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens

# def lerp(val, low, high):
#     """Linear interpolation."""
#     return low + (high - low) * val

# def get_color_from_rank(rank, max_rank):
#     """Get color from rank. Green for low rank and red for high rank."""
#     ratio = rank / max_rank
#     curved_ratio = ratio**2  # Squaring the ratio for more contrast
#     green_intensity = lerp(1 - curved_ratio, 0, 255)
#     red_intensity = lerp(curved_ratio, 0, 255)
#     return red_intensity, green_intensity

# def gen_masked_tokens(tokens, indices, alpha=0.7):
#     max_rank = len(indices)
#     tokens = tokens.copy()
#     for rank, idx in enumerate(indices, 1):  # enumerate starts from 1 for rank 1
#         red, green = get_color_from_rank(rank, max_rank)
#         # Mask the 16x16 RGB patch at the specified index
#         tokens[idx, :, :, 0] = lerp(alpha, tokens[idx, :, :, 0], green)
#         tokens[idx, :, :, 1] = lerp(alpha, tokens[idx, :, :, 1], red)
#         tokens[idx, :, :, 2] = 255 * (1 - alpha)  # Keeping blue channel constant for the entire 16x16 patch

#     return tokens


# import json

# def get_decision(ranking_file, mode="is_keep"):
#     """
#     :param ranking_file: Path to the JSON ranking file.
#     :param mode: 'is_keep' mode will consider the 'is_keep' value. 
#                  'score_only' mode will ignore 'is_keep' and rank based solely on the score.
#     :return: Dictionary with image paths as keys and sorted indices as values.
#     """
#     # Open and read the provided json file
#     with open(ranking_file, 'r') as f:
#         data = json.load(f)

#     # Initialize an empty dictionary to store results
#     imgs_ranking = {}

#     # Loop through each image path in the provided json data
#     for image_path, rankings in data.items():
#         # Create a list to store tuples of index and its corresponding score
#         indices_and_scores = []

#         # Loop through each ranking data for the image
#         for index_str, ranking_data in rankings.items():
#             score = ranking_data['score']

#             # Depending on the mode, consider 'is_keep' value or not
#             if mode == "is_keep" and ranking_data['is_keep'] == 0:
#                 # Convert index string to integer and append to our list along with its score
#                 indices_and_scores.append((int(index_str), score))
#             elif mode == "score_only":
#                 indices_and_scores.append((int(index_str), score))

#         # Sort the list of tuples based on the score (from min to max)
#         sorted_indices = sorted(indices_and_scores, key=lambda x: x[1])
#         # Extract just the indices from the sorted list
#         indices_to_keep = [index for index, _ in sorted_indices]
        
#         # If there are any indices, store them in the results dictionary
#         if indices_to_keep:
#             imgs_ranking[image_path] = indices_to_keep
#     return imgs_ranking

import json

def get_decision(ranking_file, mode="is_keep", list_format=False):
    """
    :param ranking_file: Path to the JSON ranking file.
    :param mode: 'is_keep' mode will consider the 'is_keep' value. 
                 'score_only' mode will ignore 'is_keep' and rank based solely on the score.
    :param list_format: If True, 'is_keep' is treated as a list. Otherwise, it's treated as a dictionary.
    :return: Dictionary with image paths as keys and sorted indices as values.
    """
    # Open and read the provided json file
    with open(ranking_file, 'r') as f:
        data = json.load(f)

    # Initialize an empty dictionary to store results
    imgs_ranking = {}

    # Loop through each image path in the provided json data
    for image_path, rankings in data.items():
        # If 'is_keep' is in list format
        if list_format:
            indices_to_keep = [i for i, value in enumerate(rankings['is_keep']) if value == 0]
            if indices_to_keep:
                imgs_ranking[image_path] = indices_to_keep
            continue

        # For the default dictionary format
        # Create a list to store tuples of index and its corresponding score
        indices_and_scores = []

        # Loop through each ranking data for the image
        for index_str, ranking_data in rankings.items():
            score = ranking_data['score']

            # Depending on the mode, consider 'is_keep' value or not
            if mode == "is_keep" and ranking_data['is_keep'] == 0:
                # Convert index string to integer and append to our list along with its score
                indices_and_scores.append((int(index_str), score))
            elif mode == "score_only":
                indices_and_scores.append((int(index_str), score))

        # Sort the list of tuples based on the score (from min to max)
        sorted_indices = sorted(indices_and_scores, key=lambda x: x[1])

        # Extract just the indices from the sorted list
        indices_to_keep = [index for index, _ in sorted_indices]
        
        # If there are any indices, store them in the results dictionary
        if indices_to_keep:
            imgs_ranking[image_path] = indices_to_keep

    return imgs_ranking





def gen_visualization(image, decisions):
    keep_indices = [decisions]
    image = np.asarray(image)
    image_tokens = image.reshape(14, 16, 14, 16, 3).swapaxes(1, 2).reshape(196, 16, 16, 3)
    stages = [
        recover_image(gen_masked_tokens(image_tokens, keep_indices[i]))
        for i in range(1)
    ]
    viz = np.concatenate([image] + stages, axis=1)
    return viz

def visualize_image_from_json(json_path, dataset_name ,num_images):
    # Load the JSON data

    imgs_decisions = get_decision(json_path, mode="is_keep", list_format=True)

    # Loop for the number of images specified
    for _ in range(num_images):
        # Randomly select one image path from the provided data
        image_path = random.choice(list(imgs_decisions.keys()))
        # Open and process the selected image
        image = Image.open(image_path)
        image = t_resize_crop(image)
        decisions = imgs_decisions[image_path]
        viz = gen_visualization(image, decisions)

        # Plot the visualization
        plt.figure(figsize=(20, 5))
        plt.imshow(viz)
        plt.axis('off')

        # Extract image name from the given path and define the save path
        image_name = os.path.basename(image_path)
        output_dir = "viz_output" + '/' + dataset_name
        save_path = os.path.join(output_dir, image_name +'_viz.jpg')

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the image to the specified directory
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()

# Example usage:
# json_path = '/home/jlin398/projects/GoldenPruningCLIP/CoOp/flower_rankings220153.json'
json_path='CLS_flower_rankings.json'
visualize_image_from_json(json_path, 'oxford_flowers/CLS_flower_rankings', 20)  # Display 20 random images from the json