import argparse
import os
import re
import csv
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(description='Extract accuracies from log files.')
parser.add_argument('base_dir', type=str, help='Base directory path')

# Parse arguments
args = parser.parse_args()

# Use the base_dir from the arguments
base_dir = args.base_dir
output_file = os.path.join(base_dir, 'final_results.txt')

# Extract the last directory name from the base path
directory_name = os.path.basename(base_dir)

# This regex pattern matches strings like "accuracy: 93.1%"
accuracy_pattern = re.compile(r"accuracy: (\d+\.\d+)%")

# Function to extract date from filename
def extract_date(filename):
    date_str = filename.split('-')[-5:]  # Extracts date components from the filename
    date_str = '-'.join(date_str)  # Joins to form 'MM-DD-HH-MM-SS'
    return datetime.strptime(date_str, '%m-%d-%H-%M-%S')

# Create a txt file to store the results
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')

    # Write the extracted directory name first as per requirement
    writer.writerow([directory_name])

    # Walk through the base directory to find all subdirectories
    for dir_name in sorted(next(os.walk(base_dir))[1]):
        dataset_dir = os.path.join(base_dir, dir_name)
        files = sorted(
            (f for f in os.listdir(dataset_dir) if f.startswith('log.txt-2023')),
            key=extract_date
        )
        
        writer.writerow([dir_name])

        for filename in files:
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    match = accuracy_pattern.search(line)
                    if match:
                        accuracy = match.group(1)
                        writer.writerow([accuracy])
                        break

print(f"Results have been saved to {output_file}")


# python collect_res.py /home/cwu/Workspace/GoldenPruningCLIP/output/ZeroshotCLIP_rank/vit_b16/test_1x1_3