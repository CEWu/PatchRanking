import json

def replace_path_in_json(filename, old_path, new_path):
    """
    Reads a JSON file and replaces an old path with a new path in its keys.

    :param filename: Path to the JSON file.
    :param old_path: Path string that needs to be replaced.
    :param new_path: Path string to replace with.
    :return: A dictionary with replaced paths.
    """
    # Open and read the JSON file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Replace old path with new path in each key
    updated_data = {key.replace(old_path, new_path): value for key, value in data.items()}

    return updated_data

# Usage
filename = "ranking_11.json"
old_path = "/home/jlin398/projects"
new_path = "/home/cwu/Workspace/"

updated_data = replace_path_in_json(filename, old_path, new_path)

# If you want to save the updated data back to the JSON file
with open(filename, 'w') as f:
    json.dump(updated_data, f)
