import json

def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dict if file doesn't exist

def update_json_file(file_path, new_data):
    try:
        with open(file_path, 'r') as file:
            current_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        current_data = {}

    current_data.update(new_data)

    with open(file_path, 'w') as file:
        json.dump(current_data, file, indent=4)
