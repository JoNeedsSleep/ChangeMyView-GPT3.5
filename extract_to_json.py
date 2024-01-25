import json

# File paths for the training and test data
train_fname = "train_pair_data.jsonlist"
test_fname = "heldout_pair_data.jsonlist"

# Function to read a .jsonlist file
def read_jsonlist(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]
    
# Function to save data to a JSON file
def save_as_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Load the training and test data
original_posts_train = read_jsonlist(train_fname)
original_posts_test = read_jsonlist(test_fname)

# Save as JSON files
save_as_json(original_posts_train, "train_op_data.json")
save_as_json(original_posts_test, "heldout_op_data.json")

print(len(original_posts_train))
print(len(original_posts_test))