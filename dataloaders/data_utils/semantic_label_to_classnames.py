import json


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


json_file_path = './semantic_labels.json'
label_data = load_json(json_file_path)

label_info = label_data['28_categories']
for i in range(len(label_info)):
    print('\''+label_info[i]['name']+'\',')
