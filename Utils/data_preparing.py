import json


def write_new_data_to_json(filepath,data):
	with open(filepath, 'w') as json_file :
		json.dump(data, json_file, indent = 4)

def read_data_from_json(filepath):
	with open(filepath, 'r') as json_file :
		data = json.load(json_file)
	return data
