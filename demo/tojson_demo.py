from Utils.data_preparing import write_new_data_to_json

German = {
	"name": "German",
	"path": "../Data/german-data.csv",
	"X":[i for i in range(20)],
	"y":20,
	"discrete_columns": [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19],
	"weighted":-1,
}

write_new_data_to_json("../Data/German.json",German)