import pickle
def load_(name):
	with open(f"./models/{name}/DataTransformer", "rb") as f:
		DT = pickle.load(f)
	f.close()
	return DT
