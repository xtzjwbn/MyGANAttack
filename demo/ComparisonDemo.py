import torch
import numpy as np

from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from pipeline.Comparison.WhiteboxAttackComparision import Comparison
from models.models import netClassificationMLP

import pickle

processor = TabularDataProcessor("../Data/German.json")

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Seek a better way to read models and classes.
target_model = netClassificationMLP(processor.tabular_data.Rtogether.shape[1], processor.tabular_data.nb_classes).to(device)
# target_model_path = f'../models/{processor.name}/{processor.name}_target.pth'
target_model_path = f'../models/{processor.name}/{processor.name}_target_model.pth'
target_model.load_state_dict(torch.load(target_model_path, map_location=device))
target_model.eval()

with open(f"../models/{processor.name}/GAN_Model_{processor.name}.pkl", "rb") as f :
	GAN_Model = pickle.load(f)
f.close()

comparison = Comparison(model = target_model, processor =  processor, gan_model = GAN_Model)

comparison.AddAttackModel("FGSM")
# comparison.AddAttackModel("JSMA")
# comparison.AddAttackModel("Deepfool")
comparison.AddAttackModel("Greedy")
comparison.AddAttackModel("MyGAN")
comparison.StartComparison(processor.tabular_data.Rtogether,processor.tabular_data.y)

print("OK")