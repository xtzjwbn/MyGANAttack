import torch
import numpy as np

from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from pipeline.Comparison.WhiteboxAttackComparision import Comparison
from models.models import netClassificationMLP


processor = TabularDataProcessor("../Data/German.json")

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_model = netClassificationMLP(processor.tabular_data.Rtogether.shape[1], processor.tabular_data.nb_classes).to(device)
# target_model_path = f'../models/{processor.name}/{processor.name}_target.pth'
target_model_path = f'../models/{processor.name}/{processor.name}_target_model.pth'
target_model.load_state_dict(torch.load(target_model_path, map_location=device))
target_model.eval()

comparison = Comparison(target_model,processor)

comparison.AddAttackModel("FGSM")
# comparison.AddAttackModel("JSMA")
comparison.AddAttackModel("Deepfool")
comparison.AddAttackModel("Greedy")
comparison.StartComparison(processor.tabular_data.Rtogether,processor.tabular_data.y)

print("OK")