import torch
import torch.nn as nn
import numpy as np

from pipeline.Attacks.BaseAttackModel import BaseTorchARTAttackModel

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod
from art.attacks.evasion.deepfool import DeepFool

class FGSMAttackModel(BaseTorchARTAttackModel):
	def __init__(self, classifier: PyTorchClassifier) :
		super().__init__(classifier)

	def Attack(self, x_data : np.ndarray, eps = 0.3) -> np.ndarray:
		return FastGradientMethod(estimator = self._classifier, eps = eps).generate(x=x_data)

	def Name(self) -> str:
		return "FGSM"


class JSMAAttackModel(BaseTorchARTAttackModel):
	def __init__(self, classifier: PyTorchClassifier) :
		super().__init__(classifier)

	def Attack(self,x_data : np.ndarray) -> np.ndarray:
		return SaliencyMapMethod(classifier=self._classifier).generate(x_data)
	def Name(self) -> str:
		return "JSMA"


class DeepFoolAttackModel(BaseTorchARTAttackModel):
	def __init__(self, classifier: PyTorchClassifier) :
		super().__init__(classifier)

	def Attack(self,x_data : np.ndarray) -> np.ndarray:
		return DeepFool(classifier=self._classifier).generate(x_data)
	def Name(self) -> str:
		return "Deepfool"