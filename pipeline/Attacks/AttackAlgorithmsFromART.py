import torch
import torch.nn as nn
import numpy as np

from pipeline.Attacks.BaseAttackModel import BaseTorchARTAttackModel

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod
from art.attacks.evasion.deepfool import DeepFool

class FGSMAttackModel(BaseTorchARTAttackModel):
	def __init__(self, classifier: PyTorchClassifier, eps = 0.3) :
		super().__init__(classifier)
		self._name = "FGSM"
		self._eps = eps

	def Attack(self, x_data : np.ndarray) -> np.ndarray:
		return FastGradientMethod(estimator = self._classifier, eps = self._eps).generate(x=x_data)

class JSMAAttackModel(BaseTorchARTAttackModel):
	def __init__(self, classifier: PyTorchClassifier) :
		super().__init__(classifier)
		self._name = "JSMA"

	def Attack(self,x_data : np.ndarray) -> np.ndarray:
		return SaliencyMapMethod(classifier=self._classifier).generate(x_data)


class DeepFoolAttackModel(BaseTorchARTAttackModel):
	def __init__(self, classifier: PyTorchClassifier) :
		super().__init__(classifier)
		self._name = "Deepfool"

	def Attack(self,x_data : np.ndarray) -> np.ndarray:
		return DeepFool(classifier=self._classifier).generate(x_data)