import numpy as np
import torch.nn
from art.estimators.classification import PyTorchClassifier

class BaseAttackModel :
	def __init__(self) :
		pass

	def Attack(self,x_data : np.ndarray) -> np.ndarray:
		return x_data

	def Name(self) -> str:
		return ""

class BaseTorchARTAttackModel(BaseAttackModel):
	"""
	The Base attack model of ART Attacks.
	:param classifier: PyTorchClassifier
	"""
	def __init__(self, classifier: PyTorchClassifier) :
		super().__init__()
		self._classifier = classifier

	@property
	def classifier(self):
		return self._classifier

class BaseModelAttackModel(BaseAttackModel):
	"""
	The Base attack model not relies on ART.

	:param model: PyTorch Networks Model
	"""
	def __init__(self, model: torch.nn.Module) :
		super().__init__()
		self._model = model

	@property
	def model(self):
		return self._model

