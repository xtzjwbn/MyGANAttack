from abc import ABC,abstractmethod
import torch
class BaseModel:
	def __init__(self,
	             name:str,
	             model,
	             optimizer,
	             loss,
	             schedule,
	             device_type = torch.device("cpu")):
		self._name = name
		self._model = model
		self._optimizer = optimizer
		self._loss = loss
		self._schedule = schedule
		self._device = device_type

		self._test_dataloader = None
		self._train_dataLoader = None
		self._size_train = 0
		self._size_test = 0

	@abstractmethod
	def fit(self):
		pass

	@abstractmethod
	def predict(self):
		pass

	@abstractmethod
	def save_model(self, filepath):
		pass

	@abstractmethod
	def load_model(self,filepath):
		pass