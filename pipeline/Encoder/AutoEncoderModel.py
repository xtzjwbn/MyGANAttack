import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import os

import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('TKAgg')

class AutoEncoderModel:
	"""
	TODO: Annotation
	"""
	def __init__(self,
	             name:str,
	             model,
	             optimizer,
	             loss,
	             schedule,
	             device = torch.device("cpu")):
		self._name = name
		self._model = model
		self._optimizer = optimizer
		self._loss = loss
		self._schedule = schedule
		self._device = device

		self._test_dataloader = None
		self._train_dataLoader = None
		self._size_train = 0
		self._size_test = 0

	def fit(self,
        x: np.ndarray,
        batch_size: int = 128,
        epochs: int = 100,
	    random_seed = 222,):

		if self._model is None :
			raise ValueError("A model is needed to train the model, but none for provided.")
		if self._optimizer is None :
			raise ValueError("An optimizer is needed to train the model, but none for provided.")
		if self._loss is None :
			raise ValueError("A loss function is needed to train the model, but none for provided.")

		self._model.train()

		self._train_dataLoader, self._test_dataloader, self._size_train, self._size_test = self._dataloader_setting(x, batch_size, random_seed)

		loss_history = []
		norm_history = []

		for epoch in range(epochs) :
			self._model.train()
			loss_cur = []
			norm_cur = []

			for i, rawX in enumerate(self._train_dataLoader):
				rawX = rawX[0]
				rawX = rawX.to(torch.float32)
				rawX = rawX.to(self._device)
				tX = rawX
				tX_check = rawX

				self._optimizer.zero_grad()
				encoded, decoded = self._model(tX)

				loss = self._loss(decoded, tX_check)
				loss.backward()
				self._optimizer.step()

				loss_cur.append(loss.cpu().detach().numpy())
				real_decoded_cur = decoded
				norm_calculate = np.linalg.norm((real_decoded_cur - rawX).cpu().detach().numpy(), ord=2, axis=1)
				norm_cur.append(np.mean(norm_calculate))

			if self._schedule is not None :
				self._schedule.step()
			loss_history.append(np.mean(loss_cur))
			norm_history.append(np.mean(norm_cur))

			if (epoch + 1) % 20 == 0 :
				if (epoch + 1) == 500 :
					print("OK")
				self._model.eval()
				loss_train = []
				norm_train = []
				loss_test = []
				norm_test = []
				for i, rawX in enumerate(self._train_dataLoader) :
					rawX = rawX[0]
					rawX = rawX.to(torch.float32)
					rawX = rawX.to(self._device)
					tX = rawX
					tX_check = rawX

					encoded, decoded = self._model(tX)
					loss = self._loss(decoded, tX_check)
					loss_train.append(loss.cpu().detach().numpy())
					real_decoded_cur = decoded
					norm_calculate = np.linalg.norm((real_decoded_cur - rawX).cpu().detach().numpy(), ord = 2, axis = 1)
					norm_train.append(np.mean(norm_calculate))

				for i, rawX in enumerate(self._test_dataloader) :
					rawX = rawX[0]
					rawX = rawX.to(torch.float32)
					rawX = rawX.to(self._device)
					tX = rawX
					tX_check = rawX

					encoded, decoded = self._model(tX)
					loss = self._loss(decoded, tX_check)
					loss_test.append(loss.cpu().detach().numpy())
					real_decoded_cur = decoded
					norm_calculate = np.linalg.norm((real_decoded_cur - rawX).cpu().detach().numpy(), ord = 2, axis = 1)
					norm_test.append(np.mean(norm_calculate))

				print(f"---------EPOCH:{epoch + 1}---------")
				print(f"loss_train:{np.mean(np.array(loss_train))}  norm_train:{np.mean(np.array(norm_train))}")
				print(f"loss_test :{np.mean(np.array(loss_test))}  norm_test :{np.mean(np.array(norm_test))}")
		plt.figure()
		plt.plot(loss_history)
		plt.show()

	def save_model(self, filepath):
		os.makedirs(filepath, exist_ok = True)
		targeted_model_file_name = filepath+f'/{self._name}_target_model.pth'
		torch.save(self._model.state_dict(), targeted_model_file_name)

	def _dataloader_setting(self,
	                        X : np.ndarray,
	                        batch_size : int = 128,
	                        random_seed : int = 222):

		X_train, X_test = train_test_split(X, test_size = 0.1, random_state = random_seed)
		train_set = TensorDataset(torch.from_numpy(X_train))
		train_data_loader = DataLoader(train_set, batch_size = batch_size)
		test_set = TensorDataset(torch.from_numpy(X_test))
		test_data_loader = DataLoader(test_set, batch_size = batch_size)
		return train_data_loader, test_data_loader, X_train.shape[0], X_test.shape[0]
