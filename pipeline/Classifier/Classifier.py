import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os

class Classifier():
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
        y: np.ndarray,
        batch_size: int = 128,
        epochs: int = 10,
        reg_lambda = 5e-4,
	    weighted = -1,
	    random_seed = 222,):

		if self._model is None :
			raise ValueError("A model is needed to train the model, but none for provided.")
		if self._optimizer is None :
			raise ValueError("An optimizer is needed to train the model, but none for provided.")
		if self._loss is None :
			raise ValueError("A loss function is needed to train the model, but none for provided.")

		self._model.train()

		# x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y)
		self._train_dataLoader, self._test_dataloader, self._size_train, self._size_test = self._dataloader_setting(x, y, batch_size, weighted, random_seed)

		# num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
		# ind = np.arange(len(x_preprocessed))

		loss1 = []
		loss2 = []

		for epoch in range(epochs):
			accuracy_train = 0
			loss_train = 0
			accuracy_test = 0
			loss_test = 0

			self._model.train()
			for (i, (cur_r, cur_y)) in enumerate(self._train_dataLoader):
				cur_r = cur_r.to(self._device)
				cur_y = cur_y.to(self._device)

				pred = self._model(cur_r)
				loss = self._loss(pred, cur_y.long())
				l1_loss = 0
				for name, layer in self._model.named_parameters():
					if "weight" in name:
						l1_loss += torch.sum(abs(layer))
				self._optimizer.zero_grad()
				loss_train += loss.item()
				loss += reg_lambda * l1_loss
				loss.backward()
				self._optimizer.step()
				accuracy_train += (pred.argmax(1) == cur_y).sum()
			if self._schedule is not None:
				self._schedule.step()

			self._model.eval()
			for (i, (cur_r, cur_y)) in enumerate(self._test_dataloader):
				cur_r = cur_r.to(self._device)
				cur_y = cur_y.to(self._device)

				pred = self._model(cur_r)
				loss = self._loss(pred, cur_y.long())

				loss_test += loss.item()
				accuracy_test += (pred.argmax(1) == cur_y).sum()

			loss1.append(loss_train / self._size_train)
			loss2.append(loss_test / self._size_test)
			if (epoch + 1) % 5 == 0 :
				print(
					f"EPOCH:{epoch + 1} lr = {self._optimizer.param_groups[0]['lr']} "
					f"lossTrain:{loss_train / self._size_train:.4f} accTrain:{accuracy_train / self._size_train:.4f} "
					f"lossTest:{loss_test / self._size_test:.4f} accTest:{accuracy_test / self._size_test:.4f}")


	def predict(self, x : np.ndarray,
	            batch_size: int = 128) -> np.ndarray :
		self._model.eval()

		# x_preprocessed,_ = self._apply_preprocessing(x,None)
		x_preprocessed = x

		results_list = []

		num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
		for m in range(num_batch):
			begin, end = (
	            m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )
			with torch.no_grad():
				model_outputs = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
			# output = model_outputs[-1]
			output = model_outputs
			output = output.detach().cpu().numpy().astype(np.float32)
			if len(output.shape) == 1:
				output = np.expand_dims(output.detach().cpu().numpy(), axis=1).astype(np.float32)
			results_list.append(output)

		results = np.vstack(results_list)

		# predictions = self._apply_postprocessing(preds = results, fit = False)

		return results

	def save_model(self, filepath):
		os.makedirs(filepath, exist_ok = True)
		targeted_model_file_name = filepath+f'/{self._name}.pth'
		torch.save(self._model.state_dict(), targeted_model_file_name)

	def load_model(self,filepath):
		"""
		TODO: Maybe not needed to exist.
		"""
		# targetModelPath = f'./models/{data_name}/{data_name}_myGAN_adv_train_model.pth'
		target_model = self._model
		target_model.load_state_dict(torch.load(filepath, map_location = self._device))
		target_model.eval()

	def Serialize(self,filepath):
		import pickle
		with open(filepath+f"/Classifier_{self._name}.pkl", "wb") as file :
			pickle.dump(self, file)

	def _dataloader_setting(self,
	                        X : np.ndarray,
	                        y : np.ndarray,
	                        batch_size : int = 128,
	                        weighted: int = -1,
	                        random_seed : int = 222):

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = random_seed)
		if weighted == -1 :
			train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
			train_data_loader = DataLoader(train_set, batch_size = batch_size)
			test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
			test_data_loader = DataLoader(test_set, batch_size = batch_size)
		else :
			weights_number = weighted
			weights_train = [1 if label == 0 else weights_number for label in y_train]
			weights_test = [1 if label == 0 else weights_number for label in y_test]
			sampler = WeightedRandomSampler(weights_train, len(weights_train), replacement = True)
			sampler2 = WeightedRandomSampler(weights_test, len(weights_test), replacement = True)
			train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
			train_data_loader = DataLoader(train_set, batch_size = batch_size, sampler = sampler)
			test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
			test_data_loader = DataLoader(test_set, batch_size = batch_size, sampler = sampler2)
		return train_data_loader, test_data_loader, len(y_train), len(y_test)

	# def get_size(self):
	# 	return self._size_train,self._size_test
	@property
	def size_train(self):
		return self._size_train
	@property
	def size_test(self):
		return self._size_test
	@property
	def model(self):
		return self._model

	def _apply_preprocessing(self, x, y):
		x_, y_ = x,y
		if not isinstance(x, torch.Tensor) :
			x_ = torch.tensor(x)
		if y is not None and isinstance(y, torch.Tensor):
			y_ = torch.tensor(y)

		return x_,y_