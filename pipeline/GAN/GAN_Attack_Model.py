import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

from Utils.Checking import check_none

import os

import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('TKAgg')

class GAN_Attack_Model:
	"""
	TODO: Annotation
	"""
	def __init__(self,
				 name: str,
				 target_model,
				 auto_encoder_model,
				 generator_model,
				 generator_optimizer,
				 # generator_loss,
				 # generator_schedule,
				 discriminator_model,
				 discriminator_optimizer,
				 # discriminator_loss,
				 # discriminator_schedule,
				 device = torch.device("cpu")) :

		self._name = name
		self._target_model = target_model
		self._auto_encoder_model = auto_encoder_model
		self._eps = 0
		self._alpha_norm = 0
		self._alpha_adv = 0

		self._generator_model = generator_model
		self._generator_optimizer = generator_optimizer
		# self._generator_loss = generator_loss
		# self._generator_schedule = generator_schedule

		self._discriminator_model = discriminator_model
		self._discriminator_optimizer = discriminator_optimizer
		# self._discriminator_loss = discriminator_loss
		# self._discriminator_schedule = discriminator_schedule


		self._device = device

		self._test_dataloader = None
		self._train_dataLoader = None
		self._size_train = 0
		self._size_test = 0

	def fit(self,
		X: np.ndarray,
		R: np.ndarray,
		y : np.ndarray,
		separate_num : int,
		eps:float,
		alpha_norm:float,
		alpha_adv:float,
		batch_size: int = 128,
		epochs: int = 100,
		random_seed = 222,
	    draw_pics = False):

		self._eps = eps
		self._alpha_norm = alpha_norm
		self._alpha_adv = alpha_adv

		check_none(self._target_model, "Target Model")
		check_none(self._auto_encoder_model, "Encoder Model")
		check_none(self._generator_model, "Generator Model")
		check_none(self._generator_optimizer, "Generator Optimizer")
		# check_none(self._generator_loss, "Generator Loss")
		check_none(self._discriminator_model, "Discriminator Model")
		check_none(self._discriminator_optimizer, "Discriminator Optimizer")
		# check_none(self._discriminator_loss, "Discriminator Loss")

		self._target_model.eval()
		self._auto_encoder_model.eval()

		self._generator_model.train()
		self._discriminator_model.train()

		self._train_dataLoader, self._test_dataloader, self._size_train, self._size_test = self._dataloader_setting(X,R,y, batch_size, random_seed)

		loss_D_history = []
		loss_G_history = []
		loss_G_just_history = []
		loss_norm_history = []
		loss_adv_history = []

		for epoch in range(epochs) :
			loss_D_cur = 0
			loss_G_cur = 0
			loss_G_just_cur = 0
			loss_norm_cur = 0
			loss_adv_cur = 0

			for i, (rawX, newX, ty) in enumerate(self._train_dataLoader):
				rawX = rawX.to(torch.float32)
				rawX = rawX.to(self._device)
				tX = rawX
				ty = ty.to(self._device)
				newX = newX.to(torch.float32)
				newX = newX.to(self._device)

				tX_discrete = newX[:, 0: separate_num]
				tX_continuous = tX[:, separate_num:]

				t_latent = self._auto_encoder_model.transform(tX_continuous)

				for stepD in range(1) :
					adv_data = self.myGANattack_only_on_latent(tX_discrete, t_latent)

					self._discriminator_optimizer.zero_grad()
					pred_real = self._discriminator_model(tX)
					pred_fake = self._discriminator_model(adv_data)
					# Gan Standard Loss
					loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device = self._device))
					loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device = self._device))
					loss_D = loss_D_real + loss_D_fake
					loss_D.backward(retain_graph = True)
					self._discriminator_optimizer.step()
					loss_D_cur += loss_D.cpu().detach().numpy()

				for stepG in range(1) :
					adv_data = self.myGANattack_only_on_latent(tX_discrete, t_latent)

					pred_fake = self._discriminator_model(adv_data)
					loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device = self._device))

					perturbationForX = adv_data - tX
					norm = torch.norm(perturbationForX.view(perturbationForX.shape[0], -1), 2, dim = 1)
					loss_norm = torch.mean(norm)

					#TODO: target should be change to another way
					curModelY = self._target_model(tX)
					targetLabel = 1 - curModelY.argmax(dim = 1)
					advAnswer = self._target_model(adv_data)
					crition = torch.nn.CrossEntropyLoss()
					loss_adv = crition(advAnswer, targetLabel.long())

					self._generator_optimizer.zero_grad()
					loss_G = loss_G_fake + self._alpha_adv * loss_adv + self._alpha_norm * loss_norm
					loss_G.backward()
					self._generator_optimizer.step()
					loss_G_cur += loss_G.cpu().detach().numpy()
					loss_G_just_cur += loss_G_fake.cpu().detach().numpy()
					loss_adv_cur += loss_adv.cpu().detach().numpy()
					loss_norm_cur += loss_norm.cpu().detach().numpy()

			loss_D_history.append(loss_D_cur)
			loss_G_history.append(loss_G_cur)
			loss_G_just_history.append(loss_G_just_cur)
			loss_norm_history.append(loss_norm_cur)
			loss_adv_history.append(loss_adv_cur)

			if (epoch + 1) % 5 == 0 :
				print("-------------", epoch + 1, "-----------")
				accuracyTrain = 0
				successTrain = 0
				accuracyAdvTrain = 0
				normTrain = 0
				conNormTrain = 0
				# disTrain = 0
				# disAdvTrain = 0

				accuracyTest = 0
				successTest = 0
				accuracyAdvTest = 0
				normTest = 0
				conNormTest = 0

				for i, (curX, newX, curY) in enumerate(self._train_dataLoader) :
					curX = curX.to(torch.float32)
					curX = curX.to(self._device)
					newX = newX.to(torch.float32)
					newX = newX.to(self._device)
					curY = curY.to(torch.float32)
					curY = curY.to(self._device)
					curX_discrete = newX[:, 0 :separate_num]
					curX_continuous = curX[:, separate_num:]
					cur_latent = self._auto_encoder_model.transform(curX_continuous)
					adv_data = self.myGANattack_only_on_latent(curX_discrete, cur_latent)

					predAdv = self._target_model(adv_data)
					pred = self._target_model(curX)

					accuracyAdvTrain += (predAdv.argmax(1) == curY).sum()
					accuracyTrain += (pred.argmax(1) == curY).sum()
					successTrain += (predAdv.argmax(1) != pred.argmax(1)).sum()
					ans_adv = (adv_data - curX).cpu().detach().numpy()
					ans_norm = np.linalg.norm(ans_adv, ord = 2, axis = 1)
					continuous_norm = np.mean(abs(ans_adv[:, separate_num:]), axis = 1)
					# real_ans_adv = ans_adv * Ci_between
					# real_ans_adv = ans_adv * 1
					# real_ans_norm = np.linalg.norm(real_ans_adv, ord=2, axis=1)
					# normTrain += torch.mean(torch.norm(adv_data - curX, 2, dim=1))
					normTrain += np.sum(ans_norm)
					conNormTrain += np.sum(continuous_norm)
					# realNormTrain += np.sum(real_ans_norm)

				for i, (curX, newX, curY) in enumerate(self._test_dataloader) :
					curX = curX.to(torch.float32)
					curX = curX.to(self._device)
					newX = newX.to(torch.float32)
					newX = newX.to(self._device)
					curY = curY.to(torch.float32)
					curY = curY.to(self._device)
					curX_discrete = newX[:, 0 :separate_num]
					curX_continuous = curX[:, separate_num :]
					cur_latent = self._auto_encoder_model.transform(curX_continuous)

					adv_data = self.myGANattack_only_on_latent(curX_discrete, cur_latent)

					# check_curX = curX.detach().numpy()  # Debug
					# check_curX = check_curX * Ci_between  # Debug
					# check_adv_data = adv_data.detach().numpy()  # Debug
					# check_adv_data = check_adv_data * Ci_between  # Debug
					# check_curX = curX.cpu().detach().numpy()  # Debug
					# check_curX = check_curX * 1  # Debug
					# check_adv_data = adv_data.cpu().detach().numpy()  # Debug
					# check_adv_data = check_adv_data * 1  # Debug

					predAdv = self._target_model(adv_data)
					pred = self._target_model(curX)

					accuracyAdvTest += (predAdv.argmax(1) == curY).sum()
					accuracyTest += (pred.argmax(1) == curY).sum()
					successTest += (predAdv.argmax(1) != pred.argmax(1)).sum()
					ans_adv = (adv_data - curX).cpu().detach().numpy()
					ans_norm = np.linalg.norm(ans_adv, ord = 2, axis = 1)
					continuous_norm = np.mean(abs(ans_adv[:, separate_num :]), axis = 1)
					# real_ans_adv = ans_adv * Ci_between
					# real_ans_adv = ans_adv * 1
					# real_ans_norm = np.linalg.norm(real_ans_adv, ord=2, axis=1)
					# normTest += torch.mean(torch.norm(adv_data - curX, 2, dim=1))
					normTest += np.sum(ans_norm)
					conNormTest += np.sum(continuous_norm)
					# realNormTest += np.sum(real_ans_norm)

				print(
					f"accTrain:{accuracyTrain / self._size_train:.4f} accAdvTrain:{accuracyAdvTrain / self._size_train:.4f} "
					f"sucTrain:{successTrain / self._size_train:.4f} normTrain:{normTrain / self._size_train:.4f} "
					f"continuousNormTrain:{conNormTrain / self._size_train}")
				print(
					f"accTest:{accuracyTest / self._size_test:.4f} accAdvTest:{accuracyAdvTest / self._size_test:.4f} "
					f"sucTest:{successTest / self._size_test:.4f} normTest:{normTest / self._size_test:.4f} "
					f"continuousNormTest:{conNormTest / self._size_test}")
		if draw_pics:
			plt.figure()
			plt.plot(loss_D_history, label = 'loss_D')
			# plt.plot(loss_G_history, label= 'loss_G_tot')
			plt.plot(loss_G_just_history, label = 'loss_G_fake')
			plt.plot(loss_norm_history, label = 'loss_norm')
			plt.plot(loss_adv_history, label = 'loss_adv')
			plt.legend(loc = 0)
			plt.show()

	def _dataloader_setting(self,
							X : np.ndarray,
							R : np.ndarray,
							y : np.ndarray,
							batch_size : int = 128,
							random_seed : int = 222):

		X_train, X_test, R_train, R_test, y_train, y_test  = train_test_split(X, R, y, test_size = 0.1, random_state = random_seed)
		train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(R_train), torch.from_numpy(y_train))
		train_data_loader = DataLoader(train_set, batch_size = batch_size)
		test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(R_test), torch.from_numpy(y_test))
		test_data_loader = DataLoader(test_set, batch_size = batch_size)
		return train_data_loader, test_data_loader, y_train.shape[0], y_test.shape[0]

	def myGANattack_only_on_latent(self, x_discrete, x_latent) :
		self._generator_model.eval()
		perturbation_for_latent = self._generator_model(torch.cat((x_discrete, x_latent), dim = 1))
		perturbation_for_latent = torch.clamp(perturbation_for_latent, -self._eps, self._eps)
		adv_latent = perturbation_for_latent + x_latent
		adv_continuous = self._auto_encoder_model.inverse_transform(adv_latent)
		adv_data = torch.cat((x_discrete, adv_continuous), dim = 1)
		adv_data = torch.clamp(adv_data, 0, 1)
		self._generator_model.train()
		return adv_data

	def save_model(self,filepath):
		os.makedirs(filepath, exist_ok = True)
		generator_model_file_name = filepath+f'/{self._name}_generator_model.pth'
		torch.save(self._generator_model.state_dict(), generator_model_file_name)
		discriminator_model_file_name = filepath+f'/{self._name}_discriminator_model.pth'
		torch.save(self._discriminator_model.state_dict(), discriminator_model_file_name)

	def Serialize(self,filepath):
		import pickle
		with open(filepath+f"/{self._name}.pkl", "wb") as file :
			pickle.dump(self, file)

	@property
	def size_train(self) :
		return self._size_train
	@property
	def size_test(self) :
		return self._size_test

	@property
	def alpha_adv(self):
		return self._alpha_adv
	@alpha_adv.setter
	def alpha_adv(self, value):
		if value <= 0:
			raise ValueError("alpha_adv must be non-negative.")
		self._alpha_adv = value

	@property
	def alpha_norm(self):
		return self._alpha_norm
	@alpha_norm.setter
	def alpha_norm(self, value):
		if value <= 0:
			raise ValueError("alpha_norm must be non-negative.")
		self.alpha_norm = value

	@property
	def eps(self):
		return self._eps
	@eps.setter
	def eps(self, value):
		if value <= 0:
			raise ValueError("eps must be non-negative.")
		self.eps = value

	@property
	def discriminator_model(self) :
		return self._discriminator_model
	@property
	def generator_model(self):
		return self._generator_model
	@property
	def target_model(self):
		return self._target_model
	@property
	def auto_encoder_model(self):
		return self._auto_encoder_model
