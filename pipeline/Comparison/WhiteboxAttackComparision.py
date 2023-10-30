import numpy as np
import torch
import os, csv
from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from pipeline.GAN.GAN_Attack_Model import GAN_Attack_Model

from art.estimators.classification import PyTorchClassifier

from pipeline.Attacks.BaseAttackModel import BaseAttackModel, BaseTorchARTAttackModel, BaseModelAttackModel
from pipeline.Attacks.GreedyAttackModel import GreedyAttackModel
from pipeline.Attacks.LowProFoolAttackModel import LowProFoolAttackModel
from pipeline.Attacks.MyGANAttackModel import MyGANAttackModel
from pipeline.Attacks.AttackAlgorithmsFromART import FGSMAttackModel, JSMAAttackModel, DeepFoolAttackModel

from Utils.Checking import check_none

class Comparison:
	"""
	Methods:
		AddAttackModel : Add Attack Model with name and params and save them in self._attack_algorithm_map. IT IS A DICTIONARY!
		SetData : Set dataset which used to attack
		Attacking_All : Start attack with all attacks in self._attack_algorithm_map.
						And save adv sample data in self._adv_map = {}, self._adv_map_processed = {}. They are also DICTIONARIES.
		ShowComparison : Calculate indexes using the adv data and benign data.
	"""
	def __init__(self,
				 model : torch.nn.Module,
				 processor : TabularDataProcessor = None,
				 gan_model : GAN_Attack_Model = None,
				 art_classifier : PyTorchClassifier = None,
				 fit_finished : bool = True,):
		self._name = processor.name

		self._attack_algorithm_map = {}
		self._adv_map = {}
		self._adv_map_processed = {}
		self._model = model
		self._processor = processor
		self._gan_model = gan_model

		self._x = None
		self._y = None

		# TODO: We need better decoupling.
		if art_classifier is None:
			self._art_classifier = PyTorchClassifier(model=self._model, optimizer=torch.optim.Adam(self._model.parameters()),loss=torch.nn.CrossEntropyLoss(),input_shape=processor.tabular_data.Rtogether.shape[1],nb_classes=2,)

		if not fit_finished:
			self._fit()

	def Attacking_All(self):
		for key in self._attack_algorithm_map:
			self.Attacking_One_With_Name(key)
		return self

	def Attacking_One_With_Name(self, name : str):
		if self._x is None:
			raise ValueError("You need to set data set first!")

		if name not in self._attack_algorithm_map:
			raise ValueError(f"The name '{name}' is not added first!")

		self._adv_map[name] = self._attack_algorithm_map[name].Attack(self._x)
		raw_adv_data = self._adv_map[name]
		processed_adv_real_data = self._processor.data_transformer.inverse_transform(raw_adv_data, self._processor.data_transformer.separate_num)
		processed_adv_data = self._processor.data_transformer.transform(processed_adv_real_data) # normalized
		r_dis, r_con = self._processor.data_transformer.separate_continuous_discrete_columns(processed_adv_data)
		if len(r_con) == 0:
			processed_adv_data = r_dis
		else:
			processed_adv_data = np.column_stack([r_dis, r_con])
		self._adv_map_processed[name] = processed_adv_data

		return self

	def Attacking_One_With_Params(self, name : str, **kwargs):
		saving_name = name
		for key, value in kwargs.items() :
			saving_name += f" {key}: {value}"
		self.Attacking_One_With_Name(saving_name)

		return self

	def Attacking_One_With_Algorithm_Name(self,name : str):
		ans_list = []
		for key, value in self._attack_algorithm_map.items() :
			# if key.find(name) != -1:
			# 	ans_list.append(key)
			if value.name == name:
				ans_list.append(key)


		for key in ans_list :
			self.Attacking_One_With_Name(key)

		return self


	def AddAttackModel(self, name : str, **kwargs):
		saving_name = name
		for key, value in kwargs.items() :
			saving_name += f" {key}={value}"
		if saving_name in self._attack_algorithm_map:
			return
		self._attack_algorithm_map[saving_name]= GetAttack(name = name,
													classifier = self._art_classifier,
													model = self._model,
													processor = self._processor,
													gan_model = self._gan_model,
													**kwargs)
		self._adv_map[saving_name] = None

		return self

	def SetData(self, x_data : np.ndarray, y_data : np.ndarray):
		self._x = x_data
		self._y = y_data

		return self

	def ShowComparison(self):
		# self.Attacking_All()
		check_none(self._x, "Comparison Phrase, data x ")
		check_none(self._y, "Comparison Phrase, data y ")

		predictions = self._art_classifier.predict(self._x)
		pred_benign = np.argmax(predictions, axis = 1)

		accuracy = np.sum(np.argmax(predictions, axis=1) == self._y) / len(self._y)

		print("\n\nAccuracy on benign train examples: {}%\n\n".format(accuracy * 100))

		for key in self._adv_map:
			if self._adv_map[key] is None:
				continue
			predictions_on_raw_adv = self._art_classifier.predict(self._adv_map[key])
			predictions_on_process = self._art_classifier.predict(self._adv_map_processed[key])
			pred_raw = np.argmax(predictions_on_raw_adv, axis = 1)
			pred_process = np.argmax(predictions_on_process, axis = 1)

			accuracy_raw = np.sum(pred_raw == self._y) / len(self._y)
			accuracy_processed = np.sum(pred_process == self._y) / len(self._y)

			success_rate_raw = np.sum(pred_raw != pred_benign) / len(self._y)
			success_rate_processed = np.sum(pred_process != pred_benign) / len(self._y)

			change_rate = np.sum(pred_raw != pred_process) / len(self._y)
			print(f"----------------------------{key}-----------------------------")
			print(f"----Accuracy on RAW adv-examples : {accuracy_raw * 100}%")
			print(f"Success Rate on RAW adv-examples : {success_rate_raw * 100}%")

			print(f"----Accuracy on PROCESSED adv-examples : {accuracy_processed * 100}%")
			print(f"Success Rate on PROCESSED adv-examples : {success_rate_processed * 100}%")

			print(f"Change Rate after processed : {change_rate * 100}%")

			# TODO: better data visualization
			# norm_raw = np.mean(np.linalg.norm(R_test - raw_adv_data, ord = 2, axis = 1))
			# norm_processed = np.mean(np.linalg.norm(R_test - processed_adv_data, ord = 2, axis = 1))
			# continuous_norm_raw = np.mean(
			# 	np.linalg.norm(R_test[:, DT.separate_num :] - raw_adv_data[:, DT.separate_num :], ord = 2, axis = 1))
			# continuous_norm_processed = np.mean(
			# 	np.linalg.norm(R_test[:, DT.separate_num :] - processed_adv_data[:, DT.separate_num :], ord = 2, axis = 1))
			# print(f"L2 Norm on RAW adv-example : {norm_raw}")
			# print(f"L2 Norm on PROCESSED adv-example : {norm_processed}")
			# print(f"L2 Norm on continuous RAW adv-example : {continuous_norm_raw}")
			# print(f"L2 Norm on continuous PROCESSED adv-example : {continuous_norm_processed}")
			# return_data.append(
			# 	[accuracy_raw, success_rate_raw, accuracy_processed, success_rate_processed, change_rate, norm_raw,
			# 	 norm_processed, continuous_norm_raw, continuous_norm_processed])
			print(f"------------------------------------------------------------------------\n\n")

	def AccuracyCalc(self):
		pass

	def _fit(self):
		# TODO: art_classifier fitting
		pass
		# self._art_classifier.fit()
	def AttackNameList(self):
		ans = []
		for key in self._attack_algorithm_map:
			ans.append(self._attack_algorithm_map[key].Name())
		return ans

	def save_adv_data(self, filepath):
		for key in self._adv_map:
			if self._adv_map[key] is None or self._adv_map_processed[key] is None:
				continue
			os.makedirs(filepath + f"/{self._name}", exist_ok = True)
			np.save(filepath + f"/{self._name}_adv_raw_data",self._adv_map[key])
			np.save(filepath + f"/{self._name}_adv_processed_data",self._adv_map_processed[key])

	def Serialize(self, filepath):
		import pickle
		with open(filepath+f"/{self.__class__.__name__}_{self._name}.pkl", "wb") as file :
			pickle.dump(self, file)

	@property
	def attack_algorithm_list(self):
		return self._attack_algorithm_map
	@property
	def adv_map(self):
		return self._adv_map
	@property
	def adv_map_processed(self):
		return self._adv_map_processed

	@property
	def target_model(self):
		return self._model

	@target_model.setter
	def target_model(self,model : torch.nn.Module):
		self._model = model
		self._art_classifier = PyTorchClassifier(model = self._model, loss = torch.nn.CrossEntropyLoss(),
		                                         input_shape = self._processor.tabular_data.Rtogether.shape[1],
		                                         nb_classes = 2, )

	@property
	def x(self):
		return self._x
	@property
	def y(self):
		return self._y

def GetAttack(name : str,
			  classifier : PyTorchClassifier = None,
			  model : torch.nn.Module = None,
			  processor : TabularDataProcessor = None,
			  gan_model : GAN_Attack_Model = None,
			  **kwargs
			  ) -> BaseAttackModel:
	if name == "FGSM":
		if "eps" in kwargs:
			return FGSMAttackModel(classifier, eps = kwargs["eps"])
		return FGSMAttackModel(classifier)
	elif name == "JSMA":
		return JSMAAttackModel(classifier)
	elif name == "Deepfool":
		return DeepFoolAttackModel(classifier)
	if name == "Greedy":
		if "K" in kwargs:
			return GreedyAttackModel(model, processor, K = kwargs["K"])
		return GreedyAttackModel(model, processor)
	elif name == "LowProFool":
		return LowProFoolAttackModel(model)
	elif name == "MyGAN":
		if "K" in kwargs:
			return MyGANAttackModel(model, processor, gan_model, K = kwargs["K"])
		return MyGANAttackModel(model, processor, gan_model)
	else:
		raise ValueError(f"The comparison not supports the Algorithm Name: {name}.")
