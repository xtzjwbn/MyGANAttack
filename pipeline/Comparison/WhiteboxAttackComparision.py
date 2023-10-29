import numpy as np
import torch
from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from pipeline.GAN.GAN_Attack_Model import GAN_Attack_Model

from art.estimators.classification import PyTorchClassifier

from pipeline.Attacks.BaseAttackModel import BaseAttackModel, BaseTorchARTAttackModel, BaseModelAttackModel
from pipeline.Attacks.GreedyAttackModel import GreedyAttackModel
from pipeline.Attacks.LowProFoolAttackModel import LowProFoolAttackModel
from pipeline.Attacks.MyGANAttackModel import MyGANAttackModel
from pipeline.Attacks.AttackAlgorithmsFromART import FGSMAttackModel, JSMAAttackModel, DeepFoolAttackModel

class Comparison:
	def __init__(self,
				 model : torch.nn.Module,
				 processor : TabularDataProcessor = None,
	             gan_model : GAN_Attack_Model = None,
				 art_classifier : PyTorchClassifier = None,
				 fit_finished : bool = True,):
		self._attack_algorithm_map = {}
		self._adv_map = {}
		self._adv_map_processed = {}
		self._model = model
		self._processor = processor
		self._gan_model = gan_model

		# TODO: We need better decoupling.
		if art_classifier is None:
			self._art_classifier = PyTorchClassifier(model=self._model,loss=torch.nn.CrossEntropyLoss(),input_shape=processor.tabular_data.Rtogether.shape[1],nb_classes=2,)

		if not fit_finished:
			self._fit()

	def Attacking_All(self, x_data : np.ndarray):
		for key in self._attack_algorithm_map:
			self.Attacking_One(x_data,key)

	def Attacking_One(self, x_data : np.ndarray, name : str):
		self._adv_map[name] = self._attack_algorithm_map[name].Attack(x_data)
		raw_adv_data = self._adv_map[name]
		processed_adv_real_data = self._processor.data_transformer.inverse_transform(raw_adv_data, self._processor.data_transformer.separate_num)
		processed_adv_data = self._processor.data_transformer.transform(processed_adv_real_data) # normalized
		r_dis, r_con = self._processor.data_transformer.separate_continuous_discrete_columns(processed_adv_data)
		if len(r_con) == 0:
			processed_adv_data = r_dis
		else:
			processed_adv_data = np.column_stack([r_dis, r_con])
		self._adv_map_processed[name] = processed_adv_data

	def AddAttackModel(self, name : str):
		# TODO: Add more parameters
		if name in self._attack_algorithm_map:
			return
		self._attack_algorithm_map[name]= GetAttack(name = name,
													classifier = self._art_classifier,
													model = self._model,
													processor = self._processor,
		                                            gan_model = self._gan_model)
		self._adv_map[name] = -1

	def StartComparison(self,x_data : np.ndarray, y_data : np.ndarray):
		self.Attacking_All(x_data)
		predictions = self._art_classifier.predict(x_data)
		pred_benign = np.argmax(predictions, axis = 1)
		for key in self._adv_map:
			predictions_on_raw_adv = self._art_classifier.predict(self._adv_map[key])
			predictions_on_process = self._art_classifier.predict(self._adv_map_processed[key])
			pred_raw = np.argmax(predictions_on_raw_adv, axis = 1)
			pred_process = np.argmax(predictions_on_process, axis = 1)

			accuracy_raw = np.sum(pred_raw == y_data) / len(y_data)
			accuracy_processed = np.sum(pred_process == y_data) / len(y_data)

			success_rate_raw = np.sum(pred_raw != y_data) / len(y_data)
			success_rate_processed = np.sum(pred_process != y_data) / len(y_data)

			change_rate = np.sum(pred_raw != pred_process) / len(y_data)
			print(f"----------------------------{key}----------------------------------------")
			print(f"----Accuracy on RAW adv-examples : {accuracy_raw * 100}%")
			print(f"Success Rate on RAW adv-examples : {success_rate_raw * 100}%")

			print(f"----Accuracy on PROCESSED adv-examples : {accuracy_processed * 100}%")
			print(f"Success Rate on PROCESSED adv-examples : {success_rate_processed * 100}%")

			print(f"Change Rate after processed : {change_rate * 100}%")

			# TODO: norms and better data visualization
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
	@property
	def attack_algorithm_list(self):
		return self._attack_algorithm_map
	@property
	def adv_map(self):
		return self._adv_map

	@property
	def x(self):
		return self._x
	@x.setter
	def x(self, x_):
		self._x = x_
	@property
	def y(self):
		return self._y
	@y.setter
	def y(self, y_):
		self._y = y_

def GetAttack(name : str,
			  classifier : PyTorchClassifier = None,
			  model : torch.nn.Module = None,
			  processor : TabularDataProcessor = None,
              gan_model : GAN_Attack_Model = None,
			  ) -> BaseAttackModel:
	if name == "FGSM":
		return FGSMAttackModel(classifier)
	elif name == "JSMA":
		return JSMAAttackModel(classifier)
	elif name == "Deepfool":
		return DeepFoolAttackModel(classifier)
	if name == "Greedy":
		return GreedyAttackModel(model, processor)
	elif name == "LowProFool":
		return LowProFoolAttackModel(model)
	elif name == "MyGAN":
		return MyGANAttackModel(model, processor, gan_model)
	else:
		raise ValueError(f"The comparison not supports the Algorithm Name: {name}.")
