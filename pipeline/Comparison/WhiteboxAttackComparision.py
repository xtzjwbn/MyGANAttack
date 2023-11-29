import numpy as np
import torch
import os, csv
import pandas as pd
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

		self._result_success = {}
		self._result_norm = {}

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
		check_none(self._x, "Comparison Phase, data x ")
		check_none(self._y, "Comparison Phase, data y ")

		predictions = self._art_classifier.predict(self._x)
		pred_origin = np.argmax(predictions, axis = 1)

		accuracy = np.sum(np.argmax(predictions, axis=1) == self._y) / len(self._y)

		print("\n\nAccuracy on benign train examples: {}%\n\n".format(accuracy * 100))

		for key in self._adv_map:
			if self._adv_map[key] is None:
				continue
			print(f"----------------------------(\033[0;31m{key}\033[0m)-----------------------------")
			self._result_success[key] = self._SuccessCalc(key, pred_origin)
			self._result_norm[key] = self._NormCalc(key)

			# TODO: better data visualization
			print(f"------------------------------------------------------------------------\n\n")

	def _SuccessCalc(self, key, data_origin):
		pred_raw = self._art_classifier.predict(self._adv_map[key])
		pred_processed = self._art_classifier.predict(self._adv_map_processed[key])
		pred_raw = np.argmax(pred_raw, axis = 1)
		pred_processed = np.argmax(pred_processed, axis = 1)

		accuracy_on_raw = np.sum(pred_raw == self._y) / len(self._y)
		accuracy_on_processed = np.sum(pred_processed == self._y) / len(self._y)
		success_rate_on_raw = np.sum(pred_raw != data_origin) / len(pred_raw)
		success_rate_on_processed = np.sum(pred_processed != data_origin) / len(pred_processed)
		change_rate = np.sum(pred_raw != pred_processed) / len(self._y)

		print(f"----Accuracy on RAW / PROCESSED adv-examples : (\033[0;31m{accuracy_on_raw * 100 :.3f}\033[0m)% ------> (\033[0;31m{accuracy_on_processed * 100 :.3f}\033[0m)%")
		print(f"Success Rate on RAW / PROCESSED adv-examples : (\033[0;31m{success_rate_on_raw * 100 :.3f}\033[0m)% ------> (\033[0;31m{success_rate_on_processed * 100 :.3f}\033[0m)%")
		print(f"Change Rate after processed : (\033[0;31m{change_rate * 100 :.3f}\033[0m)%")
		return {"accuracy_on_raw" : accuracy_on_raw,
		        "success_rate_on_raw" : success_rate_on_raw,
		        "accuracy_on_pro" : accuracy_on_processed,
		        "success_rate_on_pro" : success_rate_on_processed,
		        "change_rate" : change_rate}

	def _NormCalc(self, key):
		pert_raw = self._adv_map[key] - self._x
		pert_processed = self._adv_map_processed[key] - self._x

		all_l0_raw = np.mean(np.linalg.norm(pert_raw, ord = 0, axis = 1))
		all_l1_raw = np.mean(np.linalg.norm(pert_raw, ord = 1, axis = 1))
		all_l2_raw = np.mean(np.linalg.norm(pert_raw, ord = 2, axis = 1))
		all_linf_raw = np.mean(np.linalg.norm(pert_raw, ord = np.inf, axis = 1))
		all_l0_pro = np.mean(np.linalg.norm(pert_processed, ord = 0, axis = 1))
		all_l1_pro = np.mean(np.linalg.norm(pert_processed, ord = 1, axis = 1))
		all_l2_pro = np.mean(np.linalg.norm(pert_processed, ord = 2, axis = 1))
		all_linf_pro = np.mean(np.linalg.norm(pert_processed, ord = np.inf, axis = 1))

		continuous_l0 = np.mean(np.linalg.norm(pert_processed[:,self._processor.tabular_data.separate_num : ], ord = 0, axis = 1))
		continuous_l1 = np.mean(np.linalg.norm(pert_processed[:,self._processor.tabular_data.separate_num : ], ord = 1, axis = 1))
		continuous_l2 = np.mean(np.linalg.norm(pert_processed[:,self._processor.tabular_data.separate_num : ], ord = 2, axis = 1))
		continuous_linf = np.mean(np.linalg.norm(pert_processed[:,self._processor.tabular_data.separate_num : ], ord = np.inf, axis = 1))
		print(f"L0 Norm on RAW/PROCESSED adv-example : (\033[0;31m{all_l0_raw:.4f}\033[0m) ------> (\033[0;31m{all_l0_pro:.4f}\033[0m) ------> on Continuous (\033[0;31m{continuous_l0:.4f}\033[0m)")
		print(f"L1 Norm on RAW/PROCESSED adv-example : (\033[0;31m{all_l1_raw:.4f}\033[0m) ------> (\033[0;31m{all_l1_pro:.4f}\033[0m) ------> on Continuous (\033[0;31m{continuous_l1:.4f}\033[0m)")
		print(f"L2 Norm on RAW/PROCESSED adv-example : (\033[0;31m{all_l2_raw:.4f}\033[0m) ------> (\033[0;31m{all_l2_pro:.4f}\033[0m) ------> on Continuous (\033[0;31m{continuous_l2:.4f}\033[0m)")
		print(f"Linf Norm on RAW/PROCESSED adv-example : (\033[0;31m{all_linf_raw:.4f}\033[0m) ------> (\033[0;31m{all_linf_pro:.4f}\033[0m) ------> on Continuous (\033[0;31m{continuous_linf:.4f}\033[0m)")

		# discrete_change_num_raw = np.mean(np.linalg.norm(pert_processed[:, : self._processor.tabular_data.separate_num], ord = 0, axis = 1))
		discrete_change_num_pro = np.mean(np.linalg.norm(pert_processed[:, : self._processor.tabular_data.separate_num], ord = 0, axis = 1))/2
		print(f"Discrete Change Number : (\033[0;31m{discrete_change_num_pro}\033[0m)")

		return {"all_l0_raw" : all_l0_raw,
		        "all_l1_raw" : all_l1_raw,
		        "all_l2_raw" : all_l2_raw,
		        "all_linf_raw" : all_linf_raw,
		        "all_l0_pro" : all_l0_pro,
		        "all_l1_pro" : all_l1_pro,
		        "all_l2_pro" : all_l2_pro,
		        "all_linf_pro" : all_linf_pro,
		        "continuous_l0" : continuous_l0,
		        "continuous_l1" : continuous_l1,
		        "continuous_l2" : continuous_l2,
		        "continuous_linf" : continuous_linf,
		        "discrete_change_num" : discrete_change_num_pro
		        }

	def _fit(self):
		# TODO: art_classifier fitting
		pass
		# self._art_classifier.fit()
	def AttackNameList(self):
		ans = []
		for key in self._attack_algorithm_map:
			ans.append(self._attack_algorithm_map[key].Name())
		return ans

	def data_frame_output(self):
		df1 = pd.DataFrame.from_dict(self._result_success, orient = 'index')
		df2 = pd.DataFrame.from_dict(self._result_norm, orient = 'index')
		df = pd.concat([df1, df2], axis = 1)
		df['Algorithm'] = df.index
		df = df[['Algorithm'] + [col for col in df.columns if col != 'Algorithm']]
		return df

	def ExcelOutput(self, filepath):
		df = self.data_frame_output()
		with pd.ExcelWriter(filepath+f"/{self.__class__.__name__}_{self._name}.xlsx", engine='xlsxwriter') as writer:
			df.to_excel(writer, sheet_name = 'Summary', index = True)
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
	def name(self):
		return self._name

	@name.setter
	def name(self,name):
		self._name = name

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

	@property
	def result_success(self):
		return self._result_success
	@property
	def result_norm(self):
		return self._result_norm

# TODO: gan_model should be a parameter of GetAttack or a parameter in kwargs?
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
