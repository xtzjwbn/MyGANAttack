from pipeline.Preparing.data_transformer import DataTransformer
from Utils.data_preparing import read_data_from_json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TableDataStruct:
	#############################################
	# TODO: Optimize memory usage in this section
	#############################################
	X: np.ndarray
	y: np.ndarray
	nb_classes: int
	discrete_columns: List[str]
	separate_num : int
	R : np.ndarray
	Rdis : np.ndarray
	Rcon : np.ndarray
	Rtogether : np.ndarray
	Rnew : np.ndarray


class TabularDataProcessor:
	"""
	Transforming raw data to R data or inversing them.

	Variables:
	name -> name of dataset
	scale_type -> minmax or standard, otherwise
	discrete_columns -> record the discrete column of dataset
	data_transformer -> the DataTransformer of dataset
	tabular_data -> record tabular data

	Methods:
	__data_processing -> set self.tabular_data
	__data_transformer_setting -> set the DataTransformer
	"""
	def __init__(self,filepath,scale_type = "minmax"):
		self._data_info = read_data_from_json(filepath)

		name = self._data_info["name"]
		self._name = name
		self._scale_type = scale_type

		#############################################
		# TODO: write to json better
		#############################################
		if name == "German" :
			data = pd.read_csv(self._data_info["path"], header = None, delimiter = " ")
		else :
			data = pd.read_csv(self._data_info["path"], header = None, delimiter = ",")

		X = data.iloc[:, self._data_info["X"]]
		y = data.iloc[:, self._data_info["y"]].values
		if name == "German" :
			y = y - 1
		elif name == "toxicity" :
			mapping = {'Toxic' : 1, 'NonToxic' : 0}
			y = y.map(mapping)
		elif name == "pe_imports" :
			y = 1 - y


		self._discrete_columns = self._data_info["discrete_columns"]

		self._data_transformer = self.__data_transformer_setting(X)

		self._tabular_data = self.__data_processing(X,y)


	def __data_processing(self,x,y):
		R = self._data_transformer.transform(x)
		Rdis, Rcon = self._data_transformer.separate_continuous_discrete_columns(R)
		Rtogether = self._data_transformer.take_discrete_continuous_together(Rdis, Rcon)
		Rnew = self._data_transformer.separate_to_ordered_R(Rtogether)


		return TableDataStruct(X = x,
		                       y = y,
		                       nb_classes = np.max(y)+1,
		                       discrete_columns = self._discrete_columns,
		                       separate_num = self._data_transformer.separate_num,
		                       R = R,
		                       Rdis = Rdis,
		                       Rcon = Rcon,
		                       Rtogether = Rtogether,
		                       Rnew = Rnew)

	def __data_transformer_setting(self,X):
		data_transformer = DataTransformer(self._discrete_columns, need_normalized = True, scale_type = self._scale_type)
		data_transformer.fit(X)
		return data_transformer

	def Serialize(self,filepath):
		import pickle
		with open(filepath+f"/{self.__name__}_{self._name}.pkl", "wb") as file :
			pickle.dump(self, file)


	@property
	def name(self):
		return self._name
	@property
	def data_info(self):
		return self._data_info
	@property
	def scale_type(self):
		return self._scale_type
	@property
	def discrete_columns(self):
		return self._discrete_columns
	@property
	def data_transformer(self):
		return self._data_transformer
	@property
	def tabular_data(self):
		return self._tabular_data