from pipeline.data_transformer import DataTransformer
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
	discrete_columns: List[str]
	R : np.ndarray
	Rdis : np.ndarray
	Rcon : np.ndarray
	Rtogether : np.ndarray
	Rnew : np.ndarray


class TabularDataProcessor:
	def __init__(self,filepath,scale_type = "minmax"):
		self.data_info = read_data_from_json(filepath)

		name = self.data_info["name"]
		self.__name__ = name + "_Processor"
		self.name = name
		self.scale_type = scale_type

		#############################################
		# TODO: write to json better
		#############################################
		if name == "German" :
			data = pd.read_csv(self.data_info["path"], header = None, delimiter = " ")
		else :
			data = pd.read_csv(self.data_info["path"], header = None, delimiter = ",")

		X = data.iloc[:, self.data_info["X"]]
		y = data.iloc[:, self.data_info["y"]]
		if name == "German" :
			y = y - 1
		elif name == "toxicity" :
			mapping = {'Toxic' : 1, 'NonToxic' : 0}
			y = y.map(mapping)
		elif name == "pe_imports" :
			y = 1 - y


		self.discrete_columns = self.data_info["discrete_columns"]

		self.data_transformer = self.__data_transformer_setting(X)

		self.tabular_data = self.__data_processing(X,y)


	def __data_processing(self,x,y):
		R = self.data_transformer.transform(x)
		Rdis, Rcon = self.data_transformer.separate_continuous_discrete_columns(R)
		Rtogether = self.data_transformer.take_discrete_continuous_together(Rdis, Rcon)
		Rnew = self.data_transformer.separate_to_ordered_R(Rtogether)


		return TableDataStruct(X = x,
											y = y,
											discrete_columns = self.discrete_columns,
											R = R,
											Rdis = Rdis,
											Rcon = Rcon,
											Rtogether = Rtogether,
											Rnew = Rnew)

	def __data_transformer_setting(self,X):
		data_transformer = DataTransformer(self.discrete_columns, need_normalized = True, scale_type = self.scale_type)
		data_transformer.fit(X)
		return data_transformer
