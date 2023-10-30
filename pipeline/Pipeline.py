import numpy as np
import torch
import copy
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from pipeline.Classifier.Classifier import Classifier
from pipeline.Encoder.AutoEncoderModel import AutoEncoderModel
from pipeline.GAN.GAN_Attack_Model import GAN_Attack_Model
from pipeline.Attacks.GreedyAttackModel import greedy_attack
from pipeline.Comparison.WhiteboxAttackComparision import Comparison

from Utils.Checking import check_type, check_none

class Pipeline:
	"""
	The PIPELINE:
	TabularDataProcessor
		-> Classifier Setting & Fitting -> Encoder Setting & Fitting -> GAN Setting & Fitting            (MyGAN Attack Model Fitting)
		-> (Serialization)
		-> Comparison                                                                                    (Comparison with Other Attack Algorithms)
		-> Defense -> Defense Comparison                                                                 (Adv Defense)
	"""
	def __init__(self,
				 json : str = "../Data/German.json",
				 device = torch.device("cpu")):
		self._processor = TabularDataProcessor(json)
		self._name = self._processor.name

		self._classifier = None
		self._encoder = None
		self._gan = None
		self._comparison = None

		# defense
		self._new_classifier = None
		self._classifier_batch_size = None
		self._classifier_epochs = None
		self._new_classifier_map = {}

		# hyperparameter
		self._encoder_dim = None
		self._K = None
		self._eps = None
		self._alpha_adv = None
		self._alpha_norm = None

		# dataset
		self._train_x = None
		self._train_y = None
		self._test_x = None
		self._test_y = None

		self._device = device

	def ClassifierFit(self, batch_size : int = 100, epochs : int = 400):
		check_type(self._classifier, Classifier, "Classifier")
		check_type(self._train_x, np.ndarray, "Classifier Fitting Phrase, train_x")
		check_type(self._train_y, np.ndarray, "Classifier Fitting Phrase, train_y")
		self._classifier_batch_size = batch_size
		self._classifier_epochs = epochs

		weighted = self._processor.data_info["weighted"]
		print("#####################\033[0;31mClassifier Fitting Phrase\033[0m #####################")
		self._classifier.fit(x = self._train_x,
							 y = self._train_y,
							 batch_size = batch_size,
							 epochs = epochs,
							 weighted = weighted)
		print("####################################################################")

	def EncoderFit(self,batch_size : int = 100, epochs : int = 2000,):
		check_type(self._encoder, AutoEncoderModel, "Encoder")
		check_type(self._train_x, np.ndarray,"Encoder Fitting Phrase, train_x")

		print("#####################\033[0;31mEncoder Fitting Phrase\033[0m ########################")
		self._encoder.fit(x = self._train_x[:, self._processor.tabular_data.separate_num:], batch_size = batch_size, epochs = epochs)
		print("####################################################################")

	def GANFit(self,
			   K : int = 1,
			   eps : float = 1,
			   alpha_adv : float = 10,
			   alpha_norm : float = 1,
			   batch_size : int = 100,
			   epochs : int = 400
			   ):

		check_type(self._gan, GAN_Attack_Model, "GAN Model")
		check_type(self._train_x, np.ndarray, "GAN Model Fitting Phrase, train_x")
		check_type(self._train_y, np.ndarray, "GAN Model Fitting Phrase, train_y")

		self._K = K
		self._eps = eps
		self._alpha_norm = alpha_norm
		self._alpha_adv = alpha_adv

		print("#####################\033[0;31mGreedy Fitting Phrase\033[0m #########################")
		# TODO: Hope a better interface, not just greedy_attack
		A = greedy_attack(target_model = self._classifier.model,
						  processor = self._processor,
						  K = K,
						  x_data = self._train_x,
						  device = self._device)

		print("#####################\033[0;31mGAN Fitting Phrase\033[0m ############################")
		self._gan.fit(X = self._train_x,
					  R = A,
					  y = self._train_y,
					  separate_num = self._processor.tabular_data.separate_num,
					  eps = eps,
					  alpha_adv = alpha_adv,
					  alpha_norm = alpha_norm,
					  batch_size = batch_size,
					  epochs = epochs
					  )
		print("####################################################################")

	def GetComparison(self):
		if self._comparison is None:
			check_none(self._classifier, "Comparison Phrase, Classifier")
			check_none(self._processor, "Comparison Phrase, TabularDataProcessor")
			check_none(self._gan, "Comparison Phrase, GAN")
			self._comparison = Comparison(model = self._classifier.model, processor = self._processor, gan_model = self._gan)
		return self._comparison

	def AddAttackToComparison(self, name : str, **kwargs):
		check_type(self._comparison, Comparison, "Comparison")
		self._comparison.AddAttackModel(name, **kwargs)

	def ShowComparison(self):
		self._comparison.SetData(self._test_x, self._test_y).Attacking_All().ShowComparison()

	# TODO: More Defense Methods
	def DataAugmentationFit(self):
		self._comparison.SetData(self._train_x, self._train_y).Attacking_All()
		raw_adv_map = self._comparison.adv_map
		processed_adv_map =self._comparison.adv_map_processed
		weighted = self._processor.data_info["weighted"]

		for key in processed_adv_map:
			print(f"-----------Data Augmentation with (\033[0;31m{key}\033[0m) processed attack samples FITTING START!-----------")
			new_train_x = np.vstack((self._train_x, processed_adv_map[key]))
			new_train_y = np.hstack((self._train_y, self._train_y))
			self._new_classifier_map[key] = copy.deepcopy(self._new_classifier)
			self._new_classifier_map[key].fit(x = new_train_x,
							 y = new_train_y,
							 batch_size = self._classifier_batch_size,
							 epochs = self._classifier_epochs,
							 weighted = weighted)
			print("---------------------------------------------------------------------------------------------------------")

	def DefenseComparison(self):
		# check_none(self._classifier, "Defense Comparison Phrase, Classifier")
		check_none(self._processor, "Defense Comparison Phrase, TabularDataProcessor")
		check_none(self._gan, "Defense Comparison Phrase, GAN")
		for key in self._new_classifier_map:
			print(f"-----------DefenseComparison with Data Augmentation with (\033[0;31m{key}\033[0m) processed attack samples!-----------")
			comparison =  copy.deepcopy(self._comparison)
			comparison.target_model = self._new_classifier_map[key].model
			comparison.SetData(self._test_x, self._test_y).Attacking_All().ShowComparison()
			print("-----------------------------------------------------------------------------------------------------------------")

		return self._comparison




	def SetNewDataSet(self,
				   train_x : np.ndarray,
				   train_y : np.ndarray,
				   test_x : np.ndarray,
				   test_y : np.ndarray):
		self._train_x = train_x
		self._train_y = train_y
		self._test_x = test_x
		self._test_y = test_y

	def SplitTrainAndTest(self , test_size = 0.1 , random_seed = 222):
		self._train_x,self._test_x , self._train_y, self._test_y  = train_test_split(self._processor.tabular_data.Rtogether, self._processor.tabular_data.y,
																					 test_size = test_size, random_state = random_seed)

	@property
	def classifier(self):
		return self._classifier
	@classifier.setter
	def classifier(self, classifier_ : Classifier):
		self._classifier = classifier_

	# TODO: Need a better way to set 3 models
	def SetClassifier(self, model_type, learning_rate, loss, step_size = 100, gamma = 0.5) :
		model = model_type(self._processor.tabular_data.Rtogether.shape[1], self._processor.tabular_data.nb_classes).to(self._device)
		optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
		schedule = StepLR(optimizer, step_size=step_size, gamma=gamma)
		self._classifier = Classifier(
			name=self._processor.name,
			model=model,
			optimizer=optimizer,
			loss=loss,
			schedule=schedule,
			device=self._device
			)
		self._new_classifier = copy.deepcopy(self._classifier)


	@property
	def encoder(self):
		return self._encoder
	@encoder.setter
	def encoder(self, encoder_ : AutoEncoderModel):
		self._encoder = encoder_
	def SetEncoder(self, model_type, encoder_dim, learning_rate, loss, step_size = 400, gamma = 0.5):
		self._encoder_dim = encoder_dim

		model = model_type(input_dim=self._processor.tabular_data.Rcon.shape[1], code_dim=encoder_dim).to(self._device)
		optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
		schedule = StepLR(optimizer, step_size=step_size, gamma=gamma)
		self._encoder = AutoEncoderModel(name = self._processor.name,
										model = model,
										optimizer = optimizer,
										loss = loss,
										schedule = schedule,
										device = self._device,
										)

	@property
	def gan(self):
		return self._gan
	@gan.setter
	def gan(self, gan_ : GAN_Attack_Model):
		self._gan = gan_

	def SetGAN(self, generator_model_type, generator_lr, discriminator_model_type, discriminator_lr):
		check_none(self._encoder_dim, "Encoder-Dim")
		generator_model = generator_model_type(input_dim=self._encoder_dim+self._processor.tabular_data.separate_num, output_dim=self._encoder_dim).to(self._device)
		generator_optimizer = torch.optim.RMSprop(generator_model.parameters(), lr=generator_lr)
		discriminator_model = discriminator_model_type(self._processor.tabular_data.Rtogether.shape[1]).to(self._device)
		discriminator_optimizer = torch.optim.RMSprop(discriminator_model.parameters(), lr=discriminator_lr)
		self._gan = GAN_Attack_Model(
			name =  self._processor.name,
			target_model = self._classifier.model,
			auto_encoder_model = self._encoder.model,
			generator_model = generator_model,
			generator_optimizer = generator_optimizer,
			discriminator_model = discriminator_model,
			discriminator_optimizer = discriminator_optimizer,
			device = self._device)



	def Serialize(self,filepath):
		import pickle
		with open(filepath+f"/{self.__class__.__name__}_{self._name}.pkl", "wb") as file :
			pickle.dump(self, file)

	def SaveModels(self, filepath):
		self._classifier.save_model(filepath)
		self._encoder.save_model(filepath)
		self._gan.save_model(filepath)

	@property
	def processor(self):
		return self._processor
	@property
	def device(self):
		return self._device
	@device.setter
	def device(self, device_):
		self._device = device_

	@property
	def name(self):
		return self._name