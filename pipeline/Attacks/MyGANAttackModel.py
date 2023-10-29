import numpy as np
import torch
from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from pipeline.GAN.GAN_Attack_Model import GAN_Attack_Model
from pipeline.Attacks.GreedyAttackModel import greedy_attack
from tensorflow.keras.utils import to_categorical
from pipeline.Attacks.BaseAttackModel import BaseModelAttackModel

class MyGANAttackModel(BaseModelAttackModel) :
	# TODO:
	def __init__(self, model, processor : TabularDataProcessor, gan_model : GAN_Attack_Model):
		super().__init__(model)
		self._processor = processor
		self._gan_model = gan_model
		self._name = "MyGAN"

	def Attack(self,
	           x_data : np.ndarray,
	           k = 1,
	           ) -> np.ndarray:

		if self._gan_model is None:
			raise ValueError("MyGAN Attack needs a GAN attack model.")


		return MyGANattack(x_data, target_model = self._model, gan_model = self._gan_model, processor = self._processor, K = k, device = next(self._model.parameters()).device)




# TODO: The 'greedy' algorithm needs to be decoupled.
def MyGANattack(r, target_model, gan_model : GAN_Attack_Model, processor : TabularDataProcessor, K, device):
	data_dim = r.shape[1]
	cur_r = r
	if isinstance(r, np.ndarray):
		cur_r = torch.Tensor(r).to(device)
		cur_r = cur_r.to(torch.float32)
	r_new = greedy_attack(target_model = target_model, processor = processor, K = K,
	                      x_data = cur_r, device = device)
	r_new = torch.Tensor(r_new).to(device)
	r_discrete = r_new[:, 0 :processor.tabular_data.separate_num]
	r_continuous = cur_r[:, processor.tabular_data.separate_num : data_dim]

	if processor.tabular_data.separate_num == data_dim:
		return r_discrete.cpu().detach().numpy()

	r_latent = gan_model.auto_encoder_model.transform(r_continuous)

	ans_data =  gan_model.myGANattack_only_on_latent(x_discrete = r_discrete, x_latent = r_latent)
	return ans_data.cpu().detach().numpy()
