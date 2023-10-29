import numpy as np
import torch
from pipeline.Preparing.tabular_data_processor import TabularDataProcessor
from tensorflow.keras.utils import to_categorical

from pipeline.Attacks.BaseAttackModel import BaseModelAttackModel


class GreedyAttackModel(BaseModelAttackModel) :
	def __init__(self, model, processor) :
		super().__init__(model)
		self._processor = processor
		self._name = "Greedy"

	def Attack(self,
	           x_data : np.ndarray,
	           k : int = 1) -> np.ndarray:

		if self._processor is  None:
			raise ValueError("Greedy Attack needs a TabularDataProcessor.")

		return greedy_attack(target_model = self._model, processor = self._processor, K = k, x_data = x_data, device = next(self._model.parameters()).device)




# TODO: Enhance progress bar display for better visibility.
def greedy_attack(target_model : torch.nn.Module,
                  processor : TabularDataProcessor,
                  K : int,
                  x_data : np.ndarray,
                  device = torch.device("cpu")):

	"""
	:param target_model:
	:param processor:
	:param K: how many columns need to change.
	:param x_data: raw samples
	:param device:
	:return: !!! np.ndarray
	"""

	positions = 1 - to_categorical(range(processor.tabular_data.Rtogether.shape[1]),
	                               num_classes = processor.tabular_data.Rtogether.shape[1])
	for i in range(processor.tabular_data.separate_num, len(positions)) :
		positions[i, i] = 1

	x_processed = x_data
	if not isinstance(x_processed,torch.Tensor):
		x_processed = torch.Tensor(x_processed)
	return_list = []
	current_matrix = np.expand_dims(x_processed.to(torch.float16).cpu(), 1) * positions.astype(np.float16)
	cur_class = (target_model(x_processed.to(device))).argmax(dim=1)

	for sample_i in range(current_matrix.shape[0]):
		pred = target_model(torch.Tensor(current_matrix[sample_i]).cuda())
		score = pred[:, cur_class[sample_i].item()].cpu().detach().numpy()

		start_list = []
		end_list = []
		ans_list = []

		for k in range(K):
			cur_position = np.argsort(score)[k]
			if cur_position >= processor.tabular_data.separate_num:
				continue
			start, end = processor.data_transformer.locating_the_dis_column(cur_position)
			cur_expand = np.expand_dims(x_processed[sample_i].cpu().detach().numpy(), axis = 0)
			changed_sequences = np.tile(cur_expand, (end - start,1))
			changed_sequences[:, start:end] = np.diag(np.ones(end-start))
			changed_sequences = torch.Tensor(changed_sequences).to(device)
			cur_sample_pred = target_model(changed_sequences)
			# print(cur_sample_pred)
			k_score = cur_sample_pred[:, cur_class[sample_i].item()].cpu().detach().numpy()
			k_ans = np.argsort(k_score)[0]
			start_list.append(start)
			end_list.append(end)
			ans_list.append(k_ans)

		adv_x = x_processed[sample_i].clone()
		adv_x = adv_x.cpu().detach().numpy()
		for ix in range(len(start_list)):
			cur_used = np.zeros(end_list[ix]-start_list[ix])
			cur_used[ans_list[ix]] = 1
			adv_x[start_list[ix]:end_list[ix]] = cur_used

		return_list.append(torch.Tensor(adv_x).unsqueeze(0).to(device))
		if (sample_i+1) % 1000 == 0:
			print("sample {} done".format(sample_i+1))

	return torch.cat(return_list, dim=0).cpu().detach().numpy()