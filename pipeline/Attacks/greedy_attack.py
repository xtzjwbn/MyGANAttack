import numpy as np
import torch

"""
positions = 1 - to_categorical(range(dataDim), num_classes=dataDim)
for i in range(discrete_gang,len(positions)):
    positions[i,i] = 1
"""


def greedy_attack(targetModel, K, separate_number, x_data, positions, device, data_transformer):

	return_list = []
	A = np.expand_dims(x_data.to(torch.float16).cpu(), 1) * positions.astype(np.float16)
	curClass = (targetModel(x_data.to(device))).argmax(dim=1)

	for sample_i in range(A.shape[0]):
		pred = targetModel(torch.Tensor(A[sample_i]).cuda())
		score = pred[:, curClass[sample_i].item()].cpu().detach().numpy()

		start_list = []
		end_list = []
		ans_list = []

		for k in range(K):
			cur_position = np.argsort(score)[k]
			if cur_position >= separate_number:
				continue
			start, end = data_transformer.locating_the_dis_column(cur_position)
			cur_expand = np.expand_dims(x_data[sample_i].cpu().detach().numpy(), axis = 0)
			changed_sequences = np.tile(cur_expand, (end - start,1))
			changed_sequences[:, start:end] = np.diag(np.ones(end-start))
			changed_sequences = torch.Tensor(changed_sequences).to(device)
			cur_sample_pred = targetModel(changed_sequences)
			# print(cur_sample_pred)
			k_score = cur_sample_pred[:, curClass[sample_i].item()].cpu().detach().numpy()
			k_ans = np.argsort(k_score)[0]
			start_list.append(start)
			end_list.append(end)
			ans_list.append(k_ans)

		advX = x_data[sample_i].clone()
		advX = advX.cpu().detach().numpy()
		for ix in range(len(start_list)):
			cur_used = np.zeros(end_list[ix]-start_list[ix])
			cur_used[ans_list[ix]] = 1
			advX[start_list[ix]:end_list[ix]] = cur_used

		return_list.append(torch.Tensor(advX).unsqueeze(0).to(device))
		if (sample_i+1) % 1000 == 0:
			print("sample {} done".format(sample_i+1))

	return torch.cat(return_list, dim=0)