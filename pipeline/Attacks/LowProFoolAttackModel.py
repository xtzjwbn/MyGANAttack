import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from pipeline.Attacks.BaseAttackModel import BaseModelAttackModel


class LowProFoolAttackModel(BaseModelAttackModel) :

	def __init__(self,model) :
		super().__init__(model)
		self._name = "LowProFool"

	def Attack(self,
			   x_data: np.ndarray,
			   maxiters:int=2000,
			   alpha:float=0.001,
			   lambda_:float=8.5) -> np.ndarray :
		device = next(self._model.parameters()).device
		from tqdm import tqdm
		attack_lpf = np.zeros((x_data.shape[0], x_data.shape[1]))
		with tqdm(total=x_data.shape[0]) as pbar:
			pbar.set_description('LowProFool')
			for i in range(len(x_data)) :
				cur_r = torch.from_numpy(x_data[i]).to(device).to(torch.float32)
				_, _, adv_r, _ = lowProFool(cur_r, self._model)
				attack_lpf[i, :] = adv_r
				pbar.update(1)
		return attack_lpf



def lowProFool(x, model, maxiters=2000, alpha=0.001, lambda_=8.5) :
	# from GitHub
	"""
	Generates an adversarial examples x' from an original sample x
	:param x: tabular sample
	:param model: neural network
	:param maxiters: maximum number of iterations ran to generate the adversarial examples
	:param alpha: scaling factor used to control the growth of the perturbation
	:param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
	:return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
	"""

	r = Variable(torch.FloatTensor(1e-4 * np.ones(x.shape)), requires_grad = True)
	# let v = 1
	v = torch.FloatTensor(np.ones(x.shape)).cuda()
	#v = torch.FloatTensor(np.array(weights))
	x = torch.unsqueeze(x, 0).cuda()
	r = torch.unsqueeze(r, 0).cuda()
	r.retain_grad()
	v = torch.unsqueeze(v, 0).cuda()
	output = model.forward(x + r)
	orig_pred = output.argmax().cpu().numpy()
	target_pred = np.abs(1 - orig_pred)

	target = [[0., 1.]] if target_pred == 1 else [[1., 0.]]
	target = Variable(torch.tensor(target, requires_grad = False)).cuda()

	lambda_ = torch.tensor([lambda_]).cuda()

	bce = nn.BCELoss()
	l1 = lambda v, r : torch.sum(torch.abs(v * r))  # L1 norm
	l2 = lambda v, r : torch.sqrt(torch.sum(torch.mul(v, v)))  # L2 norm

	best_norm_weighted = np.inf
	best_pert_x = x

	loop_i, loop_change_class = 0, 0
	while loop_i < maxiters :

		zero_gradients(r)

		# Computing loss
		loss_1 = bce(nn.Sigmoid()(output), target)
		loss_2 = l2(v, r).cuda()
		loss = (loss_1 + lambda_ * loss_2)

		# Get the gradient
		loss.backward(retain_graph = True)
		grad_r = r.grad.data.cpu().numpy().copy()

		# Guide perturbation to the negative of the gradient
		ri = - grad_r

		# limit huge step
		ri *= alpha

		# Adds new perturbation to total perturbation
		r = r.clone().detach().cpu().numpy() + ri

		# For later computation
		r_norm_weighted = np.sum(np.abs(r))

		# Ready to feed the model
		r = Variable(torch.FloatTensor(r), requires_grad = True)
		r = r.cuda()
		r.retain_grad()

		# Compute adversarial example
		xprime = x + r

		# Clip to stay in legitimate bounds
		xprime = torch.clip(xprime, 0, 1)

		# Classify adversarial example
		output = model.forward(xprime)
		output_pred = output.argmax().cpu().numpy()

		# Keep the best adverse at each iterations
		if output_pred != orig_pred and r_norm_weighted < best_norm_weighted :
			best_norm_weighted = r_norm_weighted
			best_pert_x = xprime

		if output_pred == orig_pred :
			loop_change_class += 1

		loop_i += 1

	# Clip at the end no matter what
	best_pert_x = torch.clip(best_pert_x, 0, 1)
	output = model.forward(best_pert_x)
	output_pred = output.argmax().cpu().numpy()

	return orig_pred, output_pred, best_pert_x.clone().detach().cpu().numpy(), loop_change_class