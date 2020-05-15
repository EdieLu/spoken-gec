from __future__ import print_function
import math
import torch
import torch.nn as nn
import numpy as np

"""
	from https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/loss/loss.py
	with minor modification by YTL
"""

class Loss(object):
	""" Base class for encapsulation of the loss functions.
	This class defines interfaces that are commonly used with loss functions
	in training and inferencing.  For information regarding individual loss
	functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions
	Note:
		Do not use this class directly, use one of the sub classes.
	Args:
		name (str): name of the loss function used by logging messages.
		criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
			to http://pytorch.org/docs/master/nn.html#loss-functions for
			a list of them.
	Attributes:
		name (str): name of the loss function used by logging messages.
		criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
			to http://pytorch.org/docs/master/nn.html#loss-functions for
			a list of them.  Implementation depends on individual
			sub-classes.
		acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
		norm_term (float): normalization term that can be used to calculate
			the loss of multiple batches.  Implementation depends on individual
			sub-classes.
	"""

	def __init__(self, name, criterion):
		self.name = name
		self.criterion = criterion
		if not issubclass(type(self.criterion), nn.modules.loss._Loss):
			raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
		# accumulated loss
		self.acc_loss = 0
		# normalization term
		self.norm_term = 0

	def reset(self):
		""" Reset the accumulated loss. """
		self.acc_loss = 0
		self.norm_term = 0

	def get_loss(self):
		""" Get the loss.
		This method defines how to calculate the averaged loss given the
		accumulated loss and the normalization term.  Override to define your
		own logic.
		Returns:
			loss (float): value of the loss.
		"""
		raise NotImplementedError

	def eval_batch(self, outputs, target):
		""" Evaluate and accumulate loss given outputs and expected results.
		This method is called after each batch with the batch outputs and
		the target (expected) results.  The loss and normalization term are
		accumulated in this method.  Override it to define your own accumulation
		method.
		Args:
			outputs (torch.Tensor): outputs of a batch.
			target (torch.Tensor): expected output of a batch.
		"""
		raise NotImplementedError

	def cuda(self):
		self.criterion.cuda()

	def backward(self, retain_graph=False):
		if type(self.acc_loss) is int:
			raise ValueError("No loss to back propagate.")
		# backword
		self.acc_loss.backward(retain_graph=retain_graph)
		# print("backward here")

	def normalise(self):
		self.acc_loss /= self.norm_term

	def mul(self, coeff):
		self.acc_loss *= coeff

	def add(self, loss2):
		self.acc_loss += loss2.acc_loss



class NLLLoss(Loss):
	""" Batch averaged negative log-likelihood loss.
	Args:
		weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
		mask (int, optional): index of masked token, i.e. weight[mask] = 0.
		size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
	"""
	"""
	YTL:
		set reduction = False to allow masking for each batch
	"""

	_NAME = "NLLLoss"

	def __init__(self, weight=None, mask=None, reduction='none'):
		self.mask = mask
		if mask is not None:
			if weight is None:
				raise ValueError("Must provide weight with a mask.")
			weight[mask] = 0

		# set loss as nn.NLLLoss
		super(NLLLoss, self).__init__(
			self._NAME,
			nn.NLLLoss(weight=weight, reduction=reduction))

	def get_loss(self):
		if isinstance(self.acc_loss, int):
			return 0
		# total loss for all batches
		loss = self.acc_loss.data.detach().item()
		return loss

	def eval_batch(self, outputs, target):
		self.acc_loss += torch.sum(self.criterion(outputs, target))
		self.norm_term += 1

	def eval_batch_with_mask(self, outputs, target, mask):

		# masked_loss = torch.mul(self.criterion(outputs, target),mask)
		masked_loss = self.criterion(outputs, target).masked_select(mask)
		self.acc_loss += masked_loss.sum()
		self.norm_term += 1


class BCELoss(Loss):

	_NAME = "BCELoss"

	def __init__(self, weight=None, mask=None, reduction='none'):
		self.mask = mask
		if mask is not None:
			if weight is None:
				raise ValueError("Must provide weight with a mask.")
			weight[mask] = 0

		# set loss as nn.BCELoss
		super(BCELoss, self).__init__(
			self._NAME,
			nn.BCELoss(weight=weight, reduction=reduction))

	def get_loss(self):
		if isinstance(self.acc_loss, int):
			return 0
		# total loss for all batches
		loss = self.acc_loss.data.detach().item()
		return loss

	def eval_batch(self, outputs, target):
		self.acc_loss += torch.sum(self.criterion(outputs, target))
		self.norm_term += 1

	def eval_batch_with_mask(self, outputs, target, mask):

		# masked_loss = torch.mul(self.criterion(outputs, target),mask)
		masked_loss = self.criterion(outputs, target).masked_select(mask)
		self.acc_loss += masked_loss.sum()
		self.norm_term += 1


class CrossEntropyLoss(Loss):

	_NAME = "CrossEntropyLoss"

	def __init__(self, weight=None, mask=None, reduction='none'):
		self.mask = mask
		if mask is not None:
			if weight is None:
				raise ValueError("Must provide weight with a mask.")
			weight[mask] = 0

		# set loss as nn.CrossEntropyLoss
		super(CrossEntropyLoss, self).__init__(
			self._NAME,
			nn.CrossEntropyLoss(weight=weight, reduction=reduction))

	def get_loss(self):
		if isinstance(self.acc_loss, int):
			return 0
		# total loss for all batches
		loss = self.acc_loss.data.detach().item()
		return loss

	def eval_batch(self, outputs, target):
		self.acc_loss += torch.sum(self.criterion(outputs, target))
		self.norm_term += 1

	def eval_batch_with_mask(self, outputs, target, mask):

		# masked_loss = torch.mul(self.criterion(outputs, target),mask)
		masked_loss = self.criterion(outputs, target).masked_select(mask)
		self.acc_loss += masked_loss.sum()
		self.norm_term += 1
