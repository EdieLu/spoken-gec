import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = 'cpu'

class AttentionLayer(nn.Module):
	"""
		Attention layer according to https://arxiv.org/abs/1409.0473.
	"""

	def __init__(self, query_size, key_size, value_size=None, mode='bahdanau',
				 dropout=0.0, batch_first=True, bias=True,
				 query_transform=False, output_transform=False, 
				 output_nonlinearity='tanh', output_size=None,
				 hidden_size=1, use_gpu=False, hard_att=False):
		
		super(AttentionLayer, self).__init__()
		assert mode == 'bahdanau' or mode == 'dot_prod' or mode == 'hybrid' or mode == 'bilinear'

		"""
			query index: i
			key index: j

			bahdanau:
				att_ij = w * tanh(U * s_(i-1) + V * h_j + b)
				hidden_size: dim0 of U,V
			hybrid: 
				take into account both content and location
				att_ij = a * exp[-b * (c-j)^2]
				a,b: variance
				c: mean (exp(c0) - change in mean)
		"""

		# config device
		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')		

		# define var
		value_size = value_size or key_size  # Usually key and values are the same
		self.mode = mode
		self.query_size = query_size
		self.key_size = key_size
		self.value_size = value_size
		self.batch_first = batch_first
		self.mask = None
		self.hidden_size = hidden_size
		self.use_gpu = use_gpu
		self.hard_att = hard_att
		
		# define opertations
		if mode == 'bahdanau':
			self.linear_att_q = nn.Linear(self.query_size, self.hidden_size, bias=bias)
			self.linear_att_k = nn.Linear(self.key_size, self.hidden_size, bias=bias)
			self.linear_att_o = nn.Linear(self.hidden_size, 1, bias=bias)

		elif mode == 'hybrid':
			self.linear_att_aq = nn.Linear(self.query_size, self.hidden_size, bias=bias)
			self.linear_att_ak = nn.Linear(self.key_size, self.hidden_size, bias=bias) 
			self.linear_att_ao = nn.Linear(self.hidden_size, 1, bias=bias)
			self.linear_att_bq = nn.Linear(self.query_size, self.hidden_size, bias=bias)
			self.linear_att_bk = nn.Linear(self.key_size, self.hidden_size, bias=bias) 
			self.linear_att_bo = nn.Linear(self.hidden_size, 1, bias=bias)
			self.linear_att_cq = nn.Linear(self.query_size, self.hidden_size, bias=bias)
			self.linear_att_ck = nn.Linear(self.key_size, self.hidden_size, bias=bias) 
			self.linear_att_co = nn.Linear(self.hidden_size, 1, bias=bias)

		elif mode == 'bilinear':
			# ignore self.hidden_size if mode=bilinear
			self.linear_att_w = nn.Linear(self.key_size, self.query_size, bias=False)

		if output_transform:
			output_size = output_size or query_size
			self.linear_out = nn.Linear(query_size + value_size, output_size, bias=bias)
			self.output_size = output_size
		else:
			self.output_size = value_size
		if query_transform:
			self.linear_q = nn.Linear(query_size, key_size, bias=bias)
		self.dropout = nn.Dropout(dropout)
		self.output_nonlinearity = output_nonlinearity


	def set_mask(self, mask):
		"""
			applies a mask of b x t_k length
		"""

		self.mask = mask
		if mask is not None and not self.batch_first:
			self.mask = self.mask.t()


	def calc_score(self, att_query, att_keys, prev_c=None):
		"""
			att_query: 	b x t_q x n_q (inference: t_q=1)
			att_keys:  	b x t_k x n_k
			return:		b x t_q x t_k

			'dot_prod': att = q * k^T
			'bahdanau':	att = W * tanh(Uq + Vk + b)
			'loc_based': att = a * exp[ b(c-j)^2 ]
							j - key idx
							i - query idx
							prev_c - c_(i-1)
							a0,b0,c0 parameterised by q, k - (Uq_i + Vk_j + b)
							a = exp(a0), b=exp(b0)
							c = prev_c + exp(c0)

		"""

		b = att_query.size(0)
		t_q = att_query.size(1) # = 1 if in inference mode
		t_k = att_keys.size(1)
		n_q = att_query.size(2)
		n_k = att_keys.size(2)

		c_out = None # placeholder

		if self.mode == 'bahdanau':
			att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n_q)
			att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n_k)
			wq = self.linear_att_q(att_query).view(b, t_q, t_k, self.hidden_size)
			uk = self.linear_att_k(att_keys).view(b, t_q, t_k, self.hidden_size)
			sum_qk = wq + uk
			out = self.linear_att_o(F.tanh(sum_qk)).view(b, t_q, t_k)

		elif self.mode == 'hybrid':

			# start_time = time.time()

			att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n_q)
			att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n_k)

			if not hasattr(self, 'linear_att_ao'): # to word with old att setup
				self.hidden_size = 1 # fix

				a_w = self.linear_att_aq(att_query).view(b, t_q, t_k, self.hidden_size)
				a_uk = self.linear_att_ak(att_keys).view(b, t_q, t_k, self.hidden_size)
				a_sum_qk = a_wq + a_uk
				a_out = torch.exp(torch.tanh(a_sum_qk)).view(b, t_q, t_k)
				
				b_wq = self.linear_att_bq(att_query).view(b, t_q, t_k, self.hidden_size)
				b_uk = self.linear_att_bk(att_keys).view(b, t_q, t_k, self.hidden_size)
				b_sum_qk = b_wq + b_uk
				b_out = torch.exp(torch.tanh(b_sum_qk)).view(b, t_q, t_k)

				c_wq = self.linear_att_cq(att_query).view(b, t_q, t_k, self.hidden_size)
				c_uk = self.linear_att_ck(att_keys).view(b, t_q, t_k, self.hidden_size)
				c_sum_qk = c_wq + c_uk
				c_out = torch.exp(torch.tanh(c_sum_qk)).view(b, t_q, t_k)

			else: # new setup by default

				a_wq = self.linear_att_aq(att_query).view(b, t_q, t_k, self.hidden_size)
				a_uk = self.linear_att_ak(att_keys).view(b, t_q, t_k, self.hidden_size)
				a_sum_qk = a_wq + a_uk
				a_out = torch.exp(self.linear_att_ao(torch.tanh(a_sum_qk))).view(b, t_q, t_k)
								
				b_wq = self.linear_att_bq(att_query).view(b, t_q, t_k, self.hidden_size)
				b_uk = self.linear_att_bk(att_keys).view(b, t_q, t_k, self.hidden_size)
				b_sum_qk = b_wq + b_uk
				b_out = torch.exp(self.linear_att_bo(torch.tanh(b_sum_qk))).view(b, t_q, t_k)
				
				c_wq = self.linear_att_cq(att_query).view(b, t_q, t_k, self.hidden_size)
				c_uk = self.linear_att_ck(att_keys).view(b, t_q, t_k, self.hidden_size)
				c_sum_qk = c_wq + c_uk
				c_out = torch.exp(self.linear_att_co(torch.tanh(c_sum_qk))).view(b, t_q, t_k)

			# print(time.time() - start_time)

			if t_q != 1:
				# teacher forcing mode - t_q != 1
				key_indices = torch.arange(t_k).repeat(b, t_q).view(b, t_q, t_k).type(torch.FloatTensor).to(device=device)
				c_curr = torch.FloatTensor([0]).repeat(b, t_q, t_k).to(device=device)

				for i in range(t_q):
					c_temp = torch.sum(c_out[:,:i+1,:], dim=1)
					c_curr[:, i, :] = c_temp
				out = a_out * torch.exp(-b_out * torch.pow((c_curr - key_indices),2))
				# print(out[0].size())
				# print(torch.argmax(out[0], dim=1))

			else:
				# infernece mode: t_q = 1
				key_indices = torch.arange(t_k).repeat(b, 1).view(b, 1, t_k).type(torch.FloatTensor).to(device=device)

				c_out = prev_c + c_out
				out = a_out * torch.exp( -b_out * torch.pow((c_out - key_indices),2) )
				# print(out[0].size())
				# print(torch.argmax(out[0], dim=1))

		elif self.mode == 'bilinear':

			wk = self.linear_att_w(att_keys).view(b, t_k, n_q)
			out = torch.bmm(att_query, wk.transpose(1, 2))

		elif self.mode == 'dot_prod':

			assert n_q == n_k, 'Dot_prod attention - query, key size must agree!'
			out = torch.bmm(att_query, att_keys.transpose(1, 2))

		return out, c_out


	def forward(self, query, keys, values=None, prev_c=None, att_ref=None):

		"""
			query(out):	b x t_q x n_q
			keys(in): 	b x t_k x n_k (usually: n_k >= n_v - keys are richer) 
			vals(in): 	b x t_k x n_v
			context:	b x t_q x output_size	[weighted average of att keys]
			scores: 	b x t_q x t_k

			prev_c: for loc_based attention; None otherwise
			att_ref: reference (normalised) attention scores used to calculate weighted sum
			c_out: for loc_based attention; None otherwise

			in general
				n_q = embedding_dim
				n_k = size of key vectors
				n_v = size of value vectors
		"""

		# config device
		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')	
			
		if not self.batch_first:
			keys = keys.transpose(0, 1)
			if values is not None:
				values = values.transpose(0, 1)
			if query.dim() == 3:
				query = query.transpose(0, 1)
		
		if query.dim() == 2:
			single_query = True
			query = query.unsqueeze(1)
		else:
			single_query = False

		values = keys if values is None else values # b x t_k x n_v/n_k

		b = query.size(0)
		t_k = keys.size(1)
		t_q = query.size(1)

		if hasattr(self, 'linear_q'):
			att_query = self.linear_q(query)
		else:
			att_query = query

		scores, c_out = self.calc_score(att_query, keys, prev_c)  # b x t_q x t_k

		if self.mask is not None:
			mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
			scores.masked_fill_(mask, -1e12)

		# Normalize the scores OR use hard attention
		if hasattr(self, 'hard_att'):
			if self.hard_att:
				top_idx = torch.argmax(scores, dim=2)
				scores_view = scores.view(-1, t_k)
				scores_hard = (scores_view == scores_view.max(dim=1, keepdim=True)[0]).view_as(scores)
				scores_hard = scores_hard.type(torch.FloatTensor)
				total_score = torch.sum(scores_hard, dim=2)
				total_score = total_score.view(b,t_q,1).repeat(1,1,t_k).view_as(scores)
				scores_normalized = scores_hard / total_score
				if self.use_gpu and torch.cuda.is_available():
					scores_normalized = scores_normalized.cuda()
			else:
				scores_normalized = F.softmax(scores, dim=2)
		else:
			scores_normalized = F.softmax(scores, dim=2)
		# print(torch.argmax(scores_normalized[0], dim=1))

		# Context = the weighted average of the attention inputs
		if type(att_ref) == type(None):
			# if given reference attention scores
			context = torch.bmm(scores_normalized, values)  # b x t_q x n_v
			# print('att no ref')
		else:
			context = torch.bmm(att_ref, values)
			# print('att ref')

		if hasattr(self, 'linear_out'):
			context = self.linear_out(torch.cat([query, context], 2))
			if self.output_nonlinearity == 'tanh':
				context = F.tanh(context)
			elif self.output_nonlinearity == 'relu':
				context = F.relu(context, inplace=True)

		if single_query:
			context = context.squeeze(1)
			scores_normalized = scores_normalized.squeeze(1)
		elif not self.batch_first:
			context = context.transpose(0, 1)
			scores_normalized = scores_normalized.transpose(0, 1)

		return context, scores_normalized, c_out


