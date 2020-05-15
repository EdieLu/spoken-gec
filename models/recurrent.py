import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.attention import AttentionLayer
from utils.config import PAD, EOS, BOS, UNK
from utils.dataset import load_pretrained_embedding

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')

KEY_ATTN_SCORE = 'attention_score'
KEY_CONTEXT = 'context'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'
KEY_MODEL_STRUCT = 'model_struct'
KEY_DIS = 'discriminator_output'
CLASSIFY_PROB = 'classify_prob'

class Seq2Seq(nn.Module):

	""" enc-dec model """

	def __init__(self,
		# add params
		vocab_size_enc,
		vocab_size_dec,
		embedding_size_enc=200,
		embedding_size_dec=200,
		embedding_dropout=0,
		hidden_size_dec=200,
		hidden_size_enc=200,
		num_bilstm_enc=2,
		num_unilstm_enc=0,
		dd_num_unilstm_dec=4,
		dd_hidden_size_att=10,
		dd_att_mode='hybrid',
		dd_additional_key_size=0,
		gec_num_bilstm_dec=0,
		gec_num_unilstm_dec_preatt=0,
		gec_num_unilstm_dec_pstatt=3,
		gec_hidden_size_att=10,
		gec_att_mode='bahdanau',
		shared_embed='state',
		dropout=0.0,
		residual=True,
		batch_first=True,
		max_seq_len=32,
		batch_size=200,
		load_embedding_src=None,
		load_embedding_tgt=None,
		src_word2id=None,
		tgt_word2id=None,
		src_id2word=None,
		hard_att=False,
		add_discriminator=False,
		dloss_coeff=1.0,
		use_gpu=False,
		ptr_net='null',
		dd_classifier=False,
		connect_type='embed',
		):

		super(Seq2Seq, self).__init__()

		# config device
		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		# define var
		self.hidden_size_enc = hidden_size_enc
		self.num_bilstm_enc = num_bilstm_enc
		self.num_unilstm_enc= num_unilstm_enc
		self.hidden_size_dec = hidden_size_dec
		self.state_size = self.hidden_size_dec

		self.dd_num_unilstm_dec = dd_num_unilstm_dec
		self.dd_hidden_size_att = dd_hidden_size_att
		self.gec_num_bilstm_dec = gec_num_bilstm_dec
		self.gec_num_unilstm_dec_preatt = gec_num_unilstm_dec_preatt
		self.gec_num_unilstm_dec_pstatt = gec_num_unilstm_dec_pstatt
		self.gec_hidden_size_att = gec_hidden_size_att

		self.residual = residual
		self.batch_size = batch_size
		self.max_seq_len = max_seq_len
		self.use_gpu = use_gpu
		self.hard_att = hard_att
		self.dd_additional_key_size = dd_additional_key_size
		self.shared_embed = shared_embed
		self.add_discriminator = add_discriminator
		self.dloss_coeff = dloss_coeff
		self.ptr_net = ptr_net
		self.dd_classifier = dd_classifier
		self.connect_type = connect_type

		# use shared embedding + vocab
		self.vocab_size = vocab_size_enc
		self.embedding_size = embedding_size_enc
		self.load_embedding = load_embedding_src
		self.word2id = src_word2id
		self.id2word = src_id2word

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)
		self.beam_width = 1

		# load embeddings
		if self.load_embedding:
			embedding_matrix = np.random.rand(self.vocab_size, self.embedding_size)
			embedding_matrix = load_pretrained_embedding(
				self.word2id, embedding_matrix, self.load_embedding)
			embedding_matrix = torch.FloatTensor(embedding_matrix).to(device=device)
			self.embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.embedder = nn.Embedding(self.vocab_size, self.embedding_size,
										sparse=False, padding_idx=PAD)
		self.embedder_enc = self.embedder
		self.embedder_dec = self.embedder

		# ============================================================ [DD]
		# ----------- define enc -------------
		# embedding_size [200] -> hidden_size_enc * 2 [400]
		# enc-bilstm
		self.enc = torch.nn.LSTM(self.embedding_size, self.hidden_size_enc,
						num_layers=self.num_bilstm_enc, batch_first=batch_first,
						bias=True, dropout=dropout,
						bidirectional=True)

		# enc-unilstm
		if self.num_unilstm_enc != 0:
			if not self.residual:
				self.enc_uni = torch.nn.LSTM(
					self.hidden_size_enc * 2, self.hidden_size_enc * 2,
					num_layers=self.num_unilstm_enc, batch_first=batch_first,
					bias=True, dropout=dropout,
					bidirectional=False)
			else:
				self.enc_uni = nn.Module()
				for i in range(self.num_unilstm_enc):
					self.enc_uni.add_module(
						'l'+str(i),
						torch.nn.LSTM(
							self.hidden_size_enc * 2, self.hidden_size_enc * 2,
							num_layers=1, batch_first=batch_first, bias=True,
							dropout=dropout,bidirectional=False))


		# ------------- define dd_classifier | decoder  -------------
		if self.dd_classifier == True:
			self.dd_classify = nn.Sequential(
				nn.Linear(self.hidden_size_enc * 2, 50, bias=True),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(50, 50),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(50, 1),
				nn.Sigmoid(),
			)

		else:
			# ----------- define dd-dec -------------
			# embedding_size_dec + self.state_size [200+200] -> hidden_size_dec [200]
			# dropout_dd_dec = dropout * 2

			dropout_dd_dec = dropout
			if not self.residual:
				self.dd_dec = torch.nn.LSTM(
						self.embedding_size + self.state_size,
						self.hidden_size_dec,
						num_layers=self.dd_num_unilstm_dec, batch_first=batch_first,
						bias=True, dropout=dropout_dd_dec,
						bidirectional=False)
			else:
				dd_lstm_uni_dec_first = torch.nn.LSTM(
						self.embedding_size + self.state_size, self.hidden_size_dec,
						num_layers=1, batch_first=batch_first,
						bias=True, dropout=dropout_dd_dec,
						bidirectional=False)
				self.dd_dec = nn.Module()
				self.dd_dec.add_module('l0', dd_lstm_uni_dec_first)
				for i in range(1, self.dd_num_unilstm_dec):
					self.dd_dec.add_module(
						'l'+str(i),
						torch.nn.LSTM(self.hidden_size_dec, self.hidden_size_dec,
							num_layers=1, batch_first=batch_first, bias=True,
							dropout=dropout_dd_dec, bidirectional=False))

		# ------------ define dd-att ------------------
		# dd_query: hidden_size_dec [200]
		# dd_keys: hidden_size_enc * 2 [400] + (optional) self.additional_key_size
		# dd_values: hidden_size_enc * 2 [400]
		# dd_context: weighted sum of values [400]

		# dropout_dd_att = dropout * 2
		if not self.dd_classifier:
			dropout_dd_att = dropout
			self.dd_key_size = self.hidden_size_enc * 2 + self.dd_additional_key_size
			self.dd_value_size = self.hidden_size_enc * 2
			self.dd_query_size = self.hidden_size_dec
			self.dd_att = AttentionLayer(self.dd_query_size, self.dd_key_size,
				value_size=self.dd_value_size,
				mode=dd_att_mode, dropout=dropout_dd_att,
				query_transform=False, output_transform=False,
				hidden_size=self.dd_hidden_size_att,
				use_gpu=self.use_gpu, hard_att=self.hard_att)

		# ------------- define dd output -------------------
		# (hidden_size_enc * 2 + hidden_size_dec) [600]
		# -> self.state_size [200] -> vocab_size_dec [36858]
		if not self.dd_classifier:
			self.dd_ffn = nn.Linear(self.hidden_size_enc * 2 + self.hidden_size_dec,
				self.state_size, bias=False)
			self.dd_out = nn.Linear(self.state_size , self.vocab_size, bias=True)

		# ------------- define pointer weight -------------
		if self.ptr_net == 'comb':
			self.ptr_i = nn.Linear(self.embedding_size , 1, bias=False) #decoder input
			self.ptr_s = nn.Linear(self.hidden_size_dec , 1, bias=False) #decoder state
			self.ptr_c = nn.Linear(self.hidden_size_enc * 2 , 1, bias=True)	#context

		# ------------- define discriminator -------------
		if self.add_discriminator:
			self.discriminator = nn.Sequential(
				nn.Linear(self.gec_input_size, 128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(128, 128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(128, 1),
				nn.Sigmoid(),
			)

		# ============================================================[GEC]
		# ---------- define gec-dec:preatt --------------
		# note this is actually part of enc in traditional nmt;
		# but here included in gec dec for model clarity;
		# self.state_size [200] -> self.state_size [200]

		# bilstm
		if self.dd_classifier:
			if 'word' in self.connect_type:
				self.gec_input_size = self.embedding_size
			elif self.connect_type == 'embed':
				self.gec_input_size = self.hidden_size_enc * 2 # using sharp context
			elif self.connect_type == 'prob':
				self.gec_input_size = self.embedding_size
				# may due to change
		else:
			if 'word' in self.connect_type:
				self.gec_input_size = self.embedding_size
			elif self.connect_type == 'embed':
				if self.shared_embed == 'state':
					self.gec_input_size = self.state_size
				elif self.shared_embed == 'context':
					self.gec_input_size = self.hidden_size_enc * 2
				elif self.shared_embed == 'state_tgt':
					self.gec_input_size = self.state_size + self.embedding_size
				else:
					assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)


		if self.gec_num_bilstm_dec != 0:
			self.gec_dec_bilstm = torch.nn.LSTM(
				self.gec_input_size, self.hidden_size_enc,
				num_layers=self.gec_num_bilstm_dec, batch_first=batch_first,
				bias=True, dropout=dropout,
				bidirectional=True)
			self.gec_input_size = 2 * self.hidden_size_enc

		# unilstm
		if self.gec_num_unilstm_dec_preatt != 0:
			if not self.residual:
				self.gec_dec_preatt = torch.nn.LSTM(
					self.gec_input_size, self.gec_input_size,
					num_layers=self.gec_num_unilstm_dec_preatt,
					batch_first=batch_first,
					bias=True, dropout=dropout,
					bidirectional=False)
			else:
				gec_lstm_uni_dec_preatt_first = torch.nn.LSTM(
					self.gec_input_size, self.gec_input_size,
					num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout,
					bidirectional=False)
				self.gec_dec_preatt = nn.Module()
				self.gec_dec_preatt.add_module('l0', gec_lstm_uni_dec_preatt_first)
				for i in range(1, self.gec_num_unilstm_dec_preatt):
					self.gec_dec_preatt.add_module(
						'l'+str(i),
						torch.nn.LSTM(self.gec_input_size, self.gec_input_size,
							num_layers=1, batch_first=batch_first, bias=True,
							dropout=dropout, bidirectional=False))

		if self.gec_num_bilstm_dec > 0 or self.gec_num_unilstm_dec_preatt > 0:

			# ----------- define gec-dec:pstatt ---------------
			# embedding_size_dec  + self.state_size [200+200] -> hidden_size_dec [200]
			if not self.residual:
				self.gec_dec_pstatt = torch.nn.LSTM(
					self.embedding_size + self.state_size, self.hidden_size_dec,
					num_layers=self.gec_num_unilstm_dec_pstatt, batch_first=batch_first,
					bias=True, dropout=dropout,
					bidirectional=False)
			else:
				gec_lstm_uni_dec_first_pstatt = torch.nn.LSTM(
					self.embedding_size + self.state_size, self.hidden_size_dec,
					num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout,
					bidirectional=False)
				self.gec_dec_pstatt = nn.Module()
				self.gec_dec_pstatt.add_module('l0', gec_lstm_uni_dec_first_pstatt)
				for i in range(1, self.gec_num_unilstm_dec_pstatt):
					self.gec_dec_pstatt.add_module(
						'l'+str(i),
						torch.nn.LSTM(self.hidden_size_dec, self.hidden_size_dec,
							num_layers=1, batch_first=batch_first, bias=True,
							dropout=dropout, bidirectional=False))

			# ------------ define gec-att ------------------
			# gec_query: hidden_size_dec [200]
			# gec_keys: self.state_size [200]/[400]
			# gec_values: self.state_size [200]/[400]
			# gec_context: weighted sum of values [400]
			self.gec_key_size = self.gec_input_size
			self.gec_value_size = self.gec_input_size
			self.gec_query_size = self.hidden_size_dec
			self.gec_att = AttentionLayer(self.gec_query_size, self.gec_key_size,
				value_size=self.gec_value_size,
				mode=gec_att_mode, dropout=dropout,
				query_transform=False, output_transform=False,
				hidden_size=self.gec_hidden_size_att,
				use_gpu=self.use_gpu, hard_att=self.hard_att)

			# ------------- define gec output -------------------
			# (self.gec_value_size + hidden_size_dec) [400/600]
			# -> self.state_size [200] -> vocab_size_dec [36858]
			self.gec_ffn = nn.Linear(self.gec_value_size + self.hidden_size_dec,
				self.state_size, bias=False)
			self.gec_out = nn.Linear(self.state_size, self.vocab_size, bias=True)


	# ================================= [functions used for resuming models]

	def reset_use_gpu(self, use_gpu):

		self.use_gpu = use_gpu


	def reset_max_seq_len(self, max_seq_len):

		self.max_seq_len = max_seq_len


	def reset_batch_size(self, batch_size):

		self.batch_size = batch_size


	def set_beam_width(self, beam_width):

		self.beam_width = beam_width


	def check_classvar(self, var_name):

		"""
			fix capatiblility in later versions:
				if var is define - no change
				if not defined - set to default
		"""

		if not hasattr(self, var_name):
			if var_name == 'additional_key_size':
				var_val = 0
			elif var_name == 'shared_embed':
				var_val = 'state'
			elif var_name == 'gec_num_bilstm_dec':
				var_val = 0
			elif var_name == 'num_unilstm_enc':
				var_val == 0
			elif var_name == 'residual':
				var_val == False
			elif var_name == 'add_discriminator':
				var_val = False
			elif var_name == 'dd_classifier':
				var_val	= False
			elif var_name == 'ptr_net':
				var_val = 'null'
			elif var_name == 'connect_type':
				var_val = 'embed'
			else:
				var_val = None

			# set class attribute to default value
			setattr(self, var_name, var_val)


	# =====================================================================================

	def forward(self, dd_src, gec_src, dd_tgt=None, gec_tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=1.0,
		dd_att_key_feats=None, gec_dd_att_key_feats=None, beam_width=0,
		debug_flag=False):

		"""
			Args:
				src:
					list of src word_ids [batch_size, max_seq_len, word_ids]
					src_dd: disfluent / fluent corpus [swbd]
					src_gec: learner / native corpus [clc]
				tgt:
					list of tgt word_ids
				hidden:
					initial hidden state
					not used (all hidden initialied using None)
				is_training:
					whether in eval or train mode
				teacher_forcing_ratio:
					default at 1 - always teacher forcing

			Returns:
				decoder_outputs:
					list of step_output
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				shared_hidden:
				ret_dict:
		"""

		# decoder
		# ===================================================================
		def decode(step, step_output, step_attn,
			ret_dict, decoder_outputs, sequence_symbols, lengths):

			"""
				Greedy decoding - make it callable from multiple instances
				Note:
					it should generate EOS, PAD as used in training tgt
				Args:
					step: step idx
					step_output: log predicted_softmax [batch_size, vocab_size_dec]
					step_attn: attention scores -
						(batch_size x tgt_len(=query_len) x src_len(=key_len)
					ret_dict: return dictionary of hyp info
					decoder_outputs:
						list of log predicted_softmax - T x [batch_size, vocab_size_dec]
					sequence_symbols: list of symbols - T x [batch_size]
					lengths: list of length of each sentence
				Returns:
					symbols: most probable symbol_id [batch_size, 1]
			"""

			ret_dict[KEY_ATTN_SCORE].append(step_attn)
			decoder_outputs.append(step_output)
			symbols = decoder_outputs[-1].topk(1)[1]
			assert sum(symbols.ge(self.vocab_size).long()) == 0, 'out of range symbol {}'\
				.format(torch.masked_select(symbols, symbols.ge(self.vocab_size)))

			# import pdb; pdb.set_trace()
			eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD))
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols) + 1
				pad_idx = (lengths < (len(sequence_symbols) + 1))
				symbols_dummy = symbols.reshape(-1)
				symbols_dummy[pad_idx] = PAD
				symbols = symbols_dummy.view(-1,1).to(device)

			sequence_symbols.append(symbols)

			return symbols, ret_dict, decoder_outputs, sequence_symbols, lengths

		# =====================================================================

		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		if hasattr(self, 'dd_att'):
			self.dd_att.use_gpu = self.use_gpu
		self.gec_att.use_gpu = self.use_gpu


		# init global var
		# *********************************************
		batch_size = self.batch_size
		self.beam_width = beam_width

		# ***********************************************
		# *********************************************[DD]

		if debug_flag:
			import pdb; pdb.set_trace()

		# 0. init var
		dd_ret_dict = dict()
		dd_ret_dict[KEY_DIS] = []
		dd_ret_dict[KEY_ATTN_SCORE] = []
		dd_decoder_outputs = []
		dd_sequence_symbols = []
		dd_lengths = np.array([self.max_seq_len] * batch_size)

		# 1. convert id to embedding
		dd_emb_src = self.embedding_dropout(self.embedder_enc(dd_src))
		if type(dd_tgt) == type(None):
			dd_tgt = torch.Tensor([BOS]).repeat(
				dd_src.size()).type(torch.LongTensor).to(device=device)
		dd_emb_tgt = self.embedding_dropout(self.embedder_dec(dd_tgt))
		dd_mask_src = dd_src.data.eq(PAD)

		# ******************************************************
		# 2. run enc
		dd_enc_hidden_init = None
		dd_enc_outputs, dd_enc_hidden = self.enc(dd_emb_src, dd_enc_hidden_init)
		dd_enc_outputs = self.dropout(dd_enc_outputs)\
			.view(self.batch_size, self.max_seq_len, dd_enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				dd_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				dd_enc_outputs, dd_enc_hidden_uni = self.enc_uni(
					dd_enc_outputs, dd_enc_hidden_uni_init)
				dd_enc_outputs = self.dropout(dd_enc_outputs)\
					.view(self.batch_size, self.max_seq_len, dd_enc_outputs.size(-1))
			else:
				dd_enc_hidden_uni_init = None
				dd_enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					dd_enc_inputs = dd_enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					dd_enc_outputs, dd_enc_hidden_uni = enc_func(
						dd_enc_inputs, dd_enc_hidden_uni_init)
					dd_enc_hidden_uni_lis.append(dd_enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						dd_enc_outputs = dd_enc_outputs + dd_enc_inputs
					dd_enc_outputs = self.dropout(dd_enc_outputs)\
						.view(self.batch_size, self.max_seq_len, dd_enc_outputs.size(-1))

		# ******************************************************
		# ******************* DD DEC or CLASSIFY ***************
		if not self.dd_classifier:

			# ******************************************************
			# 2.5 att inputs: keys n values
			if type(dd_att_key_feats) == type(None):
				dd_att_keys = dd_enc_outputs
			else:
				# dd_att_key_feats: b x max_seq_len x additional_key_size
				assert self.dd_additional_key_size == dd_att_key_feats.size(-1),\
				 	'Mismatch in attention key dimension!'
				dd_att_keys = torch.cat((dd_enc_outputs, dd_att_key_feats), dim=2)
			dd_att_vals = dd_enc_outputs

			# ******************************************************
			# 3. init hidden states
			dd_dec_hidden = None

			# ******************************************************
			# 4. run dd dec: att + output
			"""
				teacher_forcing_ratio = 1.0 -> always teacher forcing

				E.g.:
					emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
					tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
					predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
					(shift-by-1)
			"""
			use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
			if not is_training:
				 use_teacher_forcing = False

			# no beam search decoding
			dd_tgt_chunk = dd_emb_tgt[:, 0].unsqueeze(1) # BOS
			dd_prev_c = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, self.max_seq_len).to(device=device)
			dd_cell_value = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, self.state_size).to(device=device)

			for dd_idx in range(self.max_seq_len - 1):

				if debug_flag: import pdb; pdb.set_trace()

				dd_predicted_logsoftmax, dd_dec_hidden, dd_step_attn,\
				 	dd_c_out, dd_att_outputs, dd_cell_value, p_gen = \
						self.forward_step(dd_att_keys, dd_att_vals,
							dd_tgt_chunk, dd_cell_value,
							self.dd_dec, self.dd_num_unilstm_dec,
							self.dd_att, self.dd_ffn, self.dd_out,
							dd_dec_hidden, dd_mask_src, dd_prev_c)
				dd_predicted_logsoftmax = dd_predicted_logsoftmax.squeeze(1)
				dd_predicted_softmax = torch.exp(dd_predicted_logsoftmax)

				# -------------------------------------------------------
				# ptr network - only used for dd, no beam search needed
				if self.ptr_net == 'comb':

					# import pdb; pdb.set_trace()
					attn_src_softmax = torch.FloatTensor([10**(-10)]).repeat(
						self.batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(self.batch_size))\
						.repeat(self.max_seq_len,1).transpose(0,1)\
						.contiguous().view(1,-1).to(device=device) # 00..11..22..b-1b-1..
					yidices = dd_src.view(1,-1)
					probs = dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					p_gen = p_gen.squeeze(1).view(self.batch_size, 1) # [b, 1]
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					predicted_softmax_comb = \
						p_gen * dd_predicted_softmax + (1-p_gen) * attn_src_softmax
					dd_step_output = torch.log(predicted_softmax_comb)

				elif self.ptr_net == 'pure':

					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(self.batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(self.batch_size))\
						.repeat(self.max_seq_len,1).transpose(0,1)\
						.contiguous().view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = dd_src.view(1,-1)
					probs = dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					dd_step_output = torch.log(attn_src_softmax)

				elif self.ptr_net == 'null':
					dd_step_output = dd_predicted_logsoftmax
				else:
					assert False, 'Not implemented: ptr_net mode - {}'.format(self.ptr_net)
				# --------------------------------------------------------

				dd_symbols, dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths = \
					decode(dd_idx, dd_step_output, dd_step_attn,
						dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths)
				dd_prev_c = dd_c_out
				if use_teacher_forcing:
					dd_tgt_chunk = dd_emb_tgt[:, dd_idx+1].unsqueeze(1)
				else:
					dd_tgt_chunk = self.embedder_dec(dd_symbols)

				# discriminator
				if self.add_discriminator:
					if self.shared_embed == 'context':
						dis_input = dd_att_outputs
					elif self.shared_embed == 'state':
						dis_input = dd_cell_value
					elif self.shared_embed == 'state_tgt':
						state_tgt = torch.cat([dd_tgt_chunk, dd_cell_value], -1)
						dis_input = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
					else:
						assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)
					dd_ret_dict[KEY_DIS].append(self.discriminator(dis_input))

			dd_ret_dict[KEY_SEQUENCE] = dd_sequence_symbols # b * max_len - 1
			dd_ret_dict[KEY_LENGTH] = dd_lengths.tolist()

		else: # do simple classification for dd

			"""
			vars:
				produce dd_decoder_outputs, dd_dec_hidden, dd_ret_dict using classifier results
				dd_decoder_outputs: log-prob list; max_len-1 * [b * vocab_size]
				dd_dec_hidden: last hidden state (not v important)
				dd_ret_dict[KEY_SEQUENCE]: word ids list; max_len-1 * [b * vocab_size]
			added:
				dd_ret_dict[CLASSIFY_PROB]: dd classifier output # b * max_len * 2
			note:
				dd_src: a 	cat si 	sat ...
				dd_mask:0	0	1	0
				dd_tgt: a 	can	sat
			"""
			# import pdb; pdb.set_trace()

			# classification
			dd_dec_hidden = None
			dd_decoder_outputs = None
			dd_probs = self.dd_classify(dd_enc_outputs) # b * max_len * 1
			dd_ret_dict[CLASSIFY_PROB] = dd_probs

			# keep fluent only
			dd_labels = dd_probs.ge(0.5).long()\
				.view(self.batch_size, self.max_seq_len) # b * max_len * 1 : 1=O, 0=E
			dummy = torch.autograd.Variable(
				torch.LongTensor(self.max_seq_len).fill_(31), requires_grad=False)
			fluent_idx = [(dd_labels[i,:] == 1).nonzero().view(-1)
				for i in range(self.batch_size)] # b * [num_fluent]
			fluent_idx.append(dummy)
			gather_col_idx = torch.nn.utils.rnn.pad_sequence(
				fluent_idx, batch_first=True, padding_value=31).long()[:-1,:]
			fluent_symbols = torch.gather(dd_src, 1, gather_col_idx) # b * max_len

			# record into dict
			dd_ret_dict[KEY_SEQUENCE] = [
				torch.LongTensor(elem).to(device=device) for elem \
				in torch.transpose(fluent_symbols, 0, 1).tolist()][:-1] # max_len - 1 * b

		# ***************************************************************
		# ***********************************************[GEC src-DD dec]

		# ******************************************************
		# 0. init var

		# intermediate output from dd-decoding
		gec_dd_ret_dict = dict()
		gec_dd_ret_dict[KEY_DIS] = []
		gec_dd_ret_dict[KEY_ATTN_SCORE] = []
		gec_dd_decoder_outputs = []
		gec_dd_sequence_symbols = []
		gec_dd_lengths = np.array([self.max_seq_len] * batch_size)
		gec_dd_embedding = []

		# 1. convert id to embedding
		gec_dd_emb_src = self.embedding_dropout(self.embedder_enc(gec_src))
		gec_dd_mask_src = gec_src.data.eq(PAD)
		if type(gec_tgt) == type(None):
			gec_tgt = torch.Tensor([BOS]).repeat(
				gec_src.size()).type(torch.LongTensor).to(device=device)
		gec_emb_tgt = self.embedding_dropout(self.embedder_dec(gec_tgt))

		# ******************************************************
		# 2. run enc
		gec_dd_enc_hidden_init = None
		gec_dd_enc_outputs, gec_dd_enc_hidden = self.enc(
			gec_dd_emb_src, gec_dd_enc_hidden_init)
		gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
			.view(self.batch_size, self.max_seq_len, gec_dd_enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				gec_dd_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				gec_dd_enc_outputs, gec_enc_hidden_uni = self.enc_uni(
					gec_dd_enc_outputs, gec_dd_enc_hidden_uni_init)
				gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
					.view(self.batch_size, self.max_seq_len, gec_dd_enc_outputs.size(-1))
			else:
				gec_dd_enc_hidden_uni_init = None
				gec_dd_enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					gec_dd_enc_inputs = gec_dd_enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					gec_dd_enc_outputs, gec_dd_enc_hidden_uni = enc_func(
						gec_dd_enc_inputs, gec_dd_enc_hidden_uni_init)
					gec_dd_enc_hidden_uni_lis.append(gec_dd_enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						gec_dd_enc_outputs = gec_dd_enc_outputs + gec_dd_enc_inputs
					gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
						.view(self.batch_size, self.max_seq_len, gec_dd_enc_outputs.size(-1))

		# ******************************************************
		# ******************* DD DEC or CLASSIFY ***************
		if not self.dd_classifier:

			# ******************************************************
			# 2.5 att inputs: keys n values
			if type(gec_dd_att_key_feats) == type(None):
				if self.dd_additional_key_size == 0:
					gec_dd_att_keys = gec_dd_enc_outputs
				else:
					# handle gec set no dd ref att probs
					dummy_feats = torch.autograd.Variable(
						torch.FloatTensor(gec_dd_enc_outputs.size()[0],
						gec_dd_enc_outputs.size()[1], self.dd_additional_key_size).fill_(0.0),
						requires_grad=False).to(device)
					gec_dd_att_keys = torch.cat((gec_dd_enc_outputs, dummy_feats), dim=2)
			else:
				# gec_dd_att_key_feats: b x max_seq_len x additional_key_size
				assert self.dd_additional_key_size == gec_dd_att_key_feats.size(-1), \
					'Mismatch in attention key dimension!'
				gec_dd_att_keys = torch.cat((gec_dd_enc_outputs, gec_dd_att_key_feats), dim=2)
			gec_dd_att_vals = gec_dd_enc_outputs

			# ******************************************************
			# 3. init hidden states
			gec_dd_dec_hidden = None

			# ******************************************************
			# 4. run dd dec: att + output
			"""
				Note:
					cannot use teacher forcing; no dd ref for clc
					have to use generated sequence

				gec_dd_embedding -> become gec_emb_src(list of list):
					is the embedding to be used for GEC decoder
					(T-1) * [batch_size x (self.hidden_size_enc * 2)] (T-1=31)
					need to convert to tensor of [b x T x h*2] (T=32)
					note: 	1. append h for initial batche of <s>'s from enc output
							2. trailing <pad> should be learnt through DD (<pad> penalised)

				E.g.:
					emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
					tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
					predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
					(shift-by-1)

					gec_dd_sequence_symbols
									= <s> w1 w2 w3 </s> <pad> <pad> <pad>	[max_seq_len]
			"""

			# no beam search decoding + no teacher forcing
			# initial <s>'s from enc output
			gec_dd_tgt_chunk = gec_emb_tgt[:, 0].unsqueeze(1) # BOS (okay to use 1st symbol = BOS)
			gec_dd_prev_c = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, self.max_seq_len).to(device=device)
			gec_dd_cell_value = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, self.state_size).to(device=device)

			for gec_dd_idx in range(self.max_seq_len - 1):

				if debug_flag:
					import pdb; pdb.set_trace()

				gec_dd_predicted_logsoftmax, gec_dd_dec_hidden, gec_dd_step_attn, \
					gec_dd_c_out, gec_dd_att_outputs, gec_dd_cell_value, p_gen = \
						self.forward_step(gec_dd_att_keys, gec_dd_att_vals,
							gec_dd_tgt_chunk, gec_dd_cell_value,
							self.dd_dec, self.dd_num_unilstm_dec,
							self.dd_att, self.dd_ffn, self.dd_out,
							gec_dd_dec_hidden, gec_dd_mask_src, gec_dd_prev_c)

				gec_dd_predicted_logsoftmax = gec_dd_predicted_logsoftmax.squeeze(1)
				gec_dd_predicted_softmax = torch.exp(gec_dd_predicted_logsoftmax)
				# -------------------------------------------------------
				# ptr network - only used for dd, no beam search needed
				if self.ptr_net == 'comb':
					# import pdb; pdb.set_trace()
					# attn_src_softmax
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(self.batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(self.batch_size))\
						.repeat(self.max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = gec_src.view(1,-1)
					probs = gec_dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					p_gen = p_gen.squeeze(1).view(self.batch_size, 1) # [b, 1]
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					predicted_softmax_comb = p_gen * gec_dd_predicted_softmax + (1-p_gen) * attn_src_softmax
					gec_dd_step_output = torch.log(predicted_softmax_comb)

				elif self.ptr_net == 'pure':
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(self.batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(self.batch_size))\
						.repeat(self.max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = gec_src.view(1,-1)
					probs = gec_dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					gec_dd_step_output = torch.log(attn_src_softmax)

				elif self.ptr_net == 'null':
					gec_dd_step_output = gec_dd_predicted_logsoftmax
				else:
					assert False, 'Not implemented: ptr_net mode - {}'.format(self.ptr_net)
				# --------------------------------------------------------

				gec_dd_symbols, gec_dd_ret_dict, gec_dd_decoder_outputs,\
					gec_dd_sequence_symbols, gec_dd_lengths = \
						decode(gec_dd_idx, gec_dd_step_output, gec_dd_step_attn,
							gec_dd_ret_dict, gec_dd_decoder_outputs,
							gec_dd_sequence_symbols, gec_dd_lengths)
				gec_dd_prev_c = gec_dd_c_out
				gec_dd_tgt_chunk = self.embedder_dec(gec_dd_symbols)
				# - context
				if self.shared_embed == 'context':
					gec_dd_embedding.append(gec_dd_att_outputs)
				elif self.shared_embed == 'state':
					gec_dd_embedding.append(gec_dd_cell_value)
				elif self.shared_embed == 'state_tgt':
					state_tgt = torch.cat([gec_dd_tgt_chunk, gec_dd_cell_value], -1)
					state_tgt = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
					gec_dd_embedding.append(state_tgt)
				else:
					assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)

				# discriminator
				if self.add_discriminator:
					if self.shared_embed == 'context':
						dis_input = gec_dd_att_outputs
					elif self.shared_embed == 'state':
						dis_input = gec_dd_cell_value
					elif self.shared_embed == 'state_tgt':
						state_tgt = torch.cat([gec_dd_tgt_chunk, gec_dd_cell_value], -1)
						dis_input = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
					else:
						assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)
					gec_dd_ret_dict[KEY_DIS].append(self.discriminator(dis_input))


			dummy = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, gec_dd_embedding[-1].size(-1)).to(device=device)
			gec_dd_embedding.append(dummy) #embedding
			gec_dd_sequence_symbols.append(
				torch.LongTensor([PAD] * self.batch_size).to(device=device).unsqueeze(1))
			gec_dd_ret_dict[KEY_SEQUENCE] = gec_dd_sequence_symbols # b * max_len
			gec_dd_ret_dict[KEY_LENGTH] = gec_dd_lengths.tolist()

		else:

			# classification
			gec_dd_dec_hidden = None
			gec_dd_decoder_outputs = None
			gec_dd_probs = self.dd_classify(gec_dd_enc_outputs) # b * max_len * 1
			gec_dd_ret_dict[CLASSIFY_PROB] = gec_dd_probs

			# keep fluent only
			gec_dd_labels = gec_dd_probs.ge(0.5).long().view(
				self.batch_size, self.max_seq_len) # b * max_len * 1 : 1=O, 0=E
			dummy = torch.autograd.Variable(torch.LongTensor(
				self.max_seq_len).fill_(31), requires_grad=False)
			fluent_idx = [(gec_dd_labels[i,:] == 1).nonzero().view(-1)
				for i in range(self.batch_size)] # b * [num_fluent]
			fluent_idx.append(dummy)
			gather_col_idx = torch.nn.utils.rnn.pad_sequence(
				fluent_idx, batch_first=True, padding_value=31).long()[:-1,:]
			gec_fluent_symbols = torch.gather(gec_src, 1, gather_col_idx) # b * max_len

			# record into dict
			gec_dd_ret_dict[KEY_SEQUENCE] = [
				torch.LongTensor(elem).to(device=device) for elem \
				in torch.transpose(gec_fluent_symbols, 0, 1).tolist()][:-1] # max_len - 1 * b

		# ***********************************************************
		# ******************************************[GEC src-GEC dec]

		# ******************************************************
		# 0. init var

		# final output from gec-decoding output
		gec_ret_dict = dict()
		gec_ret_dict[KEY_ATTN_SCORE] = []
		gec_decoder_outputs = []
		gec_sequence_symbols = []
		gec_lengths = np.array([self.max_seq_len] * batch_size)

		# 1. convert intermediate result from gec_dd to gec input
		if not self.dd_classifier:
			# a. gec_dd_embedding from list to tensor
			gec_enc_outputs = torch.cat(gec_dd_embedding, dim=1).to(device=device)
			# b. gec_dd_sequence_symbols from list to tensor; then get mask
			gec_dd_res = torch.cat(gec_dd_sequence_symbols, dim=1).to(device=device)
			gec_mask_src = gec_dd_res.data.eq(PAD)
		else:

			gec_dd_res = gec_fluent_symbols
			gec_mask_src = gec_dd_res.data.eq(PAD)

		# 1.5. different connetion type - embedding or word
		# import pdb; pdb.set_trace()
		self.check_classvar('connect_type')
		if not self.dd_classifier:
			if self.connect_type == 'embed':
				pass
			elif 'word' in self.connect_type:
				if self.connect_type == 'wordhard':
					hard = True
				elif self.connect_type == 'wordsoft':
					hard = False
				dummy = torch.FloatTensor([1e-40]).repeat(
					batch_size, gec_dd_decoder_outputs[-1].size(-1)).to(device=device) # b x z
				dummy[:,0] = .99
				gec_dd_decoder_outputs.append(torch.log(dummy))
				logits = torch.stack(gec_dd_decoder_outputs, dim=1) # b x l x  z
				samples = F.gumbel_softmax(logits, tau=1, hard=hard)
				gec_enc_outputs = torch.matmul(samples, self.embedder_dec.weight)
			else:
				assert False, 'connect_type not implemented'
		else:
			if self.connect_type == 'embed':
				gec_fluent_embeddings = gec_dd_enc_outputs[
					torch.arange(gec_dd_enc_outputs.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * (hidden_size_enc * 2)
				gec_enc_outputs = gec_fluent_embeddings
			if 'word' in self.connect_type:
				gec_enc_outputs = gec_dd_emb_src[torch.arange(
					gec_dd_emb_src.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * embedding_size
			elif self.connect_type == 'prob':
				# using the same length sequence as src
				gec_enc_outputs = gec_dd_probs * gec_dd_emb_src # b * max_len * embedding_size
				gec_mask_src = gec_dd_mask_src
			else:
				assert False, 'connect_type not implemented'

		# 2. run pre-attention dec unilstm [in traditional nmt, part of enc]
		# bilstm

		if self.gec_num_bilstm_dec != 0:
			gec_enc_hidden_init = None
			gec_enc_outputs, gec_enc_hidden = self.gec_dec_bilstm(
				gec_enc_outputs, gec_enc_hidden_init)
			gec_enc_outputs = self.dropout(gec_enc_outputs)\
				.view(self.batch_size, self.max_seq_len, gec_enc_outputs.size(-1))

		# unilstm
		if self.gec_num_unilstm_dec_preatt != 0:

			if not self.residual:
				gec_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				gec_enc_outputs, gec_enc_hidden_uni = self.gec_dec_preatt(
					gec_enc_outputs, gec_enc_hidden_uni_init)
				gec_enc_outputs = self.dropout(gec_enc_outputs)\
					.view(self.batch_size, self.max_seq_len, gec_enc_outputs.size(-1))
			else:
				gec_enc_hidden_uni_init = None
				gec_enc_hidden_uni_lis = []
				for i in range(self.gec_num_unilstm_dec_preatt):
					gec_enc_inputs = gec_enc_outputs
					gec_dec_preatt_func = getattr(self.gec_dec_preatt, 'l'+str(i))
					gec_enc_outputs, gec_enc_hidden_uni = gec_dec_preatt_func(
						gec_enc_inputs, gec_enc_hidden_uni_init)
					gec_enc_hidden_uni_lis.append(gec_enc_hidden_uni)
					if i < self.gec_num_unilstm_dec_preatt - 1: # no residual for last layer
						gec_enc_outputs = gec_enc_outputs + gec_enc_inputs
					gec_enc_outputs = self.dropout(gec_enc_outputs)\
						.view(self.batch_size, self.max_seq_len, gec_enc_outputs.size(-1))

		# ******************************************************
		# 2.5 att inputs: keys n values
		gec_att_keys = gec_enc_outputs
		gec_att_vals = gec_enc_outputs

		# ******************************************************
		# 3. init hidden states
		gec_dec_hidden = None

		# ******************************************************
		# 4. run gec dec: att + output
		"""
			teacher_forcing_ratio = 1.0 -> always teacher forcing

			E.g.:
				emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
				tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
				predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
				(shift-by-1)
		"""
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		if not is_training:
			 use_teacher_forcing = False

		# beam search decoding
		if not is_training and self.beam_width > 1:
			gec_decoder_outputs, gec_decoder_hidden, gec_metadata = \
				self.beam_search_decoding(gec_att_keys, gec_att_vals, self.gec_dec_pstatt,
				self.gec_num_unilstm_dec_pstatt, self.gec_attn, self.gec_out,
				gec_dec_hidden, gec_mask_src, beam_width=self.beam_width)

			return gec_decoder_outputs, gec_decoder_hidden, gec_metadata

		# no beam search decoding
		gec_tgt_chunk = gec_emb_tgt[:, 0].unsqueeze(1) # BOS
		gec_prev_c = torch.FloatTensor([0]).repeat(
			self.batch_size, 1, self.max_seq_len).to(device=device)
		gec_cell_value = torch.FloatTensor([0]).repeat(
			self.batch_size, 1, self.state_size).to(device=device)

		for gec_idx in range(self.max_seq_len - 1):

			if debug_flag:
				import pdb; pdb.set_trace()

			gec_predicted_logsoftmax, gec_dec_hidden, gec_step_attn, \
				gec_c_out, gec_att_outputs, gec_cell_value, _ = \
					self.forward_step(gec_att_keys, gec_att_vals,
						gec_tgt_chunk, gec_cell_value,
						self.gec_dec_pstatt, self.gec_num_unilstm_dec_pstatt,
						self.gec_att, self.gec_ffn, self.gec_out,
						gec_dec_hidden, gec_mask_src, gec_prev_c)
			gec_predicted_logsoftmax = gec_predicted_logsoftmax.squeeze(1)
			gec_step_output = gec_predicted_logsoftmax
			gec_symbols, gec_ret_dict, gec_decoder_outputs, \
				gec_sequence_symbols, gec_lengths = \
					decode(gec_idx, gec_step_output, gec_step_attn,
						gec_ret_dict, gec_decoder_outputs, gec_sequence_symbols, gec_lengths)
			gec_prev_c = gec_c_out
			if use_teacher_forcing:
				gec_tgt_chunk = gec_emb_tgt[:, gec_idx+1].unsqueeze(1)
			else:
				gec_tgt_chunk = self.embedder_dec(gec_symbols)

		gec_ret_dict[KEY_SEQUENCE] = gec_sequence_symbols
		gec_ret_dict[KEY_LENGTH] = gec_lengths.tolist()

		return dd_decoder_outputs, dd_dec_hidden, dd_ret_dict, \
					gec_dd_decoder_outputs, gec_dd_dec_hidden, gec_dd_ret_dict, \
					gec_decoder_outputs, gec_dec_hidden, gec_ret_dict


	def forward_step(self, att_keys, att_vals, tgt_chunk, prev_cell_value,
		dec_function, num_dec_layer, att_func, ffn_func, out_func,
		dec_hidden=None, mask_src=None, prev_c=None, att_ref=None):

		"""
			manual unrolling - can only operate per time step
			shared by dd and gec decoder

			Args:
				att_keys:   [batch_size, seq_len, hidden_size_enc * 2 + optional key size (key_size)]
				att_vals:   [batch_size, seq_len, hidden_size_enc * 2 (val_size)]
				tgt_chunk:  tgt word embeddings
							non teacher forcing - [batch_size, embedding_size_dec] (lose 1 dim when indexed)
							teacher forcing - [batch_size, 1, embedding_size_dec]
				prev_cell_value:
							previous cell value before prediction [batch_size, 1, self.state_size]
				num_dec_layer:
							number of decoder layer; always compatibled with dec_func
				dec_function:
							self.dd_dec / self.gec_dec_pstatt: decoder lstms
				att_func:
							self.dd_att / self.gec_att: attention mechanism function
				out_func:
							self.dd_out / self.gec_out: output dense layer
				dec_hidden:
							hidden state for dec layer
				mask_src:
							mask of PAD for src sequences
				prev_c:
							used in hybrid attention mechanism

			Returns:
				predicted_softmax: log probilities [batch_size, 1, vocab_size_dec]
				dec_hidden: a list of hidden states of each dec layer
				attn: attention weights
		"""
		# record sizes
		batch_size = tgt_chunk.size(0)
		tgt_chunk_ext = torch.cat([tgt_chunk, prev_cell_value], -1)

		tgt_chunk_ext = tgt_chunk_ext.view(-1, 1, self.embedding_size + self.state_size)

		# run dec
		# default dec_hidden: [h_0, c_0]; with h_0 [num_layers * num_directions(==1), batch, hidden_size]
		if not self.residual:
			dec_outputs, dec_hidden = dec_funcion(tgt_chunk_ext, dec_hidden)
			dec_outputs = self.dropout(dec_outputs)
		else:
			# store states layer by layer num_layers * ([1, batch, hidden_size], [1, batch, hidden_size])
			dec_hidden_lis = []
			# import pdb; pdb.set_trace()
			# layer0
			dec_func_first = getattr(dec_function, 'l0')
			if type(dec_hidden) == type(None):
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_ext, None)
			else:

				index = torch.tensor([0]).to(device=device) # choose the 0th layer
				dec_hidden_in = tuple([h.index_select(dim=0, index=index) for h in dec_hidden])
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_ext, dec_hidden_in)
			dec_hidden_lis.append(dec_hidden_out)
			# no residual for 0th layer
			dec_outputs = self.dropout(dec_outputs)

			# layer1+
			for i in range(1, num_dec_layer):
				dec_inputs = dec_outputs
				dec_func = getattr(dec_function, 'l'+str(i))
				if type(dec_hidden) == type(None):
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, None)
				else:
					index = torch.tensor([i]).to(device=device) # choose the 0th layer
					dec_hidden_in = tuple([h.index_select(dim=0, index=index) for h in dec_hidden])
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, dec_hidden_in)
				dec_hidden_lis.append(dec_hidden_out)
				if i < num_dec_layer - 1:
					dec_outputs = dec_outputs + dec_inputs
				dec_outputs = self.dropout(dec_outputs)

			# convert to tuple
			h_0 = torch.cat([h[0] for h in dec_hidden_lis], 0)
			c_0 = torch.cat([h[1] for h in dec_hidden_lis], 0)
			dec_hidden = tuple([h_0, c_0])

		# run att
		att_func.set_mask(mask_src)
		att_outputs, attn, c_out = att_func(dec_outputs, att_keys, att_vals, prev_c=prev_c, att_ref=att_ref)
		att_outputs = self.dropout(att_outputs) # weighted sum of att keys

		# run ff + softmax
		ff_inputs = torch.cat((att_outputs, dec_outputs), dim=-1)
		ff_inputs_size = att_outputs.size(-1) + self.hidden_size_dec
		cell_value = ffn_func(ff_inputs.view(-1, 1, ff_inputs_size)) # ? + 200 -> 200
		outputs = out_func(cell_value.contiguous().view(-1, self.state_size)) # 200 -> vocab_size
		predicted_logsoftmax = F.log_softmax(outputs, dim=1).view(batch_size, 1, -1)

		# ptr net
		if self.ptr_net == 'comb':
			p_gen_i = self.ptr_i(tgt_chunk)
			p_gen_s = self.ptr_s(dec_outputs)
			p_gen_c = self.ptr_c(att_outputs)
			p_gen = torch.sigmoid(p_gen_i + p_gen_s + p_gen_c)
		elif self.ptr_net == 'pure':
			p_gen = torch.FloatTensor([0]).repeat(self.batch_size, 1, 1).to(device=device)
		elif self.ptr_net == 'null':
			p_gen = torch.FloatTensor([1]).repeat(self.batch_size, 1, 1).to(device=device)
		else:
			assert False, 'Not implemented ptr_net mode - {}'.format(self.ptr_net)

		return predicted_logsoftmax, dec_hidden, attn, c_out, att_outputs, cell_value, p_gen


	# =====================================================================================
	# beam search

	def beam_search_decoding(self, att_keys, att_vals,
		dec_func, num_dec_layer, att_func, ffn_func, out_func,
		dec_hidden=None, mask_src=None, prev_c=None, beam_width=10):

		"""
			beam search decoding - only used for evaluation
			Modified from -
			 	https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py

			Shortcuts:
				beam_width: k
				batch_size: b
				vocab_size: v
				max_seq_len: l

			Args:
				att_keys:   [b x l x hidden_size_enc * 2 + optional key size (key_size)]
				att_vals:   [b x l x hidden_size_enc * 2 (val_size)]
				dec_hidden:
							initial hidden state for dec layer [b x h_dec]
				mask_src:
							mask of PAD for src sequences
				beam_width: beam width kept during searching

			Returns:
				decoder_outputs: output probabilities [(batch, 1, vocab_size)] * T
				decoder_hidden (num_layers * num_directions, batch, hidden_size):
					tensor containing the last hidden state of the decoder.
				ret_dict: dictionary containing additional information as follows
				{
					*length* :
						list of integers representing lengths of output sequences,
					*topk_length*:
						list of integers representing lengths of beam search sequences,
					*sequence* :
						list of sequences, where each sequence is a list of predicted token IDs,
					*topk_sequence* :
						list of beam search sequences, each beam is a list of token IDs,
					*outputs* : [(batch, k, vocab_size)] * sequence_length:
						A list of the output probabilities (p_n)
				}.
		"""

		# define var
		self.beam_width = beam_width
		self.pos_index = Variable(torch.LongTensor(
			range(self.batch_size)) * self.beam_width).view(-1, 1).to(device=device)

		# initialize the input vector; att_c_value
		input_var = Variable(torch.transpose(
			torch.LongTensor([[BOS] * self.batch_size * self.beam_width]), 0, 1)).to(device=device)
		input_var_emb = self.embedder_dec(input_var).to(device=device)
		prev_c = torch.FloatTensor([0]).repeat(
			self.batch_size, 1, self.max_seq_len).to(device=device)
		cell_value = torch.FloatTensor([0]).repeat(
			self.batch_size, 1, self.state_size).to(device=device)

		# inflate attention keys and values (derived from encoder outputs)
		inflated_att_keys = att_keys.repeat_interleave(self.beam_width, dim=0)
		inflated_att_vals = att_vals.repeat_interleave(self.beam_width, dim=0)
		inflated_mask_src = mask_src.repeat_interleave(self.beam_width, dim=0)
		inflated_prev_c = prev_c.repeat_interleave(self.beam_width, dim=0)
		inflated_cell_value = cell_value.repeat_interleave(self.beam_width, dim=0)

		# inflate hidden states and others
		# note that inflat_hidden_state might be faulty - currently using None so it's fine
		dec_hidden = inflat_hidden_state(dec_hidden, self.beam_width)

		# Initialize the scores; for the first step,
		# ignore the inflated copies to avoid duplicate entries in the top k
		sequence_scores = torch.Tensor(self.batch_size * self.beam_width, 1).to(device=device)
		sequence_scores.fill_(-float('Inf'))
		sequence_scores.index_fill_(0, torch.LongTensor(
			[i * self.beam_width for i in range(0, self.batch_size)]).to(device=device), 0.0)
		sequence_scores = Variable(sequence_scores)

		# Store decisions for backtracking
		stored_outputs = list()         # raw softmax scores [bk x v] * T
		stored_scores = list()          # topk scores [bk] * T
		stored_predecessors = list()    # preceding beam idx (from 0-bk) [bk] * T
		stored_emitted_symbols = list() # word ids [bk] * T
		stored_hidden = list()          #

		for _ in range(self.max_seq_len):

			predicted_logsoftmax, dec_hidden, step_attn, inflated_c_out, _, inflated_cell_value, _ = \
				self.forward_step(inflated_att_keys, inflated_att_vals,
					input_var_emb, inflated_cell_value,
					dec_func, num_dec_layer, att_func, ffn_func, out_func,
					dec_hidden, inflated_mask_src, inflated_prev_c)
			inflated_prev_c = inflated_c_out

			# retain output probs
			stored_outputs.append(predicted_logsoftmax) # [bk x v]

			# To get the full sequence scores for the new candidates,
			# add the local scores for t_i to the predecessor scores for t_(i-1)
			sequence_scores = _inflate(sequence_scores, self.vocab_size, 1)
			sequence_scores += predicted_logsoftmax.squeeze(1) # [bk x v]

			scores, candidates = sequence_scores.view(
				self.batch_size, -1).topk(self.beam_width, dim=1) # [b x kv] -> [b x k]

			# Reshape input = (bk, 1) and sequence_scores = (bk, 1)
			input_var = (candidates % self.vocab_size).view(
				self.batch_size * self.beam_width, 1).to(device=device)
			input_var_emb = self.embedder_dec(input_var)
			sequence_scores = scores.view(self.batch_size * self.beam_width, 1) #[bk x 1]

			# Update fields for next timestep: store best paths
			predecessors = (candidates / self.vocab_size + self.pos_index.expand_as(candidates)).\
							view(self.batch_size * self.beam_width, 1)

			# dec_hidden: [h_0, c_0]; with h_0 [num_layers * num_directions, batch, hidden_size]
			if isinstance(dec_hidden, tuple):
				dec_hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in dec_hidden])
			else:
				dec_hidden = dec_hidden.index_select(1, predecessors.squeeze())

			stored_scores.append(sequence_scores.clone())

			# Cache results for backtracking
			stored_predecessors.append(predecessors)
			stored_emitted_symbols.append(input_var)
			stored_hidden.append(dec_hidden)

		# Do backtracking to return the optimal values
		output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
			stored_predecessors, stored_emitted_symbols,
			stored_scores, self.batch_size, self.hidden_size_dec)

		# Build return objects
		decoder_outputs = [step[:, 0, :].squeeze(1) for step in output]
		if isinstance(h_n, tuple):
			decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
		else:
			decoder_hidden = h_n[:, :, 0, :]
		metadata = {}
		metadata['output'] = output
		metadata['h_t'] = h_t
		metadata['score'] = s
		metadata['topk_length'] = l
		metadata['topk_sequence'] = p # [b x k x 1] * T
		metadata['length'] = [seq_len[0] for seq_len in l]
		metadata['sequence'] = [seq[:, 0] for seq in p]

		return decoder_outputs, decoder_hidden, metadata


	def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):

		"""
			Backtracks over batch to generate optimal k-sequences.
			https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py

			Args:
				nw_output [(batch*k, vocab_size)] * sequence_length:
					A Tensor of outputs from network
				nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length:
					A Tensor of hidden states from network
				predecessors [(batch*k)] * sequence_length:
					A Tensor of predecessors
				symbols [(batch*k)] * sequence_length:
				 	A Tensor of predicted tokens
				scores [(batch*k)] * sequence_length:
					A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
				b: Size of the batch
				hidden_size: Size of the hidden state

			Returns:
				output [(batch, k, vocab_size)] * sequence_length:
					A list of the output probabilities (p_n)
				from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
				h_t [(batch, k, hidden_size)] * sequence_length:
					A list containing the output features (h_n)
				from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
				h_n(batch, k, hidden_size):
					A Tensor containing the last hidden state for all top-k sequences.
				score [batch, k]:
					A list containing the final scores for all top-k sequences
				length [batch, k]:
					A list specifying the length of each sequence in the top-k candidates
				p (batch, k, sequence_len):
					A Tensor containing predicted sequence [b x k x 1] * T
		"""

		# initialize return variables given different types
		output = list()
		h_t = list()
		p = list()

		# Placeholder for last hidden state of top-k sequences.
		# If a (top-k) sequence ends early in decoding, `h_n` contains
		# its hidden state when it sees EOS.  Otherwise, `h_n` contains
		# the last hidden state of decoding.
		lstm = isinstance(nw_hidden[0], tuple)
		if lstm:
			state_size = nw_hidden[0][0].size()
			h_n = tuple([torch.zeros(state_size).to(device=device),
				torch.zeros(state_size).to(device=device)])
		else:
			h_n = torch.zeros(nw_hidden[0].size()).to(device=device)

		# Placeholder for lengths of top-k sequences
		# Similar to `h_n`
		l = [[self.max_seq_len] * self.beam_width for _ in range(b)]

		# the last step output of the beams are not sorted
		# thus they are sorted here
		sorted_score, sorted_idx = scores[-1].view(
			b, self.beam_width).topk(self.beam_width)
		sorted_score = sorted_score.to(device=device)
		sorted_idx = sorted_idx.to(device=device)

		# initialize the sequence scores with the sorted last step beam scores
		s = sorted_score.clone().to(device=device)

		batch_eos_found = [0] * b   # the number of EOS found
									# in the backward loop below for each batch

		t = self.max_seq_len - 1
		# initialize the back pointer with the sorted order of the last step beams.
		# add self.pos_index for indexing variable with b*k as the first dimension.
		t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx))\
			.view(b * self.beam_width).to(device=device)


		while t >= 0:
			# Re-order the variables with the back pointer
			current_output = nw_output[t].index_select(0, t_predecessors)
			if lstm:
				current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
			else:
				current_hidden = nw_hidden[t].index_select(1, t_predecessors)
			current_symbol = symbols[t].index_select(0, t_predecessors)

			# Re-order the back pointer of the previous step with the back pointer of
			# the current step
			t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze().to(device=device)

			"""
				This tricky block handles dropped sequences that see EOS earlier.
				The basic idea is summarized below:

				  Terms:
					  Ended sequences = sequences that see EOS early and dropped
					  Survived sequences = sequences in the last step of the beams

					  Although the ended sequences are dropped during decoding,
				  their generated symbols and complete backtracking information are still
				  in the backtracking variables.
				  For each batch, everytime we see an EOS in the backtracking process,
					  1. If there is survived sequences in the return variables, replace
					  the one with the lowest survived sequence score with the new ended
					  sequences
					  2. Otherwise, replace the ended sequence with the lowest sequence
					  score with the new ended sequence
			"""

			eos_indices = symbols[t].data.squeeze(1).eq(EOS).nonzero().to(device=device)
			if eos_indices.dim() > 0:
				for i in range(eos_indices.size(0)-1, -1, -1):
					# Indices of the EOS symbol for both variables
					# with b*k as the first dimension, and b, k for
					# the first two dimensions
					idx = eos_indices[i]
					b_idx = int(idx[0] / self.beam_width)
					# The indices of the replacing position
					# according to the replacement strategy noted above
					res_k_idx = self.beam_width - (batch_eos_found[b_idx] % self.beam_width) - 1
					batch_eos_found[b_idx] += 1
					res_idx = b_idx * self.beam_width + res_k_idx

					# Replace the old information in return variables
					# with the new ended sequence information
					t_predecessors[res_idx] = predecessors[t][idx[0]].to(device=device)
					current_output[res_idx, :] = nw_output[t][idx[0], :].to(device=device)
					if lstm:
						current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].to(device=device)
						current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].to(device=device)
						h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data.to(device=device)
						h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data.to(device=device)
					else:
						current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :].to(device=device)
						h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data.to(device=device)
					current_symbol[res_idx, :] = symbols[t][idx[0]].to(device=device)
					s[b_idx, res_k_idx] = scores[t][idx[0]].data[0].to(device=device)
					l[b_idx][res_k_idx] = t + 1

			# record the back tracked results
			output.append(current_output)
			h_t.append(current_hidden)
			p.append(current_symbol)

			t -= 1

		# Sort and re-order again as the added ended sequences may change
		# the order (very unlikely)
		s, re_sorted_idx = s.topk(self.beam_width)
		for b_idx in range(b):
			l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

		re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(
			re_sorted_idx)).view(b * self.beam_width).to(device=device)

		# Reverse the sequences and re-order at the same time
		# It is reversed because the backtracking happens in reverse time order
		output = [step.index_select(0, re_sorted_idx).view(
			b, self.beam_width, -1) for step in reversed(output)]
		p = [step.index_select(0, re_sorted_idx).view(
			b, self.beam_width, -1) for step in reversed(p)]
		if lstm:
			h_t = [tuple([h.index_select(1, re_sorted_idx.to(device=device))\
				.view(-1, b, self.beam_width, hidden_size) for h in step]) for step in reversed(h_t)]
			h_n = tuple([h.index_select(1, re_sorted_idx.data.to(device=device))\
				.view(-1, b, self.beam_width, hidden_size) for h in h_n])
		else:
			h_t = [step.index_select(1, re_sorted_idx.to(device=device))\
				.view(-1, b, self.beam_width, hidden_size) for step in reversed(h_t)]
			h_n = h_n.index_select(1, re_sorted_idx.data.to(device=device))\
				.view(-1, b, self.beam_width, hidden_size)
		s = s.data

		return output, h_t, h_n, s, l, p


	# =====================================================================================
	# Training - separate dd / gec
	"""
		cropped out from forward
		training for enc+dd_dec / enc+gec_dd_dec+gec_dec
	"""

	def dd_train(self, dd_src, dd_tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=1.0,
		dd_att_key_feats=None, beam_width=0):

		# config device
		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		# init global var
		# ******************************************************
		batch_size = dd_src.size(0)
		max_seq_len = dd_src.size(1)
		self.beam_width = beam_width

		# 0. init var
		dd_ret_dict = dict()
		dd_ret_dict[KEY_ATTN_SCORE] = []
		dd_decoder_outputs = []
		dd_sequence_symbols = []
		dd_lengths = np.array([max_seq_len] * batch_size)

		# 1. convert id to embedding
		dd_emb_src = self.embedding_dropout(self.embedder_enc(dd_src))
		if type(dd_tgt) == type(None):
			dd_tgt = torch.Tensor([BOS]).repeat(dd_src.size())\
				.type(torch.LongTensor).to(device=device)
		dd_emb_tgt = self.embedding_dropout(self.embedder_dec(dd_tgt))
		dd_mask_src = dd_src.data.eq(PAD)

		# ******************************************************
		# 2. run enc
		dd_enc_hidden_init = None
		dd_enc_outputs, dd_enc_hidden = self.enc(dd_emb_src, dd_enc_hidden_init)
		dd_enc_outputs = self.dropout(dd_enc_outputs)\
						.view(batch_size, max_seq_len, dd_enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				dd_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				dd_enc_outputs, dd_enc_hidden_uni = self.enc_uni(
					dd_enc_outputs, dd_enc_hidden_uni_init)
				dd_enc_outputs = self.dropout(dd_enc_outputs)\
					.view(batch_size, max_seq_len, dd_enc_outputs.size(-1))
			else:
				dd_enc_hidden_uni_init = None
				dd_enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					dd_enc_inputs = dd_enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					dd_enc_outputs, dd_enc_hidden_uni = enc_func(
						dd_enc_inputs, dd_enc_hidden_uni_init)
					dd_enc_hidden_uni_lis.append(dd_enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						dd_enc_outputs = dd_enc_outputs + dd_enc_inputs
					dd_enc_outputs = self.dropout(dd_enc_outputs)\
						.view(batch_size, max_seq_len, dd_enc_outputs.size(-1))

		# ******************************************************
		# ******************* DD DEC or CLASSIFY ***************
		if not self.dd_classifier:

			# ******************************************************
			# 2.5 att inputs: keys n values
			if type(dd_att_key_feats) == type(None):
				dd_att_keys = dd_enc_outputs
			else:
				# dd_att_key_feats: b x max_seq_len x additional_key_size
				assert self.dd_additional_key_size == dd_att_key_feats.size(-1), \
					'Mismatch in attention key dimension!'
				dd_att_keys = torch.cat((dd_enc_outputs, dd_att_key_feats), dim=2)
			dd_att_vals = dd_enc_outputs

			# ******************************************************
			# 3. init hidden states
			dd_dec_hidden = None

			# ******************************************************
			# 4. run dd dec: att + output
			"""
				teacher_forcing_ratio = 1.0 -> always teacher forcing

				E.g.:
					emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
					tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
					predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
					(shift-by-1)
			"""
			use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
			if not is_training:
				 use_teacher_forcing = False

			# no beam search decoding
			dd_tgt_chunk = dd_emb_tgt[:, 0].unsqueeze(1) # BOS
			dd_prev_c = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, self.max_seq_len).to(device=device)
			dd_cell_value = torch.FloatTensor([0]).repeat(
				self.batch_size, 1, self.state_size).to(device=device)

			for dd_idx in range(self.max_seq_len - 1):

				dd_predicted_logsoftmax, dd_dec_hidden, dd_step_attn, \
					dd_c_out, dd_att_outputs, dd_cell_value, p_gen = \
						self.forward_step(dd_att_keys, dd_att_vals,
							dd_tgt_chunk, dd_cell_value,
							self.dd_dec, self.dd_num_unilstm_dec,
							self.dd_att, self.dd_ffn, self.dd_out,
							dd_dec_hidden, dd_mask_src, dd_prev_c)

				dd_predicted_logsoftmax = dd_predicted_logsoftmax.squeeze(1)
				dd_predicted_softmax = torch.exp(dd_predicted_logsoftmax)
				# -------------------------------------------------------
				# ptr network - only used for dd, no beam search needed
				if self.ptr_net == 'comb':
					# import pdb; pdb.set_trace()
					attn_src_softmax = torch.FloatTensor([10**(-10)]).repeat(
						self.batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(self.batch_size))\
						.repeat(self.max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..b-1b-1..
					yidices = dd_src.view(1,-1)
					probs = dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					p_gen = p_gen.squeeze(1).view(self.batch_size, 1) # [b, 1]
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					predicted_softmax_comb = p_gen * dd_predicted_softmax + (1-p_gen) * attn_src_softmax
					dd_step_output = torch.log(predicted_softmax_comb)

				elif self.ptr_net == 'pure':
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(self.batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(self.batch_size))\
						.repeat(self.max_seq_len,1).transpose(0,1)\
						.contiguous().view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = dd_src.view(1,-1)
					probs = dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					dd_step_output = torch.log(attn_src_softmax)

				elif self.ptr_net == 'null':
					dd_step_output = dd_predicted_logsoftmax
				else:
					assert False, 'Not implemented: ptr_net mode - {}'.format(self.ptr_net)
				# --------------------------------------------------------

				dd_symbols, dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths = \
					self.decode(dd_idx, dd_step_output, dd_step_attn,
						dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths)
				dd_prev_c = dd_c_out
				if use_teacher_forcing:
					dd_tgt_chunk = dd_emb_tgt[:, dd_idx+1].unsqueeze(1)
				else:
					dd_tgt_chunk = self.embedder_dec(dd_symbols)

			dd_ret_dict[KEY_SEQUENCE] = dd_sequence_symbols
			dd_ret_dict[KEY_LENGTH] = dd_lengths.tolist()

		else: # do simple classification for dd

			# classification
			dd_dec_hidden = None
			dd_decoder_outputs = None
			dd_probs = self.dd_classify(dd_enc_outputs) # b * max_len * 1
			dd_ret_dict[CLASSIFY_PROB] = dd_probs

			# keep fluent only
			# b * max_len * 1 : 1=O, 0=E
			dd_labels = dd_probs.ge(0.5).long().view(batch_size, max_seq_len)
			dummy = torch.autograd.Variable(
				torch.LongTensor(max_seq_len).fill_(31), requires_grad=False)
			fluent_idx = [(dd_labels[i,:] == 1).nonzero().view(-1)
				for i in range(batch_size)] # b * [num_fluent]
			fluent_idx.append(dummy)
			gather_col_idx = torch.nn.utils.rnn.pad_sequence(
				fluent_idx, batch_first=True, padding_value=31).long()[:-1,:]
			fluent_symbols = torch.gather(dd_src, 1, gather_col_idx) # b * max_len

			# record into dict
			dd_ret_dict[KEY_SEQUENCE] = [torch.LongTensor(elem).to(device=device) for elem \
				in torch.transpose(fluent_symbols, 0, 1).tolist()][:-1] # max_len - 1 * b

		return dd_decoder_outputs, dd_dec_hidden, dd_ret_dict


	def gec_train(self, gec_src, gec_tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=1.0,
		gec_dd_att_key_feats=None, beam_width=0):

		# config device
		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		# init global var
		# ******************************************************
		batch_size = gec_src.size(0)
		max_seq_len = gec_src.size(1)
		self.beam_width = beam_width

		# *************************************************************
		# ********************************************[GEC src-DD dec]

		# ******************************************************
		# 0. init var

		# intermediate output from dd-decoding
		gec_dd_ret_dict = dict()
		gec_dd_ret_dict[KEY_ATTN_SCORE] = []
		gec_dd_decoder_outputs = []
		gec_dd_sequence_symbols = []
		gec_dd_lengths = np.array([max_seq_len] * batch_size)
		gec_dd_embedding = []

		# 1. convert id to embedding
		gec_dd_emb_src = self.embedding_dropout(self.embedder_enc(gec_src))
		gec_dd_mask_src = gec_src.data.eq(PAD)
		if type(gec_tgt) == type(None):
			gec_tgt = torch.Tensor([BOS]).repeat(gec_src.size())\
				.type(torch.LongTensor).to(device=device)
		gec_emb_tgt = self.embedding_dropout(self.embedder_dec(gec_tgt))

		# ******************************************************
		# 2. run enc
		gec_dd_enc_hidden_init = None
		gec_dd_enc_outputs, gec_dd_enc_hidden = self.enc(gec_dd_emb_src, gec_dd_enc_hidden_init)
		gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
						.view(batch_size, max_seq_len, gec_dd_enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				gec_dd_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				gec_dd_enc_outputs, gec_enc_hidden_uni = self.enc_uni(
					gec_dd_enc_outputs, gec_dd_enc_hidden_uni_init)
				gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
					.view(batch_size, max_seq_len, gec_dd_enc_outputs.size(-1))
			else:
				gec_dd_enc_hidden_uni_init = None
				gec_dd_enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					gec_dd_enc_inputs = gec_dd_enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					gec_dd_enc_outputs, gec_dd_enc_hidden_uni = enc_func(
						gec_dd_enc_inputs, gec_dd_enc_hidden_uni_init)
					gec_dd_enc_hidden_uni_lis.append(gec_dd_enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						gec_dd_enc_outputs = gec_dd_enc_outputs + gec_dd_enc_inputs
					gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
						.view(batch_size, max_seq_len, gec_dd_enc_outputs.size(-1))

		# ******************************************************
		# ******************* DD DEC or CLASSIFY ***************
		if not self.dd_classifier:

			# ******************************************************
			# 2.5 att inputs: keys n values
			if type(gec_dd_att_key_feats) == type(None):
				if self.dd_additional_key_size == 0:
					gec_dd_att_keys = gec_dd_enc_outputs
				else:
					# handle gec set no dd ref att probs
					# print(gec_dd_enc_outputs.size())
					dummy_feats = torch.autograd.Variable(
						torch.FloatTensor(gec_dd_enc_outputs.size()[0],
						gec_dd_enc_outputs.size()[1], self.dd_additional_key_size).fill_(0.0),
						requires_grad=False).to(device)
					gec_dd_att_keys = torch.cat((gec_dd_enc_outputs, dummy_feats), dim=2)
			else:
				# gec_dd_att_key_feats: b x max_seq_len x additional_key_size
				assert self.dd_additional_key_size == gec_dd_att_key_feats.size(-1), \
					'Mismatch in attention key dimension!'
				gec_dd_att_keys = torch.cat((gec_dd_enc_outputs, gec_dd_att_key_feats), dim=2)

			gec_dd_att_vals = gec_dd_enc_outputs

			# ******************************************************
			# 3. init hidden states -
			gec_dd_dec_hidden = None

			# ******************************************************
			# 4. run dd dec: att + output
			"""
				Note:
					cannot use teacher forcing; no dd ref for clc
					have to use generated sequence

				gec_dd_embedding -> become gec_emb_src(list of list):
					is the embedding to be used for GEC decoder
					(T-1) * [batch_size x (self.hidden_size_enc * 2)] (T-1=31)
					need to convert to tensor of [b x T x h*2] (T=32)
					note: 	1. append h for initial batche of <s>'s from enc output
							2. trailing <pad> should be learnt through DD (<pad> penalised)

				E.g.:
					emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
					tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
					predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
					(shift-by-1)

					gec_dd_sequence_symbols
									= 	  w1 w2 w3 </s> <pad> <pad> <pad> + <pad>	[max_seq_len]
									(pad PAD at the end; keep max_seq_len the same)
					gec_dd intermediate state
									(showing only corresponding words)
									= 	  w1 w2 w3 </s> <pad> <pad> <pad> + 0's		[max_seq_len]
									(pad zeros at the end; keep max_seq_len the same)
			"""

			# no beam search decoding + no teacher forcing
			# initial <s>'s from enc output
			gec_dd_tgt_chunk = gec_emb_tgt[:, 0].unsqueeze(1) # BOS (okay to use 1st symbol = BOS)
			gec_dd_prev_c = torch.FloatTensor([0]).repeat(
				batch_size, 1, max_seq_len).to(device=device)
			gec_dd_cell_value = torch.FloatTensor([0]).repeat(
				batch_size, 1, self.state_size).to(device=device)

			for gec_dd_idx in range(max_seq_len - 1):
				# print(idx)
				gec_dd_predicted_logsoftmax, gec_dd_dec_hidden,\
					gec_dd_step_attn, gec_dd_c_out, gec_dd_att_outputs, gec_dd_cell_value, p_gen = \
						self.forward_step(gec_dd_att_keys, gec_dd_att_vals,
							gec_dd_tgt_chunk, gec_dd_cell_value,
							self.dd_dec, self.dd_num_unilstm_dec,
							self.dd_att, self.dd_ffn, self.dd_out,
							gec_dd_dec_hidden, gec_dd_mask_src, gec_dd_prev_c)

				gec_dd_predicted_logsoftmax = gec_dd_predicted_logsoftmax.squeeze(1)
				gec_dd_predicted_softmax = torch.exp(gec_dd_predicted_logsoftmax)
				# -------------------------------------------------------
				# ptr network - only used for dd, no beam search needed
				if self.ptr_net == 'comb':
					# import pdb; pdb.set_trace()
					attn_src_softmax = torch.FloatTensor([10**(-10)]).repeat(
						batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(batch_size))\
						.repeat(max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = gec_src.view(1,-1)
					probs = gec_dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					p_gen = p_gen.squeeze(1).view(batch_size, 1) # [b, 1]
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					predicted_softmax_comb = p_gen * gec_dd_predicted_softmax + (1-p_gen) * attn_src_softmax
					gec_dd_step_output = torch.log(predicted_softmax_comb)

				elif self.ptr_net == 'pure':
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(batch_size))\
						.repeat(max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = gec_src.view(1,-1)
					probs = gec_dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					gec_dd_step_output = torch.log(attn_src_softmax)

				elif self.ptr_net == 'null':
					gec_dd_step_output = gec_dd_predicted_logsoftmax
				else:
					assert False, 'Not implemented: ptr_net mode - {}'.format(self.ptr_net)
				# --------------------------------------------------------

				gec_dd_symbols, gec_dd_ret_dict, gec_dd_decoder_outputs, \
					gec_dd_sequence_symbols, gec_dd_lengths = \
						self.decode(gec_dd_idx, gec_dd_step_output, gec_dd_step_attn,
							gec_dd_ret_dict, gec_dd_decoder_outputs,
							gec_dd_sequence_symbols, gec_dd_lengths)
				gec_dd_prev_c = gec_dd_c_out
				gec_dd_tgt_chunk = self.embedder_dec(gec_dd_symbols)
				if self.shared_embed == 'context':
					gec_dd_embedding.append(gec_dd_att_outputs)
				elif self.shared_embed == 'state':
					gec_dd_embedding.append(gec_dd_cell_value)
				elif self.shared_embed == 'state_tgt':
					state_tgt = torch.cat([gec_dd_tgt_chunk, gec_dd_cell_value], -1)
					state_tgt = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
					gec_dd_embedding.append(state_tgt)
				else:
					assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)

			dummy = torch.FloatTensor([0]).repeat(
				batch_size, 1, gec_dd_embedding[-1].size(-1)).to(device=device)
			gec_dd_embedding.append(dummy) #embedding
			gec_dd_sequence_symbols.append(
				torch.LongTensor([PAD] * batch_size).to(device=device).unsqueeze(1))
			gec_dd_ret_dict[KEY_SEQUENCE] = gec_dd_sequence_symbols
			gec_dd_ret_dict[KEY_LENGTH] = gec_dd_lengths.tolist()

		else:

			# classification
			gec_dd_dec_hidden = None
			gec_dd_decoder_outputs = None
			gec_dd_probs = self.dd_classify(gec_dd_enc_outputs) # b * max_len * 2
			gec_dd_ret_dict[CLASSIFY_PROB] = gec_dd_probs

			# keep fluent only
			# b * max_len * 1 : 1=O, 0=E
			gec_dd_labels = gec_dd_probs.ge(0.5).long().view(batch_size, max_seq_len)
			dummy = torch.autograd.Variable(torch.LongTensor(
				max_seq_len).fill_(31), requires_grad=False)
			fluent_idx = [(gec_dd_labels[i,:] == 1).nonzero().view(-1)
				for i in range(batch_size)] # b * [num_fluent]
			fluent_idx.append(dummy)
			gather_col_idx = torch.nn.utils.rnn.pad_sequence(
				fluent_idx, batch_first=True, padding_value=31).long()[:-1,:]
			gec_fluent_symbols = torch.gather(gec_src, 1, gather_col_idx) # b * max_len

			# gec_fluent_embeddings = \
			# 	gec_dd_emb_src[torch.arange(gec_dd_emb_src.shape[0])\
			# 		.unsqueeze(-1), gather_col_idx] # b * max_len * embedding_size
			gec_fluent_embeddings = \
				gec_dd_enc_outputs[torch.arange(gec_dd_enc_outputs.shape[0])\
					.unsqueeze(-1), gather_col_idx] # b * max_len * (hidden_size_enc * 2)

			# record into dict
			gec_dd_ret_dict[KEY_SEQUENCE] = [
				torch.LongTensor(elem).to(device=device) for elem \
				in torch.transpose(gec_fluent_symbols, 0, 1).tolist()][:-1] # max_len - 1 * b

		# ****************************************************************
		# ***********************************************[GEC src-GEC dec]

		# ******************************************************
		# 0. init var

		# final output from gec-decoding output
		gec_ret_dict = dict()
		gec_ret_dict[KEY_ATTN_SCORE] = []
		gec_decoder_outputs = []
		gec_sequence_symbols = []
		gec_lengths = np.array([self.max_seq_len] * batch_size)

		# 1. convert intermediate result from gec_dd to gec input
		if not self.dd_classifier:
			# a. gec_dd_embedding from list to tensor
			gec_enc_outputs = torch.cat(gec_dd_embedding, dim=1).to(device=device)
			# b. gec_dd_sequence_symbols from list to tensor; then get mask
			gec_dd_res = torch.cat(gec_dd_sequence_symbols, dim=1).to(device=device)
			gec_mask_src = gec_dd_res.data.eq(PAD)

		else:
			gec_enc_outputs = gec_fluent_embeddings
			gec_dd_res = gec_fluent_symbols
			gec_mask_src = gec_dd_res.data.eq(PAD)

		# 1.5. different connetion type - embedding or word
		# import pdb; pdb.set_trace()
		self.check_classvar('connect_type')
		if not self.dd_classifier:
			if self.connect_type == 'embed':
				pass
			elif 'word' in self.connect_type:
				if self.connect_type == 'wordhard':
					hard = True
				elif self.connect_type == 'wordsoft':
					hard = False
				dummy = torch.FloatTensor([1e-40]).repeat(
					batch_size, gec_dd_decoder_outputs[-1].size(-1)).to(device=device) # b x z
				dummy[:,0] = .99
				gec_dd_decoder_outputs.append(torch.log(dummy))
				logits = torch.stack(gec_dd_decoder_outputs, dim=1) # b x l x  z
				samples = F.gumbel_softmax(logits, tau=1, hard=hard)
				gec_enc_outputs = torch.matmul(samples, self.embedder_dec.weight)
			else:
				assert False, 'connect_type not implemented'
		else:
			if self.connect_type == 'embed':
				gec_fluent_embeddings = gec_dd_enc_outputs[torch.arange(
					gec_dd_enc_outputs.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * (hidden_size_enc * 2)
				gec_enc_outputs = gec_fluent_embeddings
			if 'word' in self.connect_type:
				gec_enc_outputs = gec_dd_emb_src[torch.arange(
					gec_dd_emb_src.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * embedding_size
			elif self.connect_type == 'prob':
				# using the same length sequence as src
				gec_enc_outputs = gec_dd_probs * gec_dd_emb_src # b * max_len * embedding_size
				gec_mask_src = gec_dd_mask_src
			else:
				assert False, 'connect_type not implemented'

		# 2. run pre-attention dec unilstm [in traditional nmt, part of enc]
		# bilstm

		if self.gec_num_bilstm_dec != 0:
			gec_enc_hidden_init = None
			gec_enc_outputs, gec_enc_hidden = self.gec_dec_bilstm(
				gec_enc_outputs, gec_enc_hidden_init)
			gec_enc_outputs = self.dropout(gec_enc_outputs)\
				.view(batch_size, max_seq_len, gec_enc_outputs.size(-1))

		# unilstm
		if self.gec_num_unilstm_dec_preatt != 0:
			if not self.residual:
				gec_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				gec_enc_outputs, gec_enc_hidden_uni = self.gec_dec_preatt(
					gec_enc_outputs, gec_enc_hidden_uni_init)
				gec_enc_outputs = self.dropout(gec_enc_outputs)\
					.view(batch_size, max_seq_len, gec_enc_outputs.size(-1))
			else:
				gec_enc_hidden_uni_init = None
				gec_enc_hidden_uni_lis = []
				for i in range(self.gec_num_unilstm_dec_preatt):
					gec_enc_inputs = gec_enc_outputs
					gec_dec_preatt_func = getattr(self.gec_dec_preatt, 'l'+str(i))
					gec_enc_outputs, gec_enc_hidden_uni = gec_dec_preatt_func(
						gec_enc_inputs, gec_enc_hidden_uni_init)
					gec_enc_hidden_uni_lis.append(gec_enc_hidden_uni)
					if i < self.gec_num_unilstm_dec_preatt - 1: # no residual for last layer
						gec_enc_outputs = gec_enc_outputs + gec_enc_inputs
					gec_enc_outputs = self.dropout(gec_enc_outputs)\
						.view(batch_size, max_seq_len, gec_enc_outputs.size(-1))

		# ******************************************************
		# 2.5 att inputs: keys n values
		gec_att_keys = gec_enc_outputs
		gec_att_vals = gec_enc_outputs
		# print(gec_att_keys.size())

		# ******************************************************
		# 3. init hidden states - TODO
		gec_dec_hidden = None

		# ******************************************************
		# 4. run gec dec: att + output
		"""
			teacher_forcing_ratio = 1.0 -> always teacher forcing

			E.g.:
				emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
				tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
				predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
				(shift-by-1)
		"""
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		if not is_training:
			 use_teacher_forcing = False

		# beam search decoding
		if not is_training and self.beam_width > 1:
			gec_decoder_outputs, gec_decoder_hidden, gec_metadata = \
					self.beam_search_decoding(gec_att_keys, gec_att_vals, self.gec_dec_pstatt,
					self.gec_num_unilstm_dec_pstatt, self.gec_attn, self.gec_out,
					gec_dec_hidden, gec_mask_src, beam_width=self.beam_width)

			return gec_decoder_outputs, gec_decoder_hidden, gec_metadata

		# no beam search decoding
		gec_tgt_chunk = gec_emb_tgt[:, 0].unsqueeze(1) # BOS
		gec_prev_c = torch.FloatTensor([0]).repeat(batch_size, 1, max_seq_len).to(device=device)
		gec_cell_value = torch.FloatTensor([0]).repeat(batch_size, 1, self.state_size).to(device=device)

		for gec_idx in range(max_seq_len - 1):

			gec_predicted_logsoftmax, gec_dec_hidden, gec_step_attn, \
				gec_c_out, gec_att_outputs, gec_cell_value, _ = \
					self.forward_step(gec_att_keys, gec_att_vals, gec_tgt_chunk, gec_cell_value,
						self.gec_dec_pstatt, self.gec_num_unilstm_dec_pstatt,
						self.gec_att, self.gec_ffn, self.gec_out,
						gec_dec_hidden, gec_mask_src, gec_prev_c)
			gec_step_output = gec_predicted_logsoftmax.squeeze(1)
			gec_symbols, gec_ret_dict, gec_decoder_outputs, gec_sequence_symbols, gec_lengths = \
						self.decode(gec_idx, gec_step_output, gec_step_attn,
								gec_ret_dict, gec_decoder_outputs, gec_sequence_symbols, gec_lengths)
			gec_prev_c = gec_c_out
			if use_teacher_forcing:
				gec_tgt_chunk = gec_emb_tgt[:, gec_idx+1].unsqueeze(1)
			else:
				gec_tgt_chunk = self.embedder_dec(gec_symbols)
			# print('target query size: {}'.format(tgt_chunk.size()))

		# print('gec...')
		gec_ret_dict[KEY_SEQUENCE] = gec_sequence_symbols
		gec_ret_dict[KEY_LENGTH] = gec_lengths.tolist()

		return gec_dd_decoder_outputs, gec_dd_dec_hidden, gec_dd_ret_dict, \
					gec_decoder_outputs, gec_dec_hidden, gec_ret_dict


	# =====================================================================================
	# Evaluation - separate dd / gec
	"""
		cropped out from forward
		and modified to do evaluation
	"""

	def dd_eval(self, dd_src, dd_tgt,
		hidden=None, is_training=False,
		dd_att_key_feats=None,
		dd_att_scores=None,
		dd_eval_use_teacher_forcing=False,
		beam_width=1):

		"""
			Args:
				src:
					list of src word_ids [batch_size, max_seq_len, word_ids]
				tgt:
					list of tgt word_ids
				hidden:
					initial hidden state
					not used (all hidden initialied using None)
				is_training:
					whether in eval or train mode

			Returns:
				decoder_outputs:
					list of step_output
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				shared_hidden:
				ret_dict:

			Add:
				decoding only do greedy search over the input vocab - DD only deletes
		"""

		# config device
		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		if hasattr(self, 'dd_att'):
			self.dd_att.use_gpu = self.use_gpu

		# init global var
		# ******************************************************
		batch_size = dd_src.size(0)
		max_seq_len = dd_src.size(1)
		self.beam_width = beam_width

		# 0. init var
		dd_ret_dict = dict()
		dd_ret_dict[KEY_ATTN_SCORE] = []
		dd_ret_dict[KEY_DIS] = []
		dd_ret_dict[KEY_CONTEXT] = []
		dd_decoder_outputs = []
		dd_sequence_symbols = []
		dd_lengths = np.array([max_seq_len] * batch_size)

		# 1. convert id to embedding
		dd_emb_src = self.embedding_dropout(self.embedder_enc(dd_src))
		if type(dd_tgt) == type(None):
			dd_tgt = torch.Tensor([BOS]).repeat(dd_src.size()).type(
				torch.LongTensor).to(device=device)
		dd_emb_tgt = self.embedding_dropout(self.embedder_dec(dd_tgt))
		dd_mask_src = dd_src.data.eq(PAD)

		# ******************************************************
		# 2. run enc
		dd_enc_hidden_init = None
		dd_enc_outputs, dd_enc_hidden = self.enc(dd_emb_src, dd_enc_hidden_init)
		dd_enc_outputs = self.dropout(dd_enc_outputs)\
						.view(batch_size, max_seq_len, dd_enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				dd_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				dd_enc_outputs, dd_enc_hidden_uni = self.enc_uni(
					dd_enc_outputs, dd_enc_hidden_uni_init)
				dd_enc_outputs = self.dropout(dd_enc_outputs)\
					.view(batch_size, max_seq_len, dd_enc_outputs.size(-1))
			else:
				dd_enc_hidden_uni_init = None
				dd_enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					dd_enc_inputs = dd_enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					dd_enc_outputs, dd_enc_hidden_uni = enc_func(
						dd_enc_inputs, dd_enc_hidden_uni_init)
					dd_enc_hidden_uni_lis.append(dd_enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						dd_enc_outputs = dd_enc_outputs + dd_enc_inputs
					dd_enc_outputs = self.dropout(dd_enc_outputs)\
						.view(batch_size, max_seq_len, dd_enc_outputs.size(-1))

		# ******************************************************
		# ******************* DD DEC or CLASSIFY ***************
		if not self.dd_classifier:

			# ******************************************************
			# 2.5 att inputs: keys n values
			if type(dd_att_key_feats) == type(None):
				if self.dd_additional_key_size == 0:
					dd_att_keys = dd_enc_outputs
				else:
					dummy_feats = torch.autograd.Variable(
						torch.FloatTensor(dd_enc_outputs.size()[0],
						dd_enc_outputs.size()[1], self.dd_additional_key_size).fill_(0.0),
						requires_grad=False).to(device)
					dd_att_keys = torch.cat((dd_enc_outputs, dummy_feats), dim=2)
			else:
				# dd_att_key_feats: b x max_seq_len x additional_key_size
				assert self.dd_additional_key_size == dd_att_key_feats.size(-1), \
					'Mismatch in attention key dimension!'
				dd_att_keys = torch.cat((dd_enc_outputs, dd_att_key_feats), dim=2)
			dd_att_vals = dd_enc_outputs

			# ******************************************************
			# 3. init hidden states
			dd_dec_hidden = None

			# ******************************************************
			# 4. run dd dec: att + output

			# beam search decoding in DD-dec
			if not is_training and self.beam_width > 1:
				dd_decoder_outputs, dd_decoder_hidden, dd_metadata = \
					self.beam_search_decoding(dd_att_keys, dd_att_vals,
						self.dd_dec, self.dd_num_unilstm_dec,
						self.dd_att, self.dd_ffn, self.dd_out,
						dd_dec_hidden, dd_mask_src, beam_width=self.beam_width)
				return dd_decoder_outputs, dd_decoder_hidden, dd_metadata

			# no beam search decoding
			dd_tgt_chunk = dd_emb_tgt[:, 0].unsqueeze(1) # BOS
			dd_prev_c = torch.FloatTensor([0]).repeat(
				batch_size, 1, max_seq_len).to(device=device)
			dd_cell_value = torch.FloatTensor([0]).repeat(
				batch_size, 1, self.state_size).to(device=device)

			for dd_idx in range(max_seq_len - 1):

				if type(dd_att_scores) != type(None):
					step_attn_ref_detach = dd_att_scores[:, dd_idx,:].unsqueeze(1)
					step_attn_ref_detach = step_attn_ref_detach.type(torch.FloatTensor).to(device=device)
				else:
					step_attn_ref_detach = None

				dd_predicted_logsoftmax, dd_dec_hidden, dd_step_attn, \
					dd_c_out, dd_att_outputs, dd_cell_value, p_gen = \
						self.forward_step(dd_att_keys, dd_att_vals,
							dd_tgt_chunk, dd_cell_value,
							self.dd_dec, self.dd_num_unilstm_dec,
							self.dd_att, self.dd_ffn, self.dd_out,
							dd_dec_hidden, dd_mask_src, dd_prev_c, step_attn_ref_detach)

				dd_predicted_logsoftmax = dd_predicted_logsoftmax.squeeze(1)
				dd_predicted_softmax = torch.exp(dd_predicted_logsoftmax)
				# -------------------------------------------------------
				# ptr network - only used for dd, no beam search needed
				if self.ptr_net == 'comb':
					# import pdb; pdb.set_trace()
					attn_src_softmax = torch.FloatTensor([10**(-10)]).repeat(
						batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(batch_size))\
						.repeat(max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..b-1b-1..
					yidices = dd_src.view(1,-1)
					probs = dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					p_gen = p_gen.squeeze(1).view(batch_size, 1) # [b, 1]
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					predicted_softmax_comb = p_gen * dd_predicted_softmax + (1-p_gen) * attn_src_softmax
					dd_step_output = torch.log(predicted_softmax_comb)

				elif self.ptr_net == 'pure':
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(batch_size))\
						.repeat(max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = dd_src.view(1,-1)
					probs = dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001)
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					dd_step_output = torch.log(attn_src_softmax)

				elif self.ptr_net == 'null':
					dd_step_output = dd_predicted_logsoftmax
				else:
					assert False, 'Not implemented: ptr_net mode - {}'.format(self.ptr_net)
				# --------------------------------------------------------
				dd_symbols, dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths = \
							self.decode(dd_idx, dd_step_output, dd_step_attn,
									dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths)
							# self.decode_dd(dd_idx, dd_step_output, dd_step_attn, dd_src,
							# 		dd_ret_dict, dd_decoder_outputs, dd_sequence_symbols, dd_lengths)
				dd_prev_c = dd_c_out
				if dd_eval_use_teacher_forcing:
					dd_tgt_chunk = dd_emb_tgt[:, dd_idx+1].unsqueeze(1)
				else:
					dd_tgt_chunk = self.embedder_dec(dd_symbols)

				# record discriminator output - for debugging
				if self.add_discriminator:
					if self.shared_embed == 'context':
						dis_input = dd_att_outputs
					elif self.shared_embed == 'state':
						dis_input = dd_cell_value
					elif self.shared_embed == 'state_tgt':
						state_tgt = torch.cat([dd_tgt_chunk, dd_cell_value], -1)
						dis_input = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
					else:
						assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)
					dd_ret_dict[KEY_DIS].append(self.discriminator(dis_input))

				# record shared embedding - for debugging
				dd_ret_dict[KEY_CONTEXT].append(dd_att_outputs.squeeze())

			dd_ret_dict[KEY_SEQUENCE] = dd_sequence_symbols
			dd_ret_dict[KEY_LENGTH] = dd_lengths.tolist()

		else: # do simple classification for dd

			# classification
			dd_dec_hidden = None
			dd_decoder_outputs = None
			dd_probs = self.dd_classify(dd_enc_outputs) # b * max_len * 1
			dd_ret_dict[CLASSIFY_PROB] = dd_probs

			# keep fluent only
			dd_labels = dd_probs.ge(0.5).long()\
				.view(batch_size, max_seq_len) # b * max_len * 1 : 1=O, 0=E
			dummy = torch.autograd.Variable(torch.LongTensor(max_seq_len).fill_(31),
				requires_grad=False)
			fluent_idx = [(dd_labels[i,:] == 1).nonzero().view(-1)
				for i in range(batch_size)] # b * [num_fluent]
			fluent_idx.append(dummy)
			gather_col_idx = torch.nn.utils.rnn.pad_sequence(
				fluent_idx, batch_first=True, padding_value=31).long()[:-1,:]
			fluent_symbols = torch.gather(dd_src, 1, gather_col_idx) # b * max_len

			# record into dict
			dd_ret_dict[KEY_SEQUENCE] = [torch.LongTensor(elem).to(device=device) for elem \
				in torch.transpose(fluent_symbols, 0, 1).tolist()][:-1] # max_len - 1 * b

		return dd_decoder_outputs, dd_dec_hidden, dd_ret_dict


	def gec_eval(self, gec_src, gec_tgt,
		hidden=None, is_training=False,
		gec_dd_att_key_feats=None,
		gec_dd_att_scores=None,
		gec_dd_tgt=None,
		gec_dd_eval_use_teacher_forcing=False,
		gec_eval_use_teacher_forcing=False,
		beam_width=1):

		"""
			Args:
				src:
					list of src word_ids [batch_size, max_seq_len, word_ids]
				tgt:
					list of tgt word_ids
				hidden:
					initial hidden state
					not used (all hidden initialied using None)
				is_training:
					whether in eval or train mode

			Returns:
				decoder_outputs:
					list of step_output
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				shared_hidden:
				ret_dict:
		"""

		# config device
		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		if hasattr(self, 'dd_att'):
			self.dd_att.use_gpu = self.use_gpu
		self.gec_att.use_gpu = self.use_gpu

		# init global var
		# ******************************************************
		batch_size = gec_src.size(0)
		max_seq_len = gec_src.size(1)
		self.beam_width = beam_width

		# **************************************************************
		# **********************************************[GEC src-DD dec]

		# ******************************************************
		# 0. init var

		# sanity check
		if gec_dd_eval_use_teacher_forcing:
			assert type(gec_dd_tgt) != type(None), \
				'dd_tgt not given - cannot run in dd teacher forcing'
		if gec_eval_use_teacher_forcing:
			assert type(gec_tgt) != type(None), \
				'gec_tgt is None - cannot run in gec teacher forcing'

		# intermediate output from dd-decoding
		gec_dd_ret_dict = dict()
		gec_dd_ret_dict[KEY_ATTN_SCORE] = []
		gec_dd_ret_dict[KEY_DIS] = []
		gec_dd_ret_dict[KEY_CONTEXT] = []
		gec_dd_decoder_outputs = []
		gec_dd_sequence_symbols = []
		gec_dd_lengths = np.array([max_seq_len] * batch_size)
		gec_dd_embedding = []

		# 1. convert id to embedding
		gec_dd_emb_src = self.embedding_dropout(self.embedder_enc(gec_src))
		gec_dd_mask_src = gec_src.data.eq(PAD)
		if type(gec_tgt) == type(None):
			gec_tgt = torch.Tensor([BOS]).repeat(
				gec_src.size()).type(torch.LongTensor).to(device=device)
		gec_emb_tgt = self.embedding_dropout(self.embedder_dec(gec_tgt))
		if type(gec_dd_tgt) != type(None):
			gec_dd_emb_tgt = self.embedding_dropout(self.embedder_dec(gec_dd_tgt))

		# ******************************************************
		# 2. run enc
		gec_dd_enc_hidden_init = None
		gec_dd_enc_outputs, gec_dd_enc_hidden = self.enc(gec_dd_emb_src, gec_dd_enc_hidden_init)
		gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
			.view(batch_size, max_seq_len, gec_dd_enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				gec_dd_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				gec_dd_enc_outputs, gec_enc_hidden_uni = self.enc_uni(
					gec_dd_enc_outputs, gec_dd_enc_hidden_uni_init)
				gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
					.view(batch_size, max_seq_len, gec_dd_enc_outputs.size(-1))
			else:
				gec_dd_enc_hidden_uni_init = None
				gec_dd_enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					gec_dd_enc_inputs = gec_dd_enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					gec_dd_enc_outputs, gec_dd_enc_hidden_uni = enc_func(
						gec_dd_enc_inputs, gec_dd_enc_hidden_uni_init)
					gec_dd_enc_hidden_uni_lis.append(gec_dd_enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						gec_dd_enc_outputs = gec_dd_enc_outputs + gec_dd_enc_inputs
					gec_dd_enc_outputs = self.dropout(gec_dd_enc_outputs)\
						.view(batch_size, max_seq_len, gec_dd_enc_outputs.size(-1))

		# ******************************************************
		# ******************* DD DEC or CLASSIFY ***************
		if not self.dd_classifier:

			# ******************************************************
			# 2.5 att inputs: keys n values
			if type(gec_dd_att_key_feats) == type(None):
				if self.dd_additional_key_size == 0:
					gec_dd_att_keys = gec_dd_enc_outputs
				else:
					# handle gec set no dd ref att probs
					dummy_feats = torch.autograd.Variable(
						torch.FloatTensor(gec_dd_enc_outputs.size()[0],
						gec_dd_enc_outputs.size()[1], self.dd_additional_key_size).fill_(0.0),
						requires_grad=False).to(device)
					gec_dd_att_keys = torch.cat((gec_dd_enc_outputs, dummy_feats), dim=2)
			else:
				# gec_dd_att_key_feats: b x max_seq_len x additional_key_size
				assert self.dd_additional_key_size == gec_dd_att_key_feats.size(-1), \
					'Mismatch in attention key dimension!'
				gec_dd_att_keys = torch.cat((gec_dd_enc_outputs, gec_dd_att_key_feats), dim=2)
			gec_dd_att_vals = gec_dd_enc_outputs

			# ******************************************************
			# 3. init hidden states
			gec_dd_dec_hidden = None

			# ******************************************************
			# 4. run dd dec: att + output

			# no beam search decoding + no teacher forcing
			gec_dd_tgt_chunk = gec_emb_tgt[:, 0].unsqueeze(1) # BOS (okay to use 1st symbol = BOS)
			gec_dd_prev_c = torch.FloatTensor([0])\
				.repeat(batch_size, 1, max_seq_len).to(device=device)
			gec_dd_cell_value = torch.FloatTensor([0])\
				.repeat(batch_size, 1, self.state_size).to(device=device)

			for gec_dd_idx in range(max_seq_len - 1):

				if type(gec_dd_att_scores) != type(None):
					step_attn_ref_detach = gec_dd_att_scores[:, gec_dd_idx,:].unsqueeze(1)
					step_attn_ref_detach = step_attn_ref_detach.type(torch.FloatTensor).to(device=device)
				else:
					step_attn_ref_detach = None

				gec_dd_predicted_logsoftmax, gec_dd_dec_hidden, \
					gec_dd_step_attn, gec_dd_c_out, gec_dd_att_outputs, gec_dd_cell_value, p_gen = \
						self.forward_step(gec_dd_att_keys, gec_dd_att_vals,
							gec_dd_tgt_chunk, gec_dd_cell_value,
							self.dd_dec, self.dd_num_unilstm_dec,
							self.dd_att, self.dd_ffn, self.dd_out,
							gec_dd_dec_hidden, gec_dd_mask_src, gec_dd_prev_c, step_attn_ref_detach)

				gec_dd_predicted_logsoftmax = gec_dd_predicted_logsoftmax.squeeze(1)
				gec_dd_predicted_softmax = torch.exp(gec_dd_predicted_logsoftmax)
				# -------------------------------------------------------
				# ptr network - only used for dd, no beam search needed
				if self.ptr_net == 'comb':
					# import pdb; pdb.set_trace()
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(batch_size))\
						.repeat(max_seq_len,1).transpose(0,1).contiguous()\
						.view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = gec_src.view(1,-1)
					probs = gec_dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001) # handle 0s
					p_gen = p_gen.squeeze(1).view(batch_size, 1) # [b, 1]
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					predicted_softmax_comb = p_gen * gec_dd_predicted_softmax + (1-p_gen) * attn_src_softmax
					gec_dd_step_output = torch.log(predicted_softmax_comb)

				elif self.ptr_net == 'pure':
					attn_src_softmax = torch.FloatTensor([10**(-10)])\
						.repeat(batch_size, self.vocab_size).to(device=device)
					xidices = torch.LongTensor(torch.arange(batch_size))\
						.repeat(max_seq_len,1).transpose(0,1)\
						.contiguous().view(1,-1).to(device=device) # 00..11..22..bb..
					yidices = gec_src.view(1,-1)
					probs = gec_dd_step_attn.view(1,-1)
					attn_src_softmax.index_put_([xidices, yidices], probs, accumulate=True)
					attn_src_softmax = torch.clamp(attn_src_softmax, 1e-40, 1.00001) # handle 0s
					attn_src_softmax = attn_src_softmax.view(batch_size, -1)
					gec_dd_step_output = torch.log(attn_src_softmax)

				elif self.ptr_net == 'null':
					gec_dd_step_output = gec_dd_predicted_logsoftmax
				else:
					assert False, 'Not implemented: ptr_net mode - {}'.format(self.ptr_net)
				# --------------------------------------------------------

				gec_dd_symbols, gec_dd_ret_dict, gec_dd_decoder_outputs,\
					gec_dd_sequence_symbols, gec_dd_lengths = \
						self.decode(gec_dd_idx, gec_dd_step_output, gec_dd_step_attn,
							gec_dd_ret_dict, gec_dd_decoder_outputs,
							gec_dd_sequence_symbols, gec_dd_lengths)
					# self.decode_dd(gec_dd_idx, gec_dd_step_output, gec_dd_step_attn, gec_src,
					# 		gec_dd_ret_dict, gec_dd_decoder_outputs, gec_dd_sequence_symbols, gec_dd_lengths)
				gec_dd_prev_c = gec_dd_c_out
				if gec_dd_eval_use_teacher_forcing:
					gec_dd_tgt_chunk = gec_dd_emb_tgt[:, gec_dd_idx+1].unsqueeze(1)
				else:
					gec_dd_tgt_chunk = self.embedder_dec(gec_dd_symbols)

				if self.shared_embed == 'context':
					gec_dd_embedding.append(gec_dd_att_outputs)
				elif self.shared_embed == 'state':
					gec_dd_embedding.append(gec_dd_cell_value)
				elif self.shared_embed == 'state_tgt':
					state_tgt = torch.cat([gec_dd_tgt_chunk, gec_dd_cell_value], -1)
					state_tgt = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
					gec_dd_embedding.append(state_tgt)
				else:
					assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)

				# discriminator
				if self.add_discriminator:
					if self.shared_embed == 'context':
						dis_input = gec_dd_att_outputs
					elif self.shared_embed == 'state':
						dis_input = gec_dd_cell_value
					elif self.shared_embed == 'state_tgt':
						state_tgt = torch.cat([gec_dd_tgt_chunk, gec_dd_cell_value], -1)
						dis_input = state_tgt.view(-1, 1, self.embedding_size + self.state_size)
						# print('embed', state_tgt.size())
					else:
						assert False, 'Unknown shared_embed type - {}'.format(self.shared_embed)
					gec_dd_ret_dict[KEY_DIS].append(self.discriminator(dis_input))

				# record shared embedding - for debugging
				gec_dd_ret_dict[KEY_CONTEXT].append(gec_dd_att_outputs.squeeze())


			dummy = torch.FloatTensor([0]).repeat(
				batch_size, 1, gec_dd_embedding[-1].size(-1)).to(device=device)
			gec_dd_embedding.append(dummy) #embedding
			gec_dd_sequence_symbols.append(
				torch.LongTensor([PAD] * batch_size).to(device=device).unsqueeze(1)) #symbol
			gec_dd_ret_dict[KEY_SEQUENCE] = gec_dd_sequence_symbols
			gec_dd_ret_dict[KEY_LENGTH] = gec_dd_lengths.tolist()

		else:

			# classification
			gec_dd_dec_hidden = None
			gec_dd_decoder_outputs = None
			gec_dd_probs = self.dd_classify(gec_dd_enc_outputs) # b * max_len * 2
			gec_dd_ret_dict[CLASSIFY_PROB] = gec_dd_probs

			# keep fluent only
			gec_dd_labels = gec_dd_probs.ge(0.5).long()\
				.view(batch_size, max_seq_len) # b * max_len * 1 : 1=O, 0=E
			dummy = torch.autograd.Variable(
				torch.LongTensor(max_seq_len).fill_(31), requires_grad=False)
			fluent_idx = [(gec_dd_labels[i,:] == 1).nonzero().view(-1)
			 	for i in range(batch_size)] # b * [num_fluent]
			fluent_idx.append(dummy)
			gather_col_idx = torch.nn.utils.rnn.pad_sequence(
				fluent_idx, batch_first=True, padding_value=31).long()[:-1,:]
			gec_fluent_symbols = torch.gather(gec_src, 1, gather_col_idx) # b * max_len

			# gec_fluent_embeddings = \
			# 	gec_dd_emb_src[torch.arange(
					# gec_dd_emb_src.shape[0]).unsqueeze(-1), gather_col_idx] # b * max_len * embedding_size
			gec_fluent_embeddings = \
				gec_dd_enc_outputs[torch.arange(
					gec_dd_enc_outputs.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * (hidden_size_enc * 2)

			# record into dict
			gec_dd_ret_dict[KEY_SEQUENCE] = [torch.LongTensor(elem).to(device=device) for elem \
				in torch.transpose(gec_fluent_symbols, 0, 1).tolist()][:-1] # max_len - 1 * b

		# return decoder_outputs, dec_hidden, ret_dict

		# **********************************************************
		# *****************************************[GEC src-GEC dec]

		# ******************************************************
		# 0. init var

		# final output from gec-decoding output
		gec_ret_dict = dict()
		gec_ret_dict[KEY_ATTN_SCORE] = []
		gec_decoder_outputs = []
		gec_sequence_symbols = []
		gec_lengths = np.array([max_seq_len] * batch_size)

		# 1. convert intermediate result from gec_dd to gec input
		if not self.dd_classifier:
			# a. gec_dd_embedding from list to tensor
			gec_enc_outputs = torch.cat(gec_dd_embedding, dim=1).to(device=device)
			# print(gec_enc_outputs.size()) # [b x T x h*2]

			# b. gec_dd_sequence_symbols from list to tensor; then get mask
			gec_dd_res = torch.cat(gec_dd_sequence_symbols, dim=1).to(device=device)

			gec_mask_src = gec_dd_res.data.eq(PAD)

		else:
			gec_enc_outputs = gec_fluent_embeddings
			gec_dd_res = gec_fluent_symbols
			gec_mask_src = gec_dd_res.data.eq(PAD)

		# 1.5. different connetion type - embedding or word
		# import pdb; pdb.set_trace()
		self.check_classvar('connect_type')
		if not self.dd_classifier:
			if self.connect_type == 'embed':
				pass
			elif 'word' in self.connect_type:
				if self.connect_type == 'wordhard':
					hard = True
				elif self.connect_type == 'wordsoft':
					hard = False
				dummy = torch.FloatTensor([1e-40]).repeat(
					batch_size, gec_dd_decoder_outputs[-1].size(-1)).to(device=device) # b x z
				dummy[:,0] = .99
				gec_dd_decoder_outputs.append(torch.log(dummy))
				logits = torch.stack(gec_dd_decoder_outputs, dim=1) # b x l x  z
				samples = F.gumbel_softmax(logits, tau=1, hard=hard)
				gec_enc_outputs = torch.matmul(samples, self.embedder_dec.weight)
			else:
				assert False, 'connect_type not implemented'
		else:
			if self.connect_type == 'embed':
				gec_fluent_embeddings = gec_dd_enc_outputs[torch.arange(
					gec_dd_enc_outputs.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * (hidden_size_enc * 2)
				gec_enc_outputs = gec_fluent_embeddings
			if 'word' in self.connect_type:
				gec_enc_outputs = gec_dd_emb_src[torch.arange(
					gec_dd_emb_src.shape[0]).unsqueeze(-1), gather_col_idx]
					# b * max_len * embedding_size
			elif self.connect_type == 'prob':
				# using the same length sequence as src
				gec_enc_outputs = gec_dd_probs * gec_dd_emb_src # b * max_len * embedding_size
				gec_mask_src = gec_dd_mask_src
			else:
				assert False, 'connect_type not implemented'

		# 2. run pre-attention dec unilstm [in traditional nmt, part of enc]
		# bilstm

		if self.gec_num_bilstm_dec != 0:
			gec_enc_hidden_init = None
			gec_enc_outputs, gec_enc_hidden = self.gec_dec_bilstm(
				gec_enc_outputs, gec_enc_hidden_init)
			gec_enc_outputs = self.dropout(gec_enc_outputs)\
				.view(batch_size, max_seq_len, gec_enc_outputs.size(-1))

		# unilstm
		if self.gec_num_unilstm_dec_preatt != 0:
			if not self.residual:
				gec_enc_hidden_uni_init = None
				# enc_hidden_uni_init / enc_hidden_uni: n_layer*n_directions, batch, hidden_size
				gec_enc_outputs, gec_enc_hidden_uni = self.gec_dec_preatt(
					gec_enc_outputs, gec_enc_hidden_uni_init)
				gec_enc_outputs = self.dropout(gec_enc_outputs)\
					.view(batch_size, max_seq_len, gec_enc_outputs.size(-1))
			else:
				gec_enc_hidden_uni_init = None
				gec_enc_hidden_uni_lis = []
				for i in range(self.gec_num_unilstm_dec_preatt):
					gec_enc_inputs = gec_enc_outputs
					gec_dec_preatt_func = getattr(self.gec_dec_preatt, 'l'+str(i))
					gec_enc_outputs, gec_enc_hidden_uni = gec_dec_preatt_func(
						gec_enc_inputs, gec_enc_hidden_uni_init)
					gec_enc_hidden_uni_lis.append(gec_enc_hidden_uni)
					if i < self.gec_num_unilstm_dec_preatt - 1: # no residual for last layer
						gec_enc_outputs = gec_enc_outputs + gec_enc_inputs
					gec_enc_outputs = self.dropout(gec_enc_outputs)\
						.view(batch_size, max_seq_len, gec_enc_outputs.size(-1))

		# ******************************************************
		# 2.5 att inputs: keys n values
		gec_att_keys = gec_enc_outputs
		gec_att_vals = gec_enc_outputs

		# ******************************************************
		# 3. init hidden states
		gec_dec_hidden = None

		# ******************************************************
		# 4. run gec dec: att + output

		# beam search decoding
		if not is_training and self.beam_width > 1:
			gec_decoder_outputs, gec_decoder_hidden, gec_metadata = \
				self.beam_search_decoding(gec_att_keys, gec_att_vals,
					self.gec_dec_pstatt, self.gec_num_unilstm_dec_pstatt,
					self.gec_att, self.gec_ffn, self.gec_out,
					gec_dec_hidden, gec_mask_src, beam_width=self.beam_width)

			return gec_decoder_outputs, gec_decoder_hidden, gec_metadata

		# no beam search decoding
		gec_tgt_chunk = gec_emb_tgt[:, 0].unsqueeze(1) # BOS
		gec_prev_c = torch.FloatTensor([0]).repeat(
			batch_size, 1, max_seq_len).to(device=device)
		gec_cell_value = torch.FloatTensor([0]).repeat(
			batch_size, 1, self.state_size).to(device=device)

		for gec_idx in range(max_seq_len - 1):
			# print(gec_idx)
			gec_predicted_logsoftmax, gec_dec_hidden, gec_step_attn, \
				gec_c_out, gec_att_outputs, gec_cell_value, _ = \
					self.forward_step(gec_att_keys, gec_att_vals, gec_tgt_chunk, gec_cell_value,
							self.gec_dec_pstatt, self.gec_num_unilstm_dec_pstatt,
							self.gec_att, self.gec_ffn, self.gec_out,
							gec_dec_hidden, gec_mask_src, gec_prev_c)
			gec_step_output = gec_predicted_logsoftmax.squeeze(1)
			gec_symbols, gec_ret_dict, gec_decoder_outputs, gec_sequence_symbols, gec_lengths = \
						self.decode(gec_idx, gec_step_output, gec_step_attn,
								gec_ret_dict, gec_decoder_outputs, gec_sequence_symbols, gec_lengths)
			gec_prev_c = gec_c_out
			if gec_eval_use_teacher_forcing:
				gec_tgt_chunk = gec_emb_tgt[:, gec_idx+1].unsqueeze(1)
			else:
				gec_tgt_chunk = self.embedder_dec(gec_symbols)

		gec_ret_dict[KEY_SEQUENCE] = gec_sequence_symbols
		gec_ret_dict[KEY_LENGTH] = gec_lengths.tolist()

		return gec_dd_decoder_outputs, gec_dd_dec_hidden, gec_dd_ret_dict, \
					gec_decoder_outputs, gec_dec_hidden, gec_ret_dict


	# =====================================================================================
	# Sub function used in dd_train / gec_train / dd_eval / gec_eval

	def decode(self, step, step_output, step_attn,
		ret_dict, decoder_outputs, sequence_symbols, lengths):

		# same as in foward
		# used in eval scripts

		ret_dict[KEY_ATTN_SCORE].append(step_attn)
		decoder_outputs.append(step_output)
		symbols = decoder_outputs[-1].topk(1)[1]
		assert sum(symbols.ge(self.vocab_size).long()) == 0, 'out of range symbol {}'\
			.format(torch.masked_select(symbols, symbols.ge(self.vocab_size)))

		# import pdb; pdb.set_trace()
		eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD))
		if eos_batches.dim() > 0:
			eos_batches = eos_batches.cpu().view(-1).numpy()
			update_idx = ((lengths > step) & eos_batches) != 0
			lengths[update_idx] = len(sequence_symbols) + 1
			pad_idx = (lengths < (len(sequence_symbols) + 1))
			symbols_dummy = symbols.cpu().view(-1).numpy()
			symbols_dummy[pad_idx] = PAD
			symbols = torch.from_numpy(symbols_dummy).view(-1,1).to(device)

		sequence_symbols.append(symbols)

		return symbols, ret_dict, decoder_outputs, sequence_symbols, lengths


	def decode_dd(self, step, step_output, step_attn, src,
		ret_dict, decoder_outputs, sequence_symbols, lengths):

		# same as in foward
		# used in eval scripts
		# only decode over the input vocab

		ret_dict[KEY_ATTN_SCORE].append(step_attn)
		decoder_outputs.append(step_output)
		symbols = decoder_outputs[-1].topk(1)[1]

		src_detach = src.clone().detach().contiguous()
		src_detach[src==EOS] = 0

		# eos prob
		eos_prob = step_output[:, EOS].view(-1,1)


		PROB_BIAS = 5
		output_trim = torch.gather(step_output, 1, src_detach)
		val, ind = output_trim.topk(1)
		choice_eos = (eos_prob > val + (PROB_BIAS)).type('torch.LongTensor').to(device=device)
		choice_ind = (eos_prob <= (val + PROB_BIAS)).type('torch.LongTensor').to(device=device)

		symbols_trim = torch.gather(src_detach, 1, ind)
		symbols_choice = symbols_trim * choice_ind + EOS * choice_eos

		sequence_symbols.append(symbols_choice)

		eos_batches = torch.max(symbols_choice.eq(EOS), symbols_choice.data.eq(PAD))
		if eos_batches.dim() > 0:
			eos_batches = eos_batches.cpu().view(-1).numpy()
			update_idx = ((lengths > step) & eos_batches) != 0
			lengths[update_idx] = len(sequence_symbols)

		return symbols_choice, ret_dict, decoder_outputs, sequence_symbols, lengths


def get_base_hidden(hidden):

	""" strip the nested tuple, get the last hidden state """

	tuple_dim = []
	while isinstance(hidden, tuple):
		tuple_dim.append(len(hidden))
		hidden = hidden[-1]
	return hidden, tuple_dim


def _inflate(tensor, times, dim):

	"""
		Given a tensor, 'inflates' it along the given dimension
		by replicating each slice specified number of times (in-place)
		Args:
			tensor: A :class:`Tensor` to inflate
			times: number of repetitions
			dim: axis for inflation (default=0)
		Returns:
			A :class:`Tensor`
		Examples::
			>> a = torch.LongTensor([[1, 2], [3, 4]])
			>> a
			1   2
			3   4
			[torch.LongTensor of size 2x2]
			>> b = ._inflate(a, 2, dim=1)
			>> b
			1   2   1   2
			3   4   3   4
			[torch.LongTensor of size 2x4]
			>> c = _inflate(a, 2, dim=0)
			>> c
			1   2
			3   4
			1   2
			3   4
			[torch.LongTensor of size 4x2]
	"""

	repeat_dims = [1] * tensor.dim()
	repeat_dims[dim] = times
	return tensor.repeat(*repeat_dims)


def inflat_hidden_state(hidden_state, k):

	if hidden_state is None:
		hidden = None
	else:
		if isinstance(hidden_state, tuple):
			hidden = tuple([_inflate(h, k, 1) for h in hidden_state])
		else:
			hidden = _inflate(hidden_state, k, 1)
	return hidden
