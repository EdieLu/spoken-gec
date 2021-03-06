import torch
import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np
import warnings

from utils.misc import check_srctgt, get_memory_alloc, convert_dd_att_ref
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor
from utils.dataset import Dataset
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import Seq2Seq

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)

class Trainer(object):

	"""
		--- train_mode ---
		1. train_mode:
				end2end: load ddgec set
				separate: load dd set & gec set
		2. ddreg:
				False
        note: att_loss; attcls_loss are removed
	"""

	def __init__(self, expt_dir='experiment',
		load_dir=None,
		restart_dir=None,
		batch_size=64,
		random_seed=None,
		checkpoint_every=100,
		print_every=100,
		use_gpu=False,
		ddreg=False,
		learning_rate=0.001,
		max_grad_norm=1.0,
		loss_shift=False,
		max_count_no_improve=5,
		max_count_num_rollback=5,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=1.0,
		train_mode='separate',
		gec_acc_weight=1.0,
		gec_loss_weight=1.0,
		dd_loss_weight=1.0,
		ddatt_loss_weight=1.0,
		ddattcls_loss_weight=1.0,
		att_scale_up=1.0,
		save_schedule='roll_back'):

		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		print('device: {}'.format(device))

		# self.loss = loss
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.use_gpu = use_gpu
		self.ddreg = ddreg
		self.learning_rate = learning_rate
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.loss_shift = loss_shift
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.max_count_no_improve = max_count_no_improve
		self.max_count_num_rollback = max_count_num_rollback
		self.train_mode = train_mode

		# hyper params
		self.gec_acc_weight = gec_acc_weight # used in deciding saving
		self.gec_loss_weight = gec_loss_weight # used for sgd
		self.dd_loss_weight = dd_loss_weight
		self.ddatt_loss_weight = ddatt_loss_weight
		self.ddattcls_loss_weight = ddattcls_loss_weight
		self.att_scale_up = att_scale_up
		self.save_schedule = save_schedule

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir
		self.restart_dir = restart_dir

		self.batch_size = batch_size
		self.logger = logging.getLogger(__name__)
		self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)

	# ============ eval ===============
	def _evaluate_batches(self, model, batches, dataset, mode='dd'):

		model.eval()

		loss = NLLLoss()
		dsfclassify_loss = BCELoss()

		loss.reset()
		dsfclassify_loss.reset()

		match = 0
		total = 0

		tgt_id2word = dataset.tgt_id2word # same for dd/gec

		# use gec for evaluation main criteria
		time_1 = time.time()
		out_count = 0
		with torch.no_grad():
			for b_idx in range(len(batches)):

				# if b_idx > 5:
				# 	break
				# import pdb; pdb.set_trace()

				batch = batches[b_idx]

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				flt_ids = batch['flt_word_ids']
				flt_lengths = batch['flt_sentence_lengths']

				src_probs = None
				if 'src_ddfd_probs' in batch and model.dd_additional_key_size > 0:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				src_labs = None
				if 'src_ddfd_labs' in batch:
					src_labs =  batch['src_ddfd_labs']
					src_labs = _convert_to_tensor(src_labs, self.use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)
				flt_ids = _convert_to_tensor(flt_ids, self.use_gpu)

				# get padding mask
				non_padding_mask_dd_src = src_ids.data.ne(PAD)
				non_padding_mask_dd_tgt = flt_ids.data.ne(PAD)
				non_padding_mask_gec_tgt = tgt_ids.data.ne(PAD)

				# choose from diff modes
				if mode.lower() == 'gec':
					gec_dd_decoder_outputs, gec_dd_dec_hidden, gec_dd_ret_dict, \
						decoder_outputs, dec_hidden, ret_dict = \
							model.gec_eval(src_ids, tgt_ids, is_training=False,
								gec_dd_att_key_feats=src_probs)
					non_padding_mask = tgt_ids.data.ne(PAD)
					tgts = tgt_ids

				elif mode.lower() == 'dd':
					# import pdb; pdb.set_trace()
					decoder_outputs, dec_hidden, ret_dict = \
						model.dd_eval(src_ids, tgt_ids, is_training=False,
							dd_att_key_feats=src_probs)
					non_padding_mask = tgt_ids.data.ne(PAD)
					tgts = tgt_ids

				elif mode.lower() == 'ddgec-dd':
					decoder_outputs, dec_hidden, ret_dict = \
						model.dd_eval(src_ids, flt_ids, is_training=False,
							dd_att_key_feats=src_probs)
					non_padding_mask = flt_ids.data.ne(PAD)
					tgts = flt_ids

				else:
					assert False, 'Unrecognised eval mode - choose from gec/dd'

				batch_size = src_ids.size(0)
				max_seq_len = src_ids.size(1)

				# Evaluation
				seqlist = ret_dict['sequence']
				for step in range(len(seqlist)):
					target = tgt_ids[:, step+1]
					non_padding = target.ne(PAD)
					if not model.dd_classifier:
						step_output = decoder_outputs[step]
						if not self.eval_with_mask:
							loss.eval_batch(step_output.view(tgt_ids.size(0), -1), target)
						else:
							loss.eval_batch_with_mask(step_output.view(
								tgt_ids.size(0), -1), target, non_padding)
					correct = seqlist[step].view(-1).eq(target)\
						.masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()
				loss.norm_term += 1.0 * torch.sum(non_padding_mask_gec_tgt) - len(seqlist)

				if not model.dd_classifier:
					dsfclassify_loss = 0
				else:
					if type(src_labs) != type(None) \
						and (mode.lower() == 'ddgec-dd' or mode.lower() == 'dd'):
							src_labs = src_labs.view(batch_size, max_seq_len)
							probs = ret_dict['classify_prob']\
								.view(batch_size, max_seq_len) # b * max_seq_len
							dsfclassify_loss.eval_batch_with_mask(
								probs.reshape(-1), src_labs.reshape(-1)\
								.type(torch.FloatTensor).to(device),
								non_padding_mask_dd_src.reshape(-1))
							dsfclassify_loss.norm_term += 1.0 * torch.sum(non_padding_mask_dd_src) - 1

				if out_count < 3:
					srcwords = _convert_to_words_batchfirst(src_ids, tgt_id2word)
					refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], tgt_id2word)
					seqwords = _convert_to_words(seqlist, tgt_id2word)
					outsrc = 'SRC: {}\n'.format(' '.join(srcwords[0])).encode('utf-8')
					outref = 'REF: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
					outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
					sys.stdout.buffer.write(outsrc)
					sys.stdout.buffer.write(outref)
					sys.stdout.buffer.write(outline)
					out_count += 1

		att_resloss = 0
		loss.normalise()

		if type(src_labs) != type(None) and (mode.lower() == 'ddgec-dd' or mode.lower() == 'dd'):
			dsfclassify_loss.normalise()
			attcls_resloss = dsfclassify_loss.get_loss()
		else:
			attcls_resloss = 0

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

		if not model.dd_classifier:
			resloss = loss.get_loss()
		else:
			resloss = 0
		torch.cuda.empty_cache()
		time_2 = time.time()

		# use dict to store losses
		losses = {}
		losses['att_loss'] = att_resloss
		losses['attcls_loss'] = attcls_resloss

		return resloss, accuracy, losses


	# ========= train batch =========
	def _train_batch_separate(self, dd_src_ids, dd_tgt_ids,
        gec_src_ids, gec_tgt_ids, gec_flt_ids,
		model, train_step, total_train_steps,
		dd_src_probs=None, gec_src_probs=None,
		dd_src_labs=None, gec_src_labs=None):

		"""
			used in 'separate' mode - added dd-classifier case

			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				src_labs 		= 	  1  0  0  0    0
			Others:
				internal input 	= 	  w1'  w2'  w3'  </s> <pad> <pad> <pad> [df free]
				decoder_outputs	= 	  w1'' w2'' w3'' </s> <pad> <pad> <pad> [ge free]
		"""

		# import pdb; pdb.set_trace()

		batch_size = dd_src_ids.size(0)
		max_seq_len = dd_src_ids.size(1)

		# define loss
		dd_loss = NLLLoss()
		gec_loss = NLLLoss()

		# scheduled sampling
		progress = 1.0 * train_step / total_train_steps
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			teacher_forcing_ratio = 1.0 - progress

		# get padding mask
		non_padding_mask_dd_src = dd_src_ids.data.ne(PAD)
		non_padding_mask_dd_tgt = dd_tgt_ids.data.ne(PAD)
		non_padding_mask_gec_src = gec_src_ids.data.ne(PAD)
		non_padding_mask_gec_tgt = gec_tgt_ids.data.ne(PAD)
		non_padding_mask_gec_flt = gec_flt_ids.data.ne(PAD)

		# ------------------ debug -------------------
		# import pdb; pdb.set_trace()
		debug_flag = False
		# --------------------------------------------

		# Forward propagation
		dd_decoder_outputs, dd_decoder_hidden, dd_ret_dict, \
			gec_dd_decoder_outputs, gec_dd_dec_hidden, gec_dd_ret_dict, \
			gec_decoder_outputs, gec_decoder_hidden, gec_ret_dict = \
				model(dd_src_ids, gec_src_ids, dd_tgt_ids, gec_tgt_ids,
					is_training=True, teacher_forcing_ratio=teacher_forcing_ratio,
					dd_att_key_feats=dd_src_probs, gec_dd_att_key_feats=gec_src_probs,
					debug_flag=debug_flag)

		# ========================= losses ==========================
		# reset loss
		dd_loss.reset()
		gec_loss.reset()

		if not model.dd_classifier:
			# word prediction

			dummy = torch.log(torch.FloatTensor([1e-40])).repeat(batch_size,
				dd_decoder_outputs[-1].size(-1)).to(device=device)
			dd_decoder_outputs.append(dummy)
			logps = torch.stack(dd_decoder_outputs, dim=1).to(device=device)

			if not self.eval_with_mask:
				dd_loss.eval_batch(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					dd_tgt_ids[:,1:].reshape(-1))
			else:
				dd_loss.eval_batch_with_mask(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					dd_tgt_ids[:,1:].reshape(-1), non_padding_mask_dd_tgt[:,1:].reshape(-1))
			dd_loss.norm_term = 1.0 * torch.sum(non_padding_mask_dd_tgt)
			dd_loss.normalise()

			dummy = torch.log(torch.FloatTensor([1e-40])).repeat(
				batch_size, gec_decoder_outputs[-1].size(-1)).to(device=device)
			gec_decoder_outputs.append(dummy)
			logps = torch.stack(gec_decoder_outputs, dim=1).to(device=device)

			if not self.eval_with_mask:
				gec_loss.eval_batch(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					gec_tgt_ids[:,1:].reshape(-1))
			else:
				gec_loss.eval_batch_with_mask(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					gec_tgt_ids[:,1:].reshape(-1), non_padding_mask_gec_tgt[:,1:].reshape(-1))
			gec_loss.norm_term = 1.0 * torch.sum(non_padding_mask_gec_tgt)
			gec_loss.normalise()

			# Backward propagation
			model.zero_grad()
			dd_coeff = self.dd_loss_weight
			gec_coeff = self.gec_loss_weight

			if self.loss_shift:
				if progress < 0.5:
					pass
				elif progress < 0.7:
					dd_coeff *= 0.8
					gec_coeff *= 1
				elif progress < 0.9:
					dd_coeff *= 0.5
					gec_coeff *= 1
				else:
					dd_coeff *= 0.25
					gec_coeff *= 1

			dd_loss.mul(dd_coeff)
			gec_loss.mul(gec_coeff)
			dd_resloss = dd_loss.get_loss()
			gec_resloss = gec_loss.get_loss()

			# ignore these losses - from earlier exp
			dd_att_resloss = 0
			dd_dsfclassify_resloss = 0
			gec_dd_resloss = 0
			gec_att_resloss = 0
			gec_dsfclassify_resloss = 0

			gec_loss.add(dd_loss)
			gec_loss.backward()
			self.optimizer.step()

		else:

			dd_loss = BCELoss()

			# gec_loss
			dummy = torch.log(torch.FloatTensor([1e-40])).repeat(batch_size,
				gec_decoder_outputs[-1].size(-1)).to(device=device)
			gec_decoder_outputs.append(dummy)
			logps = torch.stack(gec_decoder_outputs, dim=1).to(device=device)

			if not self.eval_with_mask:
				gec_loss.eval_batch(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					gec_tgt_ids[:,1:].reshape(-1))
			else:
				gec_loss.eval_batch_with_mask(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					gec_tgt_ids[:,1:].reshape(-1), non_padding_mask_gec_tgt[:,1:].reshape(-1))
			gec_loss.norm_term = 1.0 * torch.sum(non_padding_mask_gec_tgt)
			gec_loss.normalise()

			# dd_loss
			assert type(dd_src_labs) != type(None), 'need src labels to train classifier!'
			dd_ps = dd_ret_dict['classify_prob']
			dd_loss.eval_batch_with_mask(dd_ps.reshape(-1, dd_ps.size(-1)),
				dd_src_labs.reshape(-1).type(torch.FloatTensor).to(device),
				non_padding_mask_dd_src.reshape(-1))
			dd_loss.norm_term = 1.0 * torch.sum(non_padding_mask_dd_src)
			dd_loss.normalise()

			# sgd
			model.zero_grad()
			gec_coeff = self.gec_loss_weight
			dd_coeff = self.dd_loss_weight

			if self.loss_shift:
				if progress < 0.5:
					pass
				elif progress < 0.7:
					gec_coeff *= 1
					dd_coeff *= 0.8
				else:
					gec_coeff *= 1
					dd_coeff *= 0.5

			gec_loss.mul(gec_coeff)
			gec_resloss = gec_loss.get_loss()
			dd_loss.mul(dd_coeff)
			dd_resloss = dd_loss.get_loss()

			# using classifier for dd - ignore some losses
			dd_att_resloss = 0
			dd_dsfclassify_resloss = 0
			gec_dd_resloss = 0
			gec_att_resloss = 0
			gec_dsfclassify_resloss = 0

			gec_loss.add(dd_loss)
			gec_loss.backward()
			self.optimizer.step()

		return dd_resloss, gec_resloss, gec_dd_resloss, dd_att_resloss, \
			gec_att_resloss, dd_dsfclassify_resloss, gec_dsfclassify_resloss


	def _train_batch_end2end(self, ddgec_src_ids, ddgec_tgt_ids, ddgec_flt_ids,
		model, train_step, total_train_steps, ddgec_src_probs=None, ddgec_src_labs=None):

		"""
			used in 'end2end' mode - added dd-classifier case

			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
			Others:
				internal input 	= 	  w1'  w2'  w3'  </s> <pad> <pad> <pad> [df free]
				decoder_outputs	= 	  w1'' w2'' w3'' </s> <pad> <pad> <pad> [ge free]
		"""

		# import pdb; pdb.set_trace()

		# define loss
		dd_loss = NLLLoss()
		gec_loss = NLLLoss()
		batch_size = ddgec_src_ids.size(0)
		max_seq_len = ddgec_src_ids.size(1)

		# scheduled sampling
		progress = 1.0 * train_step / total_train_steps
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			teacher_forcing_ratio = 1.0 - progress

		# get padding mask
		non_padding_mask_dd_src = ddgec_src_ids.data.ne(PAD)
		non_padding_mask_dd_tgt = ddgec_flt_ids.data.ne(PAD)
		non_padding_mask_gec_tgt = ddgec_tgt_ids.data.ne(PAD)

		# Forward propagation
		gec_dd_decoder_outputs, gec_dd_dec_hidden, gec_dd_ret_dict, \
			gec_decoder_outputs, gec_decoder_hidden, gec_ret_dict = model.gec_train(
				ddgec_src_ids, ddgec_tgt_ids, is_training=True,
				teacher_forcing_ratio=teacher_forcing_ratio, gec_dd_att_key_feats=ddgec_src_probs)

		# ========================= losses ==========================
		# reset loss
		dd_loss.reset()
		gec_loss.reset()

		if not model.dd_classifier:

			logps = torch.stack(gec_dd_decoder_outputs, dim=1).to(device=device)
			if not self.eval_with_mask:
				dd_loss.eval_batch(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					ddgec_flt_ids[:,1:].reshape(-1))
			else:
				dd_loss.eval_batch_with_mask(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					ddgec_flt_ids[:,1:].reshape(-1), non_padding_mask_dd_tgt[:,1:].reshape(-1))

			dummy = torch.log(torch.FloatTensor([1e-40])).repeat(batch_size,
				gec_decoder_outputs[-1].size(-1)).to(device=device)
			gec_decoder_outputs.append(dummy)
			logps = torch.stack(gec_decoder_outputs, dim=1).to(device=device)
			if not self.eval_with_mask:
				gec_loss.eval_batch(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					ddgec_tgt_ids[:,1:].reshape(-1))
			else:
				gec_loss.eval_batch_with_mask(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					ddgec_tgt_ids[:,1:].reshape(-1), non_padding_mask_gec_tgt[:,1:].reshape(-1))

			if not self.eval_with_mask:
				dd_loss.norm_term = batch_size * max_seq_len * 1.0
				gec_loss.norm_term = batch_size * max_seq_len * 1.0
			else:
				dd_loss.norm_term = torch.sum(non_padding_mask_dd_tgt) * 1.0
				gec_loss.norm_term = torch.sum(non_padding_mask_gec_tgt) * 1.0
			dd_loss.normalise()
			gec_loss.normalise()

			# import pdb; pdb.set_trace()
			# Backward propagation
			model.zero_grad()
			dd_coeff = self.dd_loss_weight
			gec_coeff = self.gec_loss_weight

			if self.loss_shift:
				if progress < 0.5:
					pass
				elif progress < 0.7:
					dd_coeff *= 0.8
					gec_coeff *= 1
				elif progress < 0.9:
					dd_coeff *= 0.5
					gec_coeff *= 1
				else:
					dd_coeff *= 0.25
					gec_coeff *= 1

			dd_loss.mul(dd_coeff)
			gec_loss.mul(gec_coeff)
			dd_loss.backward(retain_graph=True)
			gec_loss.backward(retain_graph=True)
			self.optimizer.step()

			att_resloss = 0
			dsfclassify_resloss = 0
			dd_resloss = dd_loss.get_loss()
			gec_resloss = gec_loss.get_loss()

		else:

			dd_loss = BCELoss()

			dummy = torch.log(torch.FloatTensor([1e-40])).repeat(
				batch_size, gec_decoder_outputs[-1].size(-1)).to(device=device)
			gec_decoder_outputs.append(dummy)
			logps = torch.stack(gec_decoder_outputs, dim=1).to(device=device)
			if not self.eval_with_mask:
				gec_loss.eval_batch(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					ddgec_tgt_ids[:,1:].reshape(-1))
			else:
				gec_loss.eval_batch_with_mask(logps[:,:-1,:].reshape(-1, logps.size(-1)),
					ddgec_tgt_ids[:,1:].reshape(-1), non_padding_mask_gec_tgt[:,1:].reshape(-1))
			gec_loss.norm_term = 1.0 * torch.sum(non_padding_mask_gec_tgt)
			gec_loss.normalise()

			# dd_dsfclassify_loss
			assert type(ddgec_src_labs) != type(None), 'need src labels to train classifier!'
			dd_ps = gec_dd_ret_dict['classify_prob']
			dd_loss.eval_batch_with_mask(dd_ps.reshape(-1, dd_ps.size(-1)),
				ddgec_src_labs.reshape(-1).type(torch.FloatTensor).to(device),
				non_padding_mask_dd_src.reshape(-1))
			dd_loss.norm_term = 1.0 * torch.sum(non_padding_mask_dd_src)
			dd_loss.normalise()

			# Backward propagation
			model.zero_grad()
			gec_coeff = self.gec_loss_weight
			dd_coeff = self.dd_loss_weight

			if self.loss_shift:
				if progress < 0.5:
					pass
				elif progress < 0.7:
					gec_coeff *= 1
					dd_coeff *= 0.8
				elif progress < 0.9:
					gec_coeff *= 1
					dd_coeff *= 0.5
				else:
					gec_coeff *= 1
					dd_coeff *= 0.25

			gec_loss.mul(gec_coeff)
			gec_loss.backward(retain_graph=True)
			dd_loss.mul(dd_coeff)
			dd_loss.backward(retain_graph=True)
			self.optimizer.step()

			att_resloss = 0
			dsfclassify_resloss = 0
			dd_resloss = dd_loss.get_loss()
			gec_resloss = gec_loss.get_loss()

		return dd_resloss, gec_resloss, att_resloss, dsfclassify_resloss


	# ========= train epochs =========
	def _train_epochs_separate(self, dd_train_set, gec_train_set,
        model, n_epochs, start_epoch, start_step,
		dd_dev_set=None, gec_dev_set=None, train_mode='separate'):

		log = self.logger

		dd_print_loss_total = 0  # Reset every print_every
		dd_epoch_loss_total = 0  # Reset every epoch
		gec_print_loss_total = 0  # Reset every print_every
		gec_epoch_loss_total = 0  # Reset every epoch
		gec_dd_print_loss_total = 0  # Reset every print_every
		gec_dd_epoch_loss_total = 0  # Reset every epoch

		dd_att_print_loss_total = 0  # Reset every print_every
		dd_att_epoch_loss_total = 0  # Reset every epoch
		gec_att_print_loss_total = 0  # Reset every print_every
		gec_att_epoch_loss_total = 0  # Reset every epoch
		dd_attcls_print_loss_total = 0  # Reset every print_every
		dd_attcls_epoch_loss_total = 0  # Reset every epoch
		gec_attcls_print_loss_total = 0  # Reset every print_every
		gec_attcls_epoch_loss_total = 0  # Reset every epoch

		att_print_loss_total = 0  # Reset every print_every
		att_epoch_loss_total = 0  # Reset every epoch
		attcls_print_loss_total = 0  # Reset every print_every
		attcls_epoch_loss_total = 0  # Reset every epoch

		# training scheduling
		step = start_step
		step_elapsed = 0
		prev_acc = 0.0
		prev_epoch_acc = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# make dir
		if not os.path.exists(os.path.join(self.expt_dir, 'checkpoints')):
			print('make expt dir')
			os.makedirs(os.path.join(self.expt_dir, 'checkpoints'))

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			for param_group in self.optimizer.optimizer.param_groups:
				print('epoch:{} lr: {}'.format(epoch, param_group['lr']))
				lr_curr = param_group['lr']

			MODE = train_mode # 'separate' or 'end2end'
			print('running mode {}'.format(MODE))

			"""
				train dd with swbd + (clc corrupt -> clc flt)
				train gec with clc (clc src -> clc tgt)
			"""
			# import pdb; pdb.set_trace()
			# ----------construct batches-----------
			# allow re-shuffling of data
			# construct dd set
			# self.ddreg only used when MODE == separate

			if type(dd_train_set.tsv_path) == type(None):
				print('--- constrcut dd train set ---')
				dd_train_batches, dd_vocab_size = dd_train_set.construct_batches(is_train=True)
				if dd_dev_set is not None:
					print('--- constrcut dd dev set ---')
					dd_dev_batches, dd_vocab_size = dd_dev_set.construct_batches(is_train=False)
			else:
				print('--- constrcut dd train set + att_ref ---')
				dd_train_batches, dd_vocab_size = dd_train_set.construct_batches_with_ddfd_prob(is_train=True)
				if dd_dev_set is not None:
					print('--- constrcut dd dev set + att_ref ---')
					assert type(dd_dev_set.tsv_path) != type(None), 'DD Dev set missing ddfd probabilities'
					dd_dev_batches, dd_vocab_size = dd_dev_set.construct_batches_with_ddfd_prob(is_train=False)
			# construct gec set
			if type(gec_train_set.tsv_path) == type(None):
				print('--- constrcut gec train set ---')
				gec_train_batches, gec_vocab_size = gec_train_set.construct_batches(is_train=True)
				if gec_dev_set is not None:
					print('--- constrcut gec dev set ---')
					gec_dev_batches, gec_vocab_size = gec_dev_set.construct_batches(is_train=False)
			else:
				print('--- constrcut gec train set + att_ref ---')
				gec_train_batches, gec_vocab_size = gec_train_set.construct_batches_with_ddfd_prob(is_train=True)
				if gec_dev_set is not None:
					print('--- constrcut gec dev set + att_ref ---')
					assert type(gec_dev_set.tsv_path) != type(None), 'GEC Dev set missing ddfd probabilities'
					gec_dev_batches, gec_vocab_size = gec_dev_set.construct_batches_with_ddfd_prob(is_train=False)

			# --------print info for each epoch----------
			# match the training step between DD & GEC & DDGEC corpora

			dd_steps_per_epoch = len(dd_train_batches)
			gec_steps_per_epoch = len(gec_train_batches)

			log.info("dd_steps_per_epoch {}".format(dd_steps_per_epoch))
			log.info("gec_steps_per_epoch {}".format(gec_steps_per_epoch))

			total_steps = gec_steps_per_epoch * n_epochs
			log.info("n_gec_epochs {}".format(n_epochs))
			log.info("total_steps {}".format(total_steps))

			# --------start training----------
			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)

			# ******************** [loop over batches] ********************
			model.train(True)
			for idx in range(len(gec_train_batches)):

				# update macro count
				step += 1
				step_elapsed += 1

				# load batch
				dd_idx = idx % dd_steps_per_epoch
				dd_batch = dd_train_batches[dd_idx]
				gec_batch = gec_train_batches[idx]
				# import pdb; pdb.set_trace()

				# load data from batch
				dd_src_ids = dd_batch['src_word_ids']
				dd_src_lengths = dd_batch['src_sentence_lengths']
				dd_tgt_ids = dd_batch['tgt_word_ids']
				dd_tgt_lengths = dd_batch['tgt_sentence_lengths']

				gec_src_ids = gec_batch['src_word_ids']
				gec_src_lengths = gec_batch['src_sentence_lengths']
				gec_tgt_ids = gec_batch['tgt_word_ids']
				gec_tgt_lengths = gec_batch['tgt_sentence_lengths']
				gec_flt_ids = gec_batch['flt_word_ids']
				gec_flt_lengths = gec_batch['flt_sentence_lengths']


				dd_src_probs = None
				dd_src_labs = None
				if 'src_ddfd_probs' in dd_batch and model.dd_additional_key_size > 0:
					dd_src_probs =  dd_batch['src_ddfd_probs']
					dd_src_probs = _convert_to_tensor(dd_src_probs, self.use_gpu).unsqueeze(2)
				if 'src_ddfd_labs' in dd_batch:
					dd_src_labs = dd_batch['src_ddfd_labs']
					dd_src_labs = _convert_to_tensor(dd_src_labs, self.use_gpu).unsqueeze(2)
				gec_src_probs = None
				gec_src_labs = None
				if 'src_ddfd_probs' in gec_batch and model.dd_additional_key_size > 0:
					gec_src_probs =  gec_batch['src_ddfd_probs']
					gec_src_probs = _convert_to_tensor(gec_src_probs, self.use_gpu).unsqueeze(2)
				if 'src_ddfd_labs' in gec_batch:
					gec_src_labs =  gec_batch['src_ddfd_labs']
					gec_src_labs = _convert_to_tensor(gec_src_labs, self.use_gpu).unsqueeze(2)

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check dd src tgt pair ---')
					log_msgs = check_srctgt(dd_src_ids, dd_tgt_ids,
						dd_train_set.src_id2word, dd_train_set.tgt_id2word)
					for log_msg in log_msgs:
						print(log_msg)
					print('--- Check gec src tgt pair ---')
					log_msgs = check_srctgt(gec_src_ids, gec_tgt_ids,
						gec_train_set.src_id2word, gec_train_set.tgt_id2word)
					for log_msg in log_msgs:
						print(log_msg)

				# convert variable to tensor
				dd_src_ids = _convert_to_tensor(dd_src_ids, self.use_gpu)
				dd_tgt_ids = _convert_to_tensor(dd_tgt_ids, self.use_gpu)
				gec_src_ids = _convert_to_tensor(gec_src_ids, self.use_gpu)
				gec_tgt_ids = _convert_to_tensor(gec_tgt_ids, self.use_gpu)
				gec_flt_ids = _convert_to_tensor(gec_flt_ids, self.use_gpu)

				# Get loss
				dd_loss, gec_loss, gec_dd_loss, dd_att_loss, gec_att_loss, dd_attcls_loss, gec_attcls_loss \
					 = self._train_batch_separate(dd_src_ids, dd_tgt_ids,
							gec_src_ids, gec_tgt_ids, gec_flt_ids,
							model, step, total_steps,
							dd_src_probs=dd_src_probs, gec_src_probs=gec_src_probs,
							dd_src_labs=dd_src_labs, gec_src_labs=gec_src_labs)

				dd_print_loss_total += dd_loss
				dd_epoch_loss_total += dd_loss
				gec_print_loss_total += gec_loss
				gec_epoch_loss_total += gec_loss
				gec_dd_print_loss_total += gec_dd_loss
				gec_dd_epoch_loss_total += gec_dd_loss

				dd_att_print_loss_total += dd_att_loss
				dd_att_epoch_loss_total += dd_att_loss
				gec_att_print_loss_total += gec_att_loss
				gec_att_epoch_loss_total += gec_att_loss
				dd_attcls_print_loss_total += dd_attcls_loss
				dd_attcls_epoch_loss_total += dd_attcls_loss
				gec_attcls_print_loss_total += gec_attcls_loss
				gec_attcls_epoch_loss_total += gec_attcls_loss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					dd_print_loss_avg = dd_print_loss_total / self.print_every
					gec_print_loss_avg = gec_print_loss_total / self.print_every
					gec_dd_print_loss_avg = gec_dd_print_loss_total / self.print_every
					dd_att_print_loss_avg = dd_att_print_loss_total / self.print_every
					gec_att_print_loss_avg = gec_att_print_loss_total / self.print_every
					dd_attcls_print_loss_avg = dd_attcls_print_loss_total / self.print_every
					gec_attcls_print_loss_avg = gec_attcls_print_loss_total / self.print_every
					dd_print_loss_total = 0
					gec_print_loss_total = 0
					gec_dd_print_loss_total = 0
					dd_att_print_loss_total = 0
					gec_att_print_loss_total = 0
					dd_attcls_print_loss_total = 0
					gec_attcls_print_loss_total = 0
					log_msg = 'Progress: %d%%, Train dd: %.4f, gec: %.4f, gec_dd: %.4f, \
						ddatt: %.4f, gecatt: %.4f, ddattcls: %.4f, gecattcls: %.4f ' % (
							step / total_steps * 100,
							dd_print_loss_avg, gec_print_loss_avg, gec_dd_print_loss_avg,
							dd_att_print_loss_avg, gec_att_print_loss_avg,
							dd_attcls_print_loss_avg, gec_attcls_print_loss_avg)
					log.info(log_msg)
					self.writer.add_scalar('dd_train_loss', dd_print_loss_avg, global_step=step)
					self.writer.add_scalar('gec_train_loss', gec_print_loss_avg, global_step=step)
					self.writer.add_scalar('ddatt_train_loss', dd_att_print_loss_avg, global_step=step)
					self.writer.add_scalar('dsfcls_train_loss', dd_attcls_print_loss_avg, global_step=step)

				# =====================================================================
				# Checkpoint
				if self.save_schedule == 'no_roll_back':
					pass
				elif self.save_schedule == 'roll_back':
					# check per self.checkpoint_every
					if step % self.checkpoint_every == 0 or step == total_steps:

						# save criteria
						if gec_dev_set is not None:
							gec_dev_loss, gec_accuracy, _ = \
								self._evaluate_batches(model, gec_dev_batches, gec_dev_set, mode='gec')
							log_msg = 'Progress: %d%%, Dev gec loss: %.4f, accuracy: %.4f' % (
										step / total_steps * 100, gec_dev_loss, gec_accuracy)
							log.info(log_msg)
							self.writer.add_scalar('gec_dev_loss', gec_dev_loss, global_step=step)
							self.writer.add_scalar('gec_dev_acc', gec_accuracy, global_step=step)

							if dd_dev_set is not None:
								dd_dev_loss, dd_accuracy, dd_dev_attlosses = \
									self._evaluate_batches(model, dd_dev_batches, dd_dev_set, mode='dd')
								dd_dev_attloss = dd_dev_attlosses['att_loss']
								dd_dev_attclsloss = dd_dev_attlosses['attcls_loss']
								log_msg = 'Progress: %d%%, Dev dd loss: %.4f, accuracy: %.4f' % (
											step / total_steps * 100, dd_dev_loss, dd_accuracy)
								log.info(log_msg)
								self.writer.add_scalar('dd_dev_loss', dd_dev_loss, global_step=step)
								self.writer.add_scalar('dd_dev_acc', dd_accuracy, global_step=step)
								self.writer.add_scalar('att_dev_loss', dd_dev_attloss, global_step=step)
								self.writer.add_scalar('dsfcls_dev_loss', dd_dev_attclsloss, global_step=step)
								ave_accuracy = self.gec_acc_weight * gec_accuracy + (1-self.gec_acc_weight) * dd_accuracy
							else:
								ave_accuracy = gec_accuracy

							# save
							if prev_acc < ave_accuracy:
								# save best model
								ckpt = Checkpoint(model=model,
										   optimizer=self.optimizer,
										   epoch=epoch, step=step,
										   input_vocab=gec_train_set.vocab_src,
										   output_vocab=gec_train_set.vocab_tgt)
								saved_path = ckpt.save(self.expt_dir)
								print('saving at {} ... '.format(saved_path))
								# reset
								prev_acc = ave_accuracy
								count_no_improve = 0
								count_num_rollback = 0
							else:
								count_no_improve += 1

							# roll back
							if count_no_improve > self.max_count_no_improve:
								# resuming
								latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
								if type(latest_checkpoint_path) != type(None):
									resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
									print('epoch:{} step: {} - rolling back {} ...'.format(
										epoch, step, latest_checkpoint_path))
									model = resume_checkpoint.model
									self.optimizer = resume_checkpoint.optimizer
									# A walk around to set optimizing parameters properly
									resume_optim = self.optimizer.optimizer
									defaults = resume_optim.param_groups[0]
									defaults.pop('params', None)
									defaults.pop('initial_lr', None)
									self.optimizer.optimizer = resume_optim.__class__(
										model.parameters(), **defaults)
									# do not roll back step counts...
									# start_epoch = resume_checkpoint.epoch
									# step = resume_checkpoint.step
								# reassure correct lr
								for param_group in self.optimizer.optimizer.param_groups:
									param_group['lr'] = lr_curr
									print('epoch:{} lr: {}'.format(epoch, param_group['lr']))

								# reset
								count_no_improve = 0
								count_num_rollback += 1

							# update learning rate
							if count_num_rollback > self.max_count_num_rollback:

								# roll back
								latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
								if type(latest_checkpoint_path) != type(None):
									resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
									print('epoch:{} step: {} - rolling back {} ...'.format(
										epoch, step, latest_checkpoint_path))
									model = resume_checkpoint.model
									self.optimizer = resume_checkpoint.optimizer
									# A walk around to set optimizing parameters properly
									resume_optim = self.optimizer.optimizer
									defaults = resume_optim.param_groups[0]
									defaults.pop('params', None)
									defaults.pop('initial_lr', None)
									self.optimizer.optimizer = resume_optim.__class__(
										model.parameters(), **defaults)
									# do not roll back step counts...
									# start_epoch = resume_checkpoint.epoch
									# step = resume_checkpoint.step
								# reassure correct lr
								for param_group in self.optimizer.optimizer.param_groups:
									param_group['lr'] = lr_curr
									print('epoch:{} lr: {}'.format(epoch, param_group['lr']))

								# decrease lr
								for param_group in self.optimizer.optimizer.param_groups:
									param_group['lr'] *= 0.5
									lr_curr = param_group['lr']
									print('reducing lr ...')
									print('step:{} - lr: {}'.format(step, param_group['lr']))

								# check early stop
								if lr_curr < 0.000125:
									print('early stop ...')
									break

								# also save to epoch dir in case - shld be no change
								ckpt = Checkpoint(model=model,
										   optimizer=self.optimizer,
										   epoch=epoch, step=step,
										   input_vocab=gec_train_set.vocab_src,
										   output_vocab=gec_train_set.vocab_tgt)
								saved_path_ep = ckpt.save_epoch(self.expt_dir, epoch)
								print('backup_saving at {} ... '.format(saved_path_ep))

								# reset
								count_no_improve = 0
								count_num_rollback = 0

							model.train(mode=True)

							# keep 5 models
							if ckpt is None:
								ckpt = Checkpoint(model=model,
										   optimizer=self.optimizer,
										   epoch=epoch, step=step,
										   input_vocab=gec_train_set.vocab_src,
										   output_vocab=gec_train_set.vocab_tgt)
							ckpt.rm_old(self.expt_dir, keep_num=5)
							print('n_no_improve {}, num_rollback {}'.format(
								count_no_improve, count_num_rollback))

						else:
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=gec_train_set.vocab_src,
									   output_vocab=gec_train_set.vocab_tgt)
							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))

				else:
					assert True, 'unimplemented mode - {}'.format(self.save_schedule)
				sys.stdout.flush()

			else:

				# per epoch save criteria
				if self.save_schedule == 'no_roll_back':
					# save criteria
					if gec_dev_set is not None:
						gec_dev_loss, gec_accuracy, _ = \
							self._evaluate_batches(model, gec_dev_batches, gec_dev_set, mode='gec')
						log_msg = 'Progress: %d%%, Dev gec loss: %.4f, accuracy: %.4f' % (
									step / total_steps * 100, gec_dev_loss, gec_accuracy)
						log.info(log_msg)
						self.writer.add_scalar('gec_dev_loss', gec_dev_loss, global_step=step)
						self.writer.add_scalar('gec_dev_acc', gec_accuracy, global_step=step)

						if dd_dev_set is not None:
							dd_dev_loss, dd_accuracy, dd_dev_attlosses = \
								self._evaluate_batches(model, dd_dev_batches, dd_dev_set, mode='dd')
							dd_dev_attloss = dd_dev_attlosses['att_loss']
							dd_dev_attclsloss = dd_dev_attlosses['attcls_loss']
							log_msg = 'Progress: %d%%, Dev dd loss: %.4f, accuracy: %.4f' % (
										step / total_steps * 100, dd_dev_loss, dd_accuracy)
							log.info(log_msg)
							self.writer.add_scalar('dd_dev_loss', dd_dev_loss, global_step=step)
							self.writer.add_scalar('dd_dev_acc', dd_accuracy, global_step=step)
							self.writer.add_scalar('att_dev_loss', dd_dev_attloss, global_step=step)
							self.writer.add_scalar('dsfcls_dev_loss', dd_dev_attclsloss, global_step=step)
							ave_accuracy = self.gec_acc_weight * gec_accuracy + (1-self.gec_acc_weight) * dd_accuracy
						else:
							ave_accuracy = gec_accuracy

						# save
						if prev_acc < ave_accuracy:
							# save best model
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=gec_train_set.vocab_src,
									   output_vocab=gec_train_set.vocab_tgt)
							saved_path = ckpt.save(self.expt_dir)
							print('epoch {} saving at {} ... '.format(epoch, saved_path))
							# reset
							prev_acc = ave_accuracy
							count_no_improve = 0
						else:
							count_no_improve += 1

						# decrease lr
						if count_no_improve != 0 and count_no_improve % 3 == 0:
							for param_group in self.optimizer.optimizer.param_groups:
								param_group['lr'] *= 0.5
								lr_curr = param_group['lr']
								print('reducing lr ...')
								print('step:{} - lr: {}'.format(step, param_group['lr']))

						# early stop
						if count_no_improve > 9:
							print('early stopping at epoch {} ...'.format(epoch))
							break

						# keep n models
						if ckpt is None:
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=gec_train_set.vocab_src,
									   output_vocab=gec_train_set.vocab_tgt)
						ckpt.rm_old(self.expt_dir, keep_num=3)
						print('n_no_improve {}, num_rollback {}'.format(count_no_improve, count_num_rollback))
					else:
						ckpt = Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=gec_train_set.vocab_src,
								   output_vocab=gec_train_set.vocab_tgt)
						saved_path = ckpt.save(self.expt_dir)
						print('saving at {} ... '.format(saved_path))

				continue
			# break nested for loop
			break

			# always use gec as the ultimate goal
			if step_elapsed == 0: continue
			gec_epoch_loss_avg = gec_epoch_loss_total / min(gec_steps_per_epoch, step - start_step)
			gec_epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, 'NLLLoss', gec_epoch_loss_avg)

			log.info('\n')
			log.info(log_msg)


	def _train_epochs_end2end(self, ddgec_train_set, model,
        n_epochs, start_epoch, start_step,
		ddgec_dev_set=None, train_mode='end2end'):

		log = self.logger

		dd_print_loss_total = 0  # Reset every print_every
		dd_epoch_loss_total = 0  # Reset every epoch
		gec_print_loss_total = 0  # Reset every print_every
		gec_epoch_loss_total = 0  # Reset every epoch
		gec_dd_print_loss_total = 0  # Reset every print_every
		gec_dd_epoch_loss_total = 0  # Reset every epoch

		dd_att_print_loss_total = 0  # Reset every print_every
		dd_att_epoch_loss_total = 0  # Reset every epoch
		gec_att_print_loss_total = 0  # Reset every print_every
		gec_att_epoch_loss_total = 0  # Reset every epoch
		dd_attcls_print_loss_total = 0  # Reset every print_every
		dd_attcls_epoch_loss_total = 0  # Reset every epoch
		gec_attcls_print_loss_total = 0  # Reset every print_every
		gec_attcls_epoch_loss_total = 0  # Reset every epoch

		att_print_loss_total = 0  # Reset every print_every
		att_epoch_loss_total = 0  # Reset every epoch
		attcls_print_loss_total = 0  # Reset every print_every
		attcls_epoch_loss_total = 0  # Reset every epoch

		# training scheduling
		step = start_step
		step_elapsed = 0
		prev_acc = 0.0
		prev_epoch_acc = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# make dir
		if not os.path.exists(os.path.join(self.expt_dir, 'checkpoints')):
			print('make expt dir')
			os.makedirs(os.path.join(self.expt_dir, 'checkpoints'))

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			for param_group in self.optimizer.optimizer.param_groups:
				print('epoch:{} lr: {}'.format(epoch, param_group['lr']))
				lr_curr = param_group['lr']

			MODE = train_mode # 'separate' or 'end2end'
			print('running mode {}'.format(MODE))

			"""
				train dd and gec with clc corrupt -> clc flt -> clc target)
			"""

			# ----------construct batches-----------
			# allow re-shuffling of data
			# construct ddgec set
			time_st1 = time.time()
			if type(ddgec_train_set.tsv_path) == type(None):
				print('--- constrcut ddgec train set ---')
				ddgec_train_batches, ddgec_vocab_size = ddgec_train_set.construct_batches(is_train=True)
				if ddgec_dev_set is not None:
					print('--- constrcut ddgec dev set ---')
					ddgec_dev_batches, ddgec_vocab_size = ddgec_dev_set.construct_batches(is_train=False)
			else:
				print('--- constrcut ddgec train set ---')
				ddgec_train_batches, ddgec_vocab_size = ddgec_train_set.construct_batches_with_ddfd_prob(is_train=True)
				if ddgec_dev_set is not None:
					print('--- constrcut ddgec dev set ---')
					assert type(ddgec_dev_set.tsv_path) != type(None), 'DDGEC Dev set missing ddfd probabilities'
					ddgec_dev_batches, ddgec_vocab_size = ddgec_dev_set.construct_batches_with_ddfd_prob(is_train=False)
			time_st2 = time.time()

			# --------print info for each epoch----------
			# match the training step between DD & GEC & DDGEC corpora
			ddgec_steps_per_epoch = len(ddgec_train_batches)
			log.info("ddgec_steps_per_epoch {}".format(ddgec_steps_per_epoch))

			total_steps = ddgec_steps_per_epoch * n_epochs
			log.info("n_gec_epochs {}".format(n_epochs))
			log.info("total_steps {}".format(total_steps))

			# --------start training----------
			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)

			# ******************** [loop over batches] ********************
			model.train(True)
			for idx in range(len(ddgec_train_batches)):

				# time
				batch_time_1 = time.time()

				# update macro count
				step += 1
				step_elapsed += 1

				# load batch
				ddgec_batch = ddgec_train_batches[idx]

				# load data from batch
				ddgec_src_ids = ddgec_batch['src_word_ids']
				ddgec_src_lengths = ddgec_batch['src_sentence_lengths']
				ddgec_tgt_ids = ddgec_batch['tgt_word_ids']
				ddgec_tgt_lengths = ddgec_batch['tgt_sentence_lengths']
				ddgec_flt_ids = ddgec_batch['flt_word_ids']
				ddgec_flt_lengths = ddgec_batch['flt_sentence_lengths']

				ddgec_src_probs = None
				ddgec_src_labs = None
				if 'src_ddfd_probs' in ddgec_batch and model.dd_additional_key_size > 0:
					ddgec_src_probs =  ddgec_batch['src_ddfd_probs']
					ddgec_src_probs = _convert_to_tensor(ddgec_src_probs, self.use_gpu).unsqueeze(2)
				if 'src_ddfd_labs' in ddgec_batch:
					ddgec_src_labs =  ddgec_batch['src_ddfd_labs']
					ddgec_src_labs = _convert_to_tensor(ddgec_src_labs, self.use_gpu).unsqueeze(2)

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check ddgec src tgt pair ---')
					log_msgs = check_srctgt(ddgec_src_ids, ddgec_tgt_ids,
                        ddgec_train_set.src_id2word, ddgec_train_set.tgt_id2word)
					for log_msg in log_msgs:
						print(log_msg)

				# convert variable to tensor
				ddgec_src_ids = _convert_to_tensor(ddgec_src_ids, self.use_gpu)
				ddgec_tgt_ids = _convert_to_tensor(ddgec_tgt_ids, self.use_gpu)
				ddgec_flt_ids = _convert_to_tensor(ddgec_flt_ids, self.use_gpu)

				batch_time_2 = time.time()

				# Get loss
				dd_loss, gec_loss, att_loss, attcls_loss = self._train_batch_end2end(
					ddgec_src_ids, ddgec_tgt_ids, ddgec_flt_ids,
					model, step, total_steps,
					ddgec_src_probs=ddgec_src_probs, ddgec_src_labs=ddgec_src_labs)
				batch_time_3 = time.time()

				dd_print_loss_total += dd_loss
				dd_epoch_loss_total += dd_loss
				gec_print_loss_total += gec_loss
				gec_epoch_loss_total += gec_loss
				att_print_loss_total += att_loss
				att_epoch_loss_total += att_loss
				attcls_print_loss_total += attcls_loss
				attcls_epoch_loss_total += attcls_loss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					dd_print_loss_avg = dd_print_loss_total / self.print_every
					gec_print_loss_avg = gec_print_loss_total / self.print_every
					att_print_loss_avg = att_print_loss_total / self.print_every
					attcls_print_loss_avg = attcls_print_loss_total / self.print_every
					dd_print_loss_total = 0
					gec_print_loss_total = 0
					att_print_loss_total = 0
					attcls_print_loss_total = 0
					log_msg = 'Progress: %d%%, Train dd: %.4f, gec: %.4f, att: %.4f, attcls: %.4f' % (
						step / total_steps * 100,
						dd_print_loss_avg, gec_print_loss_avg, att_print_loss_avg, attcls_print_loss_avg)
					log.info(log_msg)
					self.writer.add_scalar('dd_train_loss', dd_print_loss_avg, global_step=step)
					self.writer.add_scalar('gec_train_loss', gec_print_loss_avg, global_step=step)
					self.writer.add_scalar('att_train_loss', att_print_loss_avg, global_step=step)
					self.writer.add_scalar('dsfcls_train_loss', attcls_print_loss_avg, global_step=step)

				batch_time_4 = time.time()

				# =====================================================================
				# Checkpoint

				if self.save_schedule == 'no_roll_back':
					pass

				elif self.save_schedule == 'roll_back':

					if step % self.checkpoint_every == 0 or step == total_steps:

						# save criteria
						if ddgec_dev_set is not None:
							gec_dev_loss, gec_accuracy, _ = \
								self._evaluate_batches(model, ddgec_dev_batches, ddgec_dev_set, mode='gec')
							log_msg = 'Progress: %d%%, Dev gec loss: %.4f, accuracy: %.4f' % (
										step / total_steps * 100, gec_dev_loss, gec_accuracy)
							log.info(log_msg)
							self.writer.add_scalar('gec_dev_loss', gec_dev_loss, global_step=step)
							self.writer.add_scalar('gec_dev_acc', gec_accuracy, global_step=step)

							dd_dev_loss, dd_accuracy, dd_dev_attlosses = \
								self._evaluate_batches(model, ddgec_dev_batches, ddgec_dev_set, mode='ddgec-dd')
							dd_dev_attloss = dd_dev_attlosses['att_loss']
							dd_dev_attclsloss = dd_dev_attlosses['attcls_loss']
							log_msg = 'Progress: %d%%, Dev dd loss: %.4f, accuracy: %.4f' % (
										step / total_steps * 100, dd_dev_loss, dd_accuracy)
							log.info(log_msg)
							self.writer.add_scalar('dd_dev_loss', dd_dev_loss, global_step=step)
							self.writer.add_scalar('dd_dev_acc', dd_accuracy, global_step=step)
							self.writer.add_scalar('att_dev_loss', dd_dev_attloss, global_step=step)
							self.writer.add_scalar('dsfcls_dev_loss', dd_dev_attclsloss, global_step=step)
							ave_accuracy = self.gec_acc_weight * gec_accuracy + (1-self.gec_acc_weight) * dd_accuracy

							# save
							if prev_acc < ave_accuracy:
								# save best model
								ckpt = Checkpoint(model=model,
										   optimizer=self.optimizer,
										   epoch=epoch, step=step,
										   input_vocab=ddgec_train_set.vocab_src,
										   output_vocab=ddgec_train_set.vocab_tgt)
								saved_path = ckpt.save(self.expt_dir)
								print('saving at {} ... '.format(saved_path))
								# reset
								prev_acc = ave_accuracy
								count_no_improve = 0
								count_num_rollback = 0
							else:
								count_no_improve += 1

							# roll back
							if count_no_improve > self.max_count_no_improve:
								# resuming
								latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
								if type(latest_checkpoint_path) != type(None):
									resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
									print('epoch:{} step: {} - rolling back {} ...'.format(
										epoch, step, latest_checkpoint_path))
									model = resume_checkpoint.model
									self.optimizer = resume_checkpoint.optimizer
									# A walk around to set optimizing parameters properly
									resume_optim = self.optimizer.optimizer
									defaults = resume_optim.param_groups[0]
									defaults.pop('params', None)
									defaults.pop('initial_lr', None)
									self.optimizer.optimizer = resume_optim.__class__(
										model.parameters(), **defaults)
									# do not roll back step counts...
									# start_epoch = resume_checkpoint.epoch
									# step = resume_checkpoint.step
								# reassure correct lr
								for param_group in self.optimizer.optimizer.param_groups:
									param_group['lr'] = lr_curr
									print('epoch:{} lr: {}'.format(epoch, param_group['lr']))

								# reset
								count_no_improve = 0
								count_num_rollback += 1

							# update learning rate
							if count_num_rollback > self.max_count_num_rollback:

								# roll back
								latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
								if type(latest_checkpoint_path) != type(None):
									resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
									print('epoch:{} step: {} - rolling back {} ...'\
										.format(epoch, step, latest_checkpoint_path))
									model = resume_checkpoint.model
									self.optimizer = resume_checkpoint.optimizer
									# A walk around to set optimizing parameters properly
									resume_optim = self.optimizer.optimizer
									defaults = resume_optim.param_groups[0]
									defaults.pop('params', None)
									defaults.pop('initial_lr', None)
									self.optimizer.optimizer = resume_optim.__class__(
										model.parameters(), **defaults)
									# do not roll back step counts...
									# start_epoch = resume_checkpoint.epoch
									# step = resume_checkpoint.step
								# reassure correct lr
								for param_group in self.optimizer.optimizer.param_groups:
									param_group['lr'] = lr_curr
									print('epoch:{} lr: {}'.format(epoch, param_group['lr']))

								# decrease lr
								for param_group in self.optimizer.optimizer.param_groups:
									param_group['lr'] *= 0.5
									lr_curr = param_group['lr']
									print('reducing lr ...')
									print('step:{} - lr: {}'.format(step, param_group['lr']))

								# check early stop
								if lr_curr < 0.000125:
									print('early stop ...')
									break

								# reset
								count_no_improve = 0
								count_num_rollback = 0

							model.train(mode=True)

							# keep 5 models
							if ckpt is None:
								ckpt = Checkpoint(model=model,
										   optimizer=self.optimizer,
										   epoch=epoch, step=step,
										   input_vocab=ddgec_train_set.vocab_src,
										   output_vocab=ddgec_train_set.vocab_tgt)
							ckpt.rm_old(self.expt_dir, keep_num=KEEP_NUM_END2END)
							print('n_no_improve {}, num_rollback {}'.format(
								count_no_improve, count_num_rollback))

						else:
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=ddgec_train_set.vocab_src,
									   output_vocab=ddgec_train_set.vocab_tgt)
							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))

					batch_time_5 = time.time()

				sys.stdout.flush()
				batch_time_ed = time.time()

			else:
				# be executed if for loop is not breaked and executed till the end
				# per epoch save
				if self.save_schedule == 'no_roll_back':
					if ddgec_dev_set is not None:
						gec_dev_loss, gec_accuracy, _ = \
							self._evaluate_batches(model, ddgec_dev_batches, ddgec_dev_set, mode='gec')
						log_msg = 'Progress: %d%%, Dev gec loss: %.4f, accuracy: %.4f' % (
									step / total_steps * 100, gec_dev_loss, gec_accuracy)
						log.info(log_msg)
						self.writer.add_scalar('gec_dev_loss', gec_dev_loss, global_step=step)
						self.writer.add_scalar('gec_dev_acc', gec_accuracy, global_step=step)

						dd_dev_loss, dd_accuracy, dd_dev_attlosses = \
							self._evaluate_batches(model, ddgec_dev_batches, ddgec_dev_set, mode='ddgec-dd')
						dd_dev_attloss = dd_dev_attlosses['att_loss']
						dd_dev_attclsloss = dd_dev_attlosses['attcls_loss']
						log_msg = 'Progress: %d%%, Dev dd loss: %.4f, accuracy: %.4f' % (
									step / total_steps * 100, dd_dev_loss, dd_accuracy)
						log.info(log_msg)
						self.writer.add_scalar('dd_dev_loss', dd_dev_loss, global_step=step)
						self.writer.add_scalar('dd_dev_acc', dd_accuracy, global_step=step)
						self.writer.add_scalar('att_dev_loss', dd_dev_attloss, global_step=step)
						self.writer.add_scalar('dsfcls_dev_loss', dd_dev_attclsloss, global_step=step)
						ave_accuracy = self.gec_acc_weight * gec_accuracy + (1-self.gec_acc_weight) * dd_accuracy

						# save
						if prev_acc < ave_accuracy:
							# save best model
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=ddgec_train_set.vocab_src,
									   output_vocab=ddgec_train_set.vocab_tgt)
							saved_path = ckpt.save(self.expt_dir)
							print('epoch {} saving at {} ... '.format(epoch, saved_path))
							# reset
							prev_acc = ave_accuracy
							count_no_improve = 0
							count_num_rollback = 0
						else:
							count_no_improve += 1

						# decrease lr
						if count_no_improve != 0 and count_no_improve % 5 == 0:
							for param_group in self.optimizer.optimizer.param_groups:
								param_group['lr'] *= 0.5
								lr_curr = param_group['lr']
								print('reducing lr ...')
								print('step:{} - lr: {}'.format(step, param_group['lr']))

						# early stop
						if count_no_improve > 15:
							print('early stopping at epoch {} ...'.format(epoch))
							break

						# keep 5 models
						if ckpt is None:
							ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=ddgec_train_set.vocab_src,
							   output_vocab=ddgec_train_set.vocab_tgt)
						ckpt.rm_old(self.expt_dir, keep_num=KEEP_NUM_END2END)
						print('n_no_improve {}, num_rollback {}'
							.format(count_no_improve, count_num_rollback))

				continue

			# break nested for loop
			break

			# always use gec as the ultimate goal
			if step_elapsed == 0: continue
			gec_epoch_loss_avg = gec_epoch_loss_total / min(
				ddgec_steps_per_epoch, step - start_step)
			gec_epoch_loss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f" \
				% (epoch, 'NLLLoss', gec_epoch_loss_avg)

			log.info('\n')
			log.info(log_msg)

			time_st3 = time.time()
			print('per epoch training time {}'.format(time_st3 - time_st2))


	# ========= train =========
	def train_separate(self, dd_train_set, gec_train_set, model,
		num_epochs=20, optimizer=None, dd_dev_set=None, gec_dev_set=None):

		"""
			mode = 'separate'
			Run training.
			Args:
				*_train_set:	Train Dataset
				*_dev_set:		Dev Dataset, optional
				model: 			model to run training on,
								if `resume=True`, it would be overwritten
								by the model loaded from the latest checkpoint.
				num_epochs (int, optional):
								number of epochs to run (according to gec_train_set)
				optimizer (seq2seq.optim.Optimizer, optional):
								optimizer for training
								(default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))

			Returns:
				model (seq2seq.models): trained model.
		"""

		start_epoch, step, model = self.model_init(model, optimizer)

		self._train_epochs_separate(dd_train_set, gec_train_set, model, num_epochs, start_epoch, step,
							dd_dev_set=dd_dev_set, gec_dev_set=gec_dev_set, train_mode='separate')

		return model


	def train_end2end(self, ddgec_train_set, model,
		num_epochs=20, optimizer=None, ddgec_dev_set=None):

		"""
			mode = 'end2ends'
			Run training.
			Args:
				*_train_set:	Train Dataset
				*_dev_set:		Dev Dataset, optional
				model: 			model to run training on,
								if `resume=True`, it would be overwritten
								by the model loaded from the latest checkpoint.
				num_epochs (int, optional):
					number of epochs to run (according to gec_train_set)
				optimizer (seq2seq.optim.Optimizer, optional):
								optimizer for training
								(default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))

			Returns:
				model (seq2seq.models): trained model.
		"""

		start_epoch, step, model = self.model_init(model, optimizer)

		self._train_epochs_end2end(ddgec_train_set, model, num_epochs,
			start_epoch, step, ddgec_dev_set=ddgec_dev_set, train_mode='end2end')

		return model


	def model_init(self, model, optimizer):

		torch.cuda.empty_cache()

		if self.load_dir:
			latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.load_dir)
			# latest_checkpoint_path = Checkpoint.get_latest_epoch_checkpoint(self.load_dir)
			print('resuming from {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model.to(device)
			print(model)
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

		else:
			if self.restart_dir:
				print('restarting from {} ...'.format(self.restart_dir))
				# only load the model but start new training schedule
				resume_checkpoint = Checkpoint.load(self.restart_dir)
				model_old = resume_checkpoint.model.to(device)

			start_epoch = 1
			step = 0

			for name, param in model.named_parameters():
				loaded = False
				log = self.logger.info('{}:{}'.format(name, param.size()))
				if self.restart_dir:
					for name2, param_old in model_old.named_parameters():
						if name == name2:
							assert param.data.size() == param_old.data.size(), \
								'name_old {} {} : name {} {}'.format(
									name2, param_old.data.size() , name, param.data.size())
							param.data = param_old.data
							print('loading {}'.format(name))
							loaded = True

				if not loaded:
					print('not preloaded - {}'.format(name))

			if optimizer is None:

				optimizer = Optimizer(torch.optim.Adam(model.parameters(),
					lr=self.learning_rate), max_grad_norm=self.max_grad_norm)

			self.optimizer = optimizer


		self.logger.info("Optimizer: %s, Scheduler: %s" \
			% (self.optimizer.optimizer, self.optimizer.scheduler))

		return start_epoch, step, model
