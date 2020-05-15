# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import collections
import codecs
import numpy as np
import random

from utils.config import PAD, UNK, BOS, EOS

class Dataset(object):

	""" load src-tgt from file - does not involve torch """

	def __init__(self,
		# add params 
		path_src,
		path_tgt,
		path_vocab_src,
		path_vocab_tgt,
		attkey_path=None,
		reflab_path=None,
		max_seq_len=32,
		batch_size=64,
		use_gpu=True,
		# for debug
		attscore_path=None,
		path_flt=None,
		seqtag_sharpen=False,
		seqrev=False,
		tf_prob=False
		):

		"""
			Vars to be used - 
			always non-empty
				self.train_src_word_ids [l,1]
				self.train_src_sentence_lengths [l,1]
				self.train_tgt_word_ids [l,1]
				self.train_tgt_sentence_lengths [l,1]
				self.train_flt_word_ids [l,1]
				self.train_flt_sentence_lengths [l,1]
			all 0s if not defined
				self.train_src_ddfd_probs [l,1]
				self.train_src_ddfd_labs [l,1]
			None if not defined
				self.train_attscores [l-1,l,1]
		"""

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.path_vocab_src = path_vocab_src
		self.path_vocab_tgt = path_vocab_tgt
		self.reflab_path = reflab_path
		self.attkey_path = attkey_path
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.path_flt = path_flt

		# under tweaking ... 
		self.seqtag_sharpen = seqtag_sharpen
		self.seqrev = seqrev
		self.tf_prob = tf_prob 

		self.attscore_path = attscore_path
		if type(self.path_flt) == type(None):
			self.path_flt = self.path_tgt
			# self.path_flt = self.path_src

		print('src path {}'.format(self.path_src))
		print('flt path {}'.format(self.path_flt))
		print('tgt path {}'.format(self.path_tgt))

		self.load_vocab()
		self.load_sentences()
		self.load_ddfd_prob()
		self.load_attscore()
		self.preprocess()


	def load_vocab(self):
		
		self.vocab_src = []
		self.vocab_tgt = []
		with codecs.open(self.path_vocab_src, encoding='UTF-8') as f:
			vocab_src_lines	= f.readlines()
		with codecs.open(self.path_vocab_tgt, encoding='UTF-8') as f:
			vocab_tgt_lines = f.readlines()

		self.src_word2id = collections.OrderedDict()
		self.tgt_word2id = collections.OrderedDict()
		self.src_id2word = collections.OrderedDict()
		self.tgt_id2word = collections.OrderedDict()

		for i, word in enumerate(vocab_src_lines):
			word = word.strip() # remove \n
			self.vocab_src.append(word)
			self.src_word2id[word] = i
			self.src_id2word[i] = word

		for i, word in enumerate(vocab_tgt_lines):
			word = word.strip() # remove \n
			self.vocab_tgt.append(word)
			self.tgt_word2id[word] = i
			self.tgt_id2word[i] = word


	def load_sentences(self):

		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()
		with codecs.open(self.path_tgt, encoding='UTF-8') as f:
			self.tgt_sentences = f.readlines()

		if not hasattr(self, 'path_flt'):
			self.path_flt = self.path_tgt
			# print('here !!')
		with codecs.open(self.path_flt, encoding='UTF-8') as f:
			self.flt_sentences = f.readlines()

		assert len(self.src_sentences) == len(self.tgt_sentences), 'Mismatch src:tgt - {}:{}' \
					.format(len(self.src_sentences),len(self.tgt_sentences))
		assert len(self.src_sentences) == len(self.flt_sentences), 'Mismatch src:flt - {}:{}' \
					.format(len(self.src_sentences),len(self.flt_sentences))

		if self.seqrev:
			for idx in range(len(self.src_sentences)):
				src_sent_rev = self.src_sentences[idx].strip().split()[::-1]
				tgt_sent_rev = self.tgt_sentences[idx].strip().split()[::-1]
				flt_sent_rev = self.flt_sentences[idx].strip().split()[::-1]
				# print(self.src_sentences[idx])
				self.src_sentences[idx] = ' '.join(src_sent_rev)
				self.tgt_sentences[idx] = ' '.join(tgt_sent_rev)
				self.flt_sentences[idx] = ' '.join(flt_sent_rev)
				# print(self.src_sentences[idx])
				# input('...')


	def load_ddfd_prob(self):

		""" 
			laod the probability of each src word being disfluency or filler
			token label prob_o prob_e
		"""
		THRESHOLD = 0.35

		if self.attkey_path == None:
			self.ddfd_seq_probs = None
			self.ddfd_seq_labs = None
		else:
			with codecs.open(self.attkey_path, encoding='UTF-8') as f:
				lines = f.readlines()

				seq = []
				lab_seq = []
				self.ddfd_seq_probs = []
				self.ddfd_seq_labs = []
				for line in lines:
					if line == '\n':
						if len(seq):
							if self.seqrev:
								# reverse sequence for reverse decoding
								seq = seq[::-1]
								lab_seq = lab_seq[::-1]
							self.ddfd_seq_probs.append(seq)
							self.ddfd_seq_labs.append(lab_seq)
							seq = []
							lab_seq = []
					else:
						elems = line.strip().split('\t')
						tok = elems[0]
						if len(elems) == 4:
							lab = elems[-3]
							prob = float(elems[-1][2:]) # [-1]:prob for E; [-2]:prob for O; [-3] lab                                 
						elif len(elems) == 2:
							lab = elems[1][-1]
							if lab == 'E':
								prob = 1.0
							else:
								prob = 0.0
							# print(lab, prob)
							# input('...')
						else:
							assert False, 'check tsv file, requires either 2 or 4 elems per line'

						lab_seq.append(lab)
						if self.tf_prob:
							ref_prob = 1.0 if lab == 'E' else 0.0
							seq.append(ref_prob)
						elif self.seqtag_sharpen:
							res = 1.0 if prob > THRESHOLD else 0.0
							seq.append(res)
						else:
							seq.append(prob)

			# print('ddfd prob loaded ... {}'.format(self.attkey_path))
			# print('seq_probs', len(self.ddfd_seq_probs))
			# print('seq_labs', len(self.ddfd_seq_labs))
			assert len(self.src_sentences) == len(self.ddfd_seq_probs), 'Mismatch src:ddfd_prob - {}:{}' \
						.format(len(self.src_sentences),len(self.ddfd_seq_probs))


	def load_ddfd_lab(self):

		""" 
			deprecated - (included into load_ddfd_prob)
			laod the label of each src word being disfluency/filler 
			1 line = 1 sequence 
		"""

		if self.reflab_path == None:
			self.ddfd_seq_labs = None
		else:
			with codecs.open(self.reflab_path, encoding='UTF-8') as f:
				lines = f.readlines()

				self.ddfd_seq_labs = []
				for line in lines:
					elems = line.strip().split()
					self.ddfd_seq_labs.append(elems)

			assert len(self.src_sentences) == len(self.ddfd_seq_labs), 'Mismatch src:ddfd_lab - {}:{}' \
						.format(len(self.src_sentences),len(self.ddfd_seq_labs))


	def load_attscore(self):

		""" 
			Use:
				laod reference attention scores 
			Create:
				self.train_attscores
		"""

		if self.attscore_path == None:
			self.train_attscores = None
		else:
			self.train_attscores = list(np.load(self.attscore_path))


	def preprocess(self):

		"""
			Use:
				map word2id once for all epoches (improved data loading efficiency)
				shuffling is done later
			Returns:
				0 - over the entire epoch
				1 - ids of src/flt/tgt
				2 - probs/labs for each word ([0.0, 1.0])
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				ddfd_prob:		p1 p2  p3  p4  ...        0   0   0   ...
				ddfd_labs: 		0  1   0   0   ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
			Note:
				src/flt/tgt are always given
				probs/labs might be None (set to all zeors)
			Create
				self.train_src_ddfd_probs
				self.train_src_ddfd_labs
				self.train_src_word_ids
				self.train_src_sentence_lengths
				self.train_tgt_word_ids
				self.train_tgt_sentence_lengths
				self.train_flt_word_ids
				self.train_flt_sentence_lengths
		"""

		self.vocab_size = {'src': len(self.src_word2id), 'tgt': len(self.tgt_word2id)}
		print("num_vocab_src: ", self.vocab_size['src'])
		print("num_vocab_tgt: ", self.vocab_size['tgt'])

		# declare temporary vars
		train_src_ddfd_probs = []
		train_src_ddfd_labs = []
		train_src_word_ids = []
		train_src_sentence_lengths = []
		train_tgt_word_ids = []
		train_tgt_sentence_lengths = []		
		train_flt_word_ids = []
		train_flt_sentence_lengths = []		

		for idx in range(len(self.src_sentences)):
			src_sentence = self.src_sentences[idx]
			tgt_sentence = self.tgt_sentences[idx]
			flt_sentence = self.flt_sentences[idx]
			src_words = src_sentence.strip().split()
			tgt_words = tgt_sentence.strip().split()
			flt_words = flt_sentence.strip().split()
			src_probs = None
			src_labs = None
			if type(self.ddfd_seq_probs) != type(None):
				src_probs =  self.ddfd_seq_probs[idx]
			if type(self.ddfd_seq_labs) != type (None):
				src_labs =  self.ddfd_seq_labs[idx]
			# print(src_words)
			# print(src_probs)
			# input('...')

			# sanity check - ddfd length okay
			if type(src_probs) != type(None):
				assert len(src_words) == len(src_probs), 'Mismatch length of src words, probs {}:{}' \
							.format(len(src_words),len(src_probs))

			# ignore long seq
			if len(src_words) > self.max_seq_len - 1 or len(tgt_words) > self.max_seq_len - 2 or len(flt_words) > self.max_seq_len - 2:
				# src + EOS
				# tgt + BOS + EOS
				continue

			# source
			src_ids = [PAD] * self.max_seq_len
			for i, word in enumerate(src_words):
				if word in self.src_word2id:
					src_ids[i] = self.src_word2id[word]
				else:
					src_ids[i] = UNK
			src_ids[i+1] = EOS
			train_src_word_ids.append(src_ids)
			train_src_sentence_lengths.append(len(src_words)+1) # include one EOS

			# probs
			full_probs = [float(0)] * self.max_seq_len
			if type(src_probs) != type(None):
				for i, prob in enumerate(src_probs): #flaot seq
					full_probs[i] = prob # prediction of token being E
			train_src_ddfd_probs.append(full_probs)

			# labs
			full_labs = [float(0)] * self.max_seq_len
			if type(src_labs) != type(None):
				for i, lab in enumerate(src_labs): #str seq
					if lab == 'E':
						full_labs[i] = float(1) # label of token being E = 1
			train_src_ddfd_labs.append(full_labs)			

			# target
			tgt_ids = [PAD] * self.max_seq_len
			tgt_ids[0] = BOS
			for i, word in enumerate(tgt_words):
				if word in self.tgt_word2id:
					tgt_ids[i+1] = self.tgt_word2id[word]
				else:
					tgt_ids[i+1] = UNK
			tgt_ids[i+2] = EOS
			train_tgt_word_ids.append(tgt_ids)
			train_tgt_sentence_lengths.append(len(tgt_words)+2) # include EOS + BOS

			# flt target
			flt_ids = [PAD] * self.max_seq_len
			flt_ids[0] = BOS
			for i, word in enumerate(flt_words):
				if word in self.tgt_word2id:
					flt_ids[i+1] = self.tgt_word2id[word]
				else:
					flt_ids[i+1] = UNK
			flt_ids[i+2] = EOS
			train_flt_word_ids.append(flt_ids)
			train_flt_sentence_lengths.append(len(flt_words)+2) # include EOS + BOS

		assert (len(train_src_word_ids) == len(train_tgt_word_ids)), "train_src_word_ids != train_tgt_word_ids"
		assert (len(train_flt_word_ids) == len(train_tgt_word_ids)), "train_flt_word_ids != train_tgt_word_ids"
		assert (len(train_src_word_ids) == len(train_src_ddfd_probs)), "train_src_word_ids != train_src_ddfd_probs"

		self.num_training_sentences = len(train_src_word_ids)
		print("num_sentences: ", self.num_training_sentences) # only those that are not too long

		# set class var to be used in batchify
		self.train_src_ddfd_probs = train_src_ddfd_probs
		self.train_src_ddfd_labs = train_src_ddfd_labs
		self.train_src_word_ids = train_src_word_ids
		self.train_src_sentence_lengths = train_src_sentence_lengths
		self.train_tgt_word_ids = train_tgt_word_ids
		self.train_tgt_sentence_lengths = train_tgt_sentence_lengths
		self.train_flt_word_ids = train_flt_word_ids
		self.train_flt_sentence_lengths = train_flt_sentence_lengths


	def construct_batches(self, is_train=False):

		"""
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 	a cat cat sat on the mat EOS PAD PAD ...
				tgt:	BOS a cat sat on the mat EOS PAD PAD ...
		"""

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids, self.train_flt_word_ids, 
						self.train_src_sentence_lengths, self.train_tgt_sentence_lengths, self.train_flt_sentence_lengths))
		if is_train:
			random.shuffle(_x)
		train_src_word_ids, train_tgt_word_ids, train_flt_word_ids, \
				train_src_sentence_lengths, train_tgt_sentence_lengths, train_flt_sentence_lengths = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'flt_word_ids': train_flt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'flt_sentence_lengths': train_flt_sentence_lengths[i_start:i_end]}
			batches.append(batch)

		# add the last batch 
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_flt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []
			last_flt_sentence_lengths = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_flt_word_ids.extend(train_flt_word_ids[i_start:i_end])
			last_flt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_flt_sentence_lengths.extend(train_flt_sentence_lengths[i_start:i_end])
			last_flt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'flt_word_ids': last_flt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths,
				'flt_sentence_lengths': last_flt_sentence_lengths}
			batches.append(batch)

		print("num_batches: ", len(batches))

		return batches, self.vocab_size


	def construct_batches_with_ddfd_prob(self, is_train=False):

		"""
			Add ddfd probabilities to each batch - used as keys for attention mechanism 
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				ddfd_prob:		p1 p2  p3  p4  ...        0   0   0   ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
		"""

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids, self.train_flt_word_ids, 
				self.train_src_sentence_lengths, self.train_tgt_sentence_lengths, self.train_flt_sentence_lengths, 
				self.train_src_ddfd_probs, self.train_src_ddfd_labs))
		if is_train:
			random.shuffle(_x)
		train_src_word_ids, train_tgt_word_ids, train_flt_word_ids,\
			train_src_sentence_lengths, train_tgt_sentence_lengths, train_flt_sentence_lengths,\
			train_src_ddfd_probs, train_src_ddfd_labs = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'flt_word_ids': train_flt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'flt_sentence_lengths': train_flt_sentence_lengths[i_start:i_end],
				'src_ddfd_probs': train_src_ddfd_probs[i_start:i_end],
				'src_ddfd_labs': train_src_ddfd_labs[i_start:i_end]}
			batches.append(batch)

		# add the last batch (underfull batch - add paddings)
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			dummy_prob = [float(0)] * self.max_seq_len
			dummy_lab = [int(0)] * self.max_seq_len
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_flt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []
			last_flt_sentence_lengths = []
			last_src_ddfd_probs = []
			last_src_ddfd_labs = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_flt_word_ids.extend(train_flt_word_ids[i_start:i_end])
			last_flt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_flt_sentence_lengths.extend(train_flt_sentence_lengths[i_start:i_end])
			last_flt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_src_ddfd_probs.extend(train_src_ddfd_probs[i_start:i_end])
			last_src_ddfd_probs.extend([dummy_prob] * (pad_i_end - pad_i_start))
			last_src_ddfd_labs.extend(train_src_ddfd_labs[i_start:i_end])
			last_src_ddfd_labs.extend([dummy_lab] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'flt_word_ids': last_flt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths,
				'flt_sentence_lengths': last_flt_sentence_lengths,
				'src_ddfd_probs': last_src_ddfd_probs,
				'src_ddfd_labs': last_src_ddfd_labs}
			batches.append(batch)

		print("num_batches: ", len(batches))

		return batches, self.vocab_size


	def construct_batches_with_ddfd_lab(self, is_train=False):

		"""
			Deprecated
			Add ddfd labs to each batch -	used for attention mechanism eval 
											(later might use in training)
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				ddfd_lab:		0  1   0   ...        0   0   0   ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
		"""

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids, self.train_flt_word_ids, 
				self.train_src_sentence_lengths, self.train_tgt_sentence_lengths, self.train_flt_sentence_lengths,
				self.train_src_ddfd_labs))
		if is_train:
			random.shuffle(_x)
		train_src_word_ids, train_tgt_word_ids, train_flt_word_ids,\
			train_src_sentence_lengths, train_tgt_sentence_lengths, train_flt_sentence_lengths,\
			train_src_ddfd_labs = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'flt_word_ids': train_flt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'flt_sentence_lengths': train_flt_sentence_lengths[i_start:i_end],
				'src_ddfd_labs': train_src_ddfd_labs[i_start:i_end]}
			batches.append(batch)

		# add the last batch (underfull batch - add paddings)
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			dummy_lab = [int(0)] * self.max_seq_len
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_flt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []
			last_flt_sentence_lengths = []
			last_src_ddfd_labs = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_flt_word_ids.extend(train_flt_word_ids[i_start:i_end])
			last_flt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_flt_sentence_lengths.extend(train_flt_sentence_lengths[i_start:i_end])
			last_flt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_src_ddfd_labs.extend(train_src_ddfd_labs[i_start:i_end])
			last_src_ddfd_labs.extend([dummy_lab] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'flt_word_ids': last_flt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths,
				'flt_sentence_lengths': last_flt_sentence_lengths,
				'src_ddfd_labs': last_src_ddfd_labs}
			batches.append(batch)

		print("num_batches: ", len(batches))

		return batches, self.vocab_size


	def construct_batches_with_attscore(self, is_train=False):

		"""
			used in translation only 
			Add reference att scores - kept unchanged

			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
				attscore:	31 * 32 array 
		"""

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids, self.train_flt_word_ids, 
				self.train_src_sentence_lengths, self.train_tgt_sentence_lengths, self.train_flt_sentence_lengths, 
				self.train_attscores))
		if is_train:
			random.shuffle(_x)
		train_src_word_ids, train_tgt_word_ids, train_flt_word_ids,\
			train_src_sentence_lengths, train_tgt_sentence_lengths, train_flt_sentence_lengths,\
			train_attscores = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'flt_word_ids': train_flt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'flt_sentence_lengths': train_flt_sentence_lengths[i_start:i_end],
				'attscores': train_attscores[i_start:i_end]}
			batches.append(batch)

		# add the last batch (underfull batch - add paddings)
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			dummy_prob = np.zeros((self.max_seq_len-1, self.max_seq_len))
			# dummy_prob.astype(np.double)
			# print(dummy_prob)
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_flt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []
			last_flt_sentence_lengths = []
			last_attscores = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_flt_word_ids.extend(train_flt_word_ids[i_start:i_end])
			last_flt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_flt_sentence_lengths.extend(train_flt_sentence_lengths[i_start:i_end])
			last_flt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_attscores.extend(train_attscores[i_start:i_end])
			last_attscores.extend([dummy_prob] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'flt_word_ids': last_flt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths,
				'flt_sentence_lengths': last_flt_sentence_lengths,
				'attscores': last_attscores}
			batches.append(batch)

		print("num_batches: ", len(batches))

		return batches, self.vocab_size


def load_pretrained_embedding(word2id, embedding_matrix, embedding_path):

	""" assign value to src_word_embeddings and tgt_word_embeddings """

	counter = 0
	with codecs.open(embedding_path, encoding="UTF-8") as f:
		for line in f:
			items = line.strip().split()
			if len(items) <= 2:
				continue
			word = items[0].lower()
			if word in word2id:
				id = word2id[word]
				vector = np.array(items[1:])
				embedding_matrix[id] = vector
				counter += 1	

	print('loaded pre-trained embedding:', embedding_path)
	print('embedding vectors found:', counter)

	return embedding_matrix





























