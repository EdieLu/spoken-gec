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

from utils.misc import set_global_seeds, print_config, save_config, check_srctgt
from utils.misc import validate_config, get_memory_alloc, convert_dd_att_ref, convert_dd_att_ref_inv
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor, _del_var
from utils.dataset import Dataset
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import Seq2Seq
from trainer import Trainer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)

# defalt as 'cpu'
device = torch.device('cpu')
KEEP_NUM_END2END = 1

def load_arguments(parser):

	""" Seq2Seq model

		end2end: mainly used for finetuing
		separate: used for pre-training

	"""

	# paths
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')

	parser.add_argument('--dd_train_path_src', type=str, default=None, help='dd_train src dir')
	parser.add_argument('--dd_train_path_tgt', type=str, default=None, help='dd_train tgt dir')
	parser.add_argument('--dd_train_path_flt', type=str, default=None, help='dd_train flt dir')
	parser.add_argument('--dd_dev_path_src', type=str, default=None, help='dd_dev src dir')
	parser.add_argument('--dd_dev_path_tgt', type=str, default=None, help='dd_dev tgt dir')
	parser.add_argument('--dd_dev_path_flt', type=str, default=None, help='dd_dev flt dir')

	parser.add_argument('--gec_train_path_src', type=str, default=None, help='gec_train src dir')
	parser.add_argument('--gec_train_path_tgt', type=str, default=None, help='gec_train tgt dir')
	parser.add_argument('--gec_train_path_flt', type=str, default=None, help='gec_train flt dir')
	parser.add_argument('--gec_dev_path_src', type=str, default=None, help='gec_dev src dir')
	parser.add_argument('--gec_dev_path_tgt', type=str, default=None, help='gec_dev tgt dir')
	parser.add_argument('--gec_dev_path_flt', type=str, default=None, help='gec_dev flt dir')

	parser.add_argument('--ddgec_train_path_src', type=str, default=None, help='ddgec_train src dir')
	parser.add_argument('--ddgec_train_path_tgt', type=str, default=None, help='ddgec_train tgt dir')
	parser.add_argument('--ddgec_train_path_flt', type=str, default=None, help='ddgec_train tgt dir')
	parser.add_argument('--ddgec_dev_path_src', type=str, default=None, help='ddgec_dev src dir')
	parser.add_argument('--ddgec_dev_path_tgt', type=str, default=None, help='ddgec_dev tgt dir')
	parser.add_argument('--ddgec_dev_path_flt', type=str, default=None, help='ddgec_dev tgt dir')

	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--restart', type=str, default=None,
		help='model load dir, but start new training schedule')
	parser.add_argument('--load_embedding_src', type=str, default=None, help='pretrained src embedding')
	parser.add_argument('--load_embedding_tgt', type=str, default=None, help='pretrained tgt embedding')

	parser.add_argument('--dd_train_tsv_path', type=str, default=None,
		help='dd train set additional attention key - tsv file')
	parser.add_argument('--dd_dev_tsv_path', type=str, default=None,
		help='dd dev set additional attention key - tsv file')
	parser.add_argument('--gec_train_tsv_path', type=str, default=None,
		help='gec train set additional attention key - tsv file')
	parser.add_argument('--gec_dev_tsv_path', type=str, default=None,
		help='gec dev set additional attention key - tsv file')
	parser.add_argument('--ddgec_train_tsv_path', type=str, default=None,
		help='ddgec train set additional attention key - tsv file')
	parser.add_argument('--ddgec_dev_tsv_path', type=str, default=None,
		help='ddgec dev set additional attention key - tsv file')
	parser.add_argument('--dd_additional_key_size', type=int, default=0, \
		help='dd: additional attention key size: keys = [values, add_feats]')

	# model: shared enc
	parser.add_argument('--embedding_size_enc', type=int, default=200, help='encoder embedding size')
	parser.add_argument('--embedding_size_dec', type=int, default=200, help='decoder embedding size')
	parser.add_argument('--hidden_size_enc', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_bilstm_enc', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--num_unilstm_enc', type=int, default=0, help='number of encoder unilstm layers')

	# model: dd / gec dec
	parser.add_argument('--hard_att', type=str, default='False', help='use hard attention or not')
	parser.add_argument('--dd_num_unilstm_dec', type=int, default=3, \
		help='dd: number of encoder bilstm layers')
	parser.add_argument('--dd_hidden_size_att', type=int, default=10, \
		help='dd: hidden size for bahdanau / hybrid attention')
	parser.add_argument('--dd_att_mode', type=str, default='hybrid', \
		help='dd: attention mechanism mode - bahdanau / hybrid / dot_prod')
	parser.add_argument('--gec_num_bilstm_dec', type=int, default=0, \
		help='gec: number of decoder bilstm layers before attention (part of enc in nmt convention)')
	parser.add_argument('--gec_num_unilstm_dec_preatt', type=int, default=0, \
		help='gec: number of decoder unilstm layers before attention (part of enc in nmt convention)')
	parser.add_argument('--gec_num_unilstm_dec_pstatt', type=int, default=3, \
		help='gec: number of decoder unilstm layers after attention')
	parser.add_argument('--gec_hidden_size_att', type=int, default=10, \
		help='gec: hidden size for bahdanau / hybrid attention')
	parser.add_argument('--gec_att_mode', type=str, default='bahdanau', \
		help='gec: attention mechanism mode - bahdanau / hybrid / dot_prod')
	parser.add_argument('--shared_embed', type=str, default='state', \
		help='shared embedding between enc and gec dec - state / context / TOADD')

	# train
	parser.add_argument('--num_epochs', type=int, default=10,
		help='number of training epochs - used only if ddinit is False')
	parser.add_argument('--dd_num_epochs', type=int, default=10, help='number of dd (step1) training epochs')
	parser.add_argument('--gec_num_epochs', type=int, default=10, help='number of gec (step2) training epochs')
	parser.add_argument('--random_seed', type=int, default=888, help='random seed')
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0, help='ratio of teacher forcing')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--max_grad_norm', type=float, default=1.0, help='optimiser gradient norm clipping: max grad norm')
	parser.add_argument('--residual', type=str, default='False', help='residual connection')
	parser.add_argument('--batch_first', type=str, default='True', help='batch as the first dimension')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--scheduled_sampling', type=str, default='False',
		help='gradually turn off teacher forcing (if True, use teacher_forcing_ratio as the starting point)')
	parser.add_argument('--add_discriminator', type=str, default='False',
		help='whether or not use discriminator for domain adaptation')
	parser.add_argument('--dloss_coeff', type=float, default=0.0,
		help='coefficient of discriminator loss, only used when disc used')

	# training scheduling
	parser.add_argument('--ddreg', type=str, default='False', help='whether or not - use regularisation on dd')
	parser.add_argument('--max_count_no_improve', type=int, default=5,
		help='maxmimum patience count for validation on minibatch not improving before roll back')
	parser.add_argument('--max_count_num_rollback', type=int, default=5,
		help='maxmimum num of rollback before reducing lr by a factor of 2')
	parser.add_argument('--train_mode', type=str, default='separate',
		help='separate: train dd gec separately | end2end: joint fine tuning')
	parser.add_argument('--save_schedule', type=str, default='roll_back',
		help='roll_back: eval per checkpoint_every, with roll back | \
		no_roll_back: check per epoch, w/o rollback \
		[ignores max_count_no_improve/max_count_num_rollback/checkpoint_every]')

	# tweaking
	parser.add_argument('--connect_type', type=str, default='embed',
		help='embed or word connecting dd and gec')
	parser.add_argument('--dd_classifier', type=str, default='False',
		help='whether or not use simple dd classifier instead of attention')
	parser.add_argument('--ptr_net', type=str, default='comb',
		help='whether or not to use pointer network - use attention weights to directly map embedding \
		comb | pure | null: comb combines posterior and att weights (combination ratio is learnt); \
		pure use att weights only; none use posterior only')
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')

	# loss related coeff
	parser.add_argument('--gec_acc_weight', type=float, default=1.0,
		help='determines saving [0.0~1.0]: \
			stopping criteria combines gec and dd acc (1.0 means dd acc ignored)')
	parser.add_argument('--loss_shift', type=str, default='False',
		help='gradually shift loss coeff towards gec loss; not used in end2end mode')
	parser.add_argument('--gec_loss_weight', type=float, default=1.0,
		help='determines gec weight in sgd [0.0~1.0]: sgd combines att, gec and dd loss')
	parser.add_argument('--dd_loss_weight', type=float, default=1.0,
		help='determines dd weight in sgd [0.0~1.0]: sgd combines att, gec and dd loss')
	parser.add_argument('--ddatt_loss_weight', type=float, default=0.0,
		help='determines attloss weight in sgd [0.0~1.0]: sgd combines att, gec and dd loss')
	parser.add_argument('--ddattcls_loss_weight', type=float, default=0.0,
		help='determines attcls loss weight in sgd [0.0~1.0] (att column wise regularisation)')
	parser.add_argument('--att_scale_up', type=float, default=0.0,
		help='scale up att scores before cross entropy loss in regularisation')

	# save and print
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')

	return parser


def main():

	# import pdb; pdb.set_trace()
	# load config
	warnings.filterwarnings("ignore")
	parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Joint Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# set random seed
	if config['random_seed'] is not None:
		set_global_seeds(config['random_seed'])

	# resume or not
	if config['load']:
		print('resuming {} ...'.format(config['load']))
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	elif config['restart']:
		print('restarting from {} ...'.format(config['restart']))
		config_save_dir = os.path.join(config['save'], 'model-restart.cfg')
	else:
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# load vacabulary
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	assert config['load_embedding_src'] == config['load_embedding_tgt'], \
		'src tgt embeddings are different'
	if type(config['load_embedding_src']) == type(None):
		load_embedding = False
	else:
		load_embedding = True
	print('load embedding: {}'.format(load_embedding))

	# load dataset
	time_st1 = time.time()
	if config['train_mode'] == 'separate':

		ddgec_train_set = None
		ddgec_dev_set = None

		# load train set
		dd_train_path_src = config['dd_train_path_src']
		dd_train_path_tgt = config['dd_train_path_tgt']
		dd_train_tsv_path = config['dd_train_tsv_path']
		dd_train_set = Dataset(dd_train_path_src, dd_train_path_tgt,
			path_vocab_src, path_vocab_tgt, seqrev=config['seqrev'],
			tsv_path=dd_train_tsv_path, set_type='dd',
			max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
			use_gpu=config['use_gpu'])
		gec_train_path_src = config['gec_train_path_src']
		gec_train_path_tgt = config['gec_train_path_tgt']
		gec_train_tsv_path = config['gec_train_tsv_path']
		gec_train_set = Dataset(gec_train_path_src, gec_train_path_tgt,
			path_vocab_src, path_vocab_tgt, seqrev=config['seqrev'],
			tsv_path=gec_train_tsv_path, set_type='gec',
			max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
			use_gpu=config['use_gpu'])
		# note: vocab size of dd / gec should agree
		vocab_size_enc = len(dd_train_set.vocab_src)
		vocab_size_dec = len(dd_train_set.vocab_tgt)
		assert vocab_size_enc == vocab_size_dec, \
			'mismatch vocab size: {} - {}'.format(vocab_size_enc, vocab_size_dec)
		vocab_size = vocab_size_enc
		src_word2id = gec_train_set.src_word2id
		tgt_word2id = gec_train_set.tgt_word2id
		src_id2word = gec_train_set.src_id2word

		# load dev set
		if config['dd_dev_path_src'] and config['dd_dev_path_tgt']:
			dd_dev_path_src = config['dd_dev_path_src']
			dd_dev_path_tgt = config['dd_dev_path_tgt']
			dd_dev_tsv_path = config['dd_dev_tsv_path']
			dd_dev_set = Dataset(dd_dev_path_src, dd_dev_path_tgt,
				path_vocab_src, path_vocab_tgt, seqrev=config['seqrev'],
				tsv_path=dd_dev_tsv_path, set_type='dd',
				max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
				use_gpu=config['use_gpu'])
		else:
			dd_dev_set = None
		if config['gec_dev_path_src'] and config['gec_dev_path_tgt']:
			gec_dev_path_src = config['gec_dev_path_src']
			gec_dev_path_tgt = config['gec_dev_path_tgt']
			gec_dev_tsv_path = config['gec_dev_tsv_path']
			gec_dev_set = Dataset(gec_dev_path_src, gec_dev_path_tgt,
				path_vocab_src, path_vocab_tgt, seqrev=config['seqrev'],
				tsv_path=gec_dev_tsv_path, set_type='gec',
				max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
				use_gpu=config['use_gpu'])
		else:
			gec_dev_set = None

	elif config['train_mode'] == 'end2end':

		dd_train_set = None
		gec_train_set = None
		dd_dev_set = None
		gec_dev_set = None

		ddgec_train_path_src = config['ddgec_train_path_src']
		ddgec_train_path_tgt = config['ddgec_train_path_tgt']
		ddgec_train_path_flt = config['ddgec_train_path_flt']
		ddgec_train_tsv_path = config['ddgec_train_tsv_path']
		ddgec_train_set = Dataset(ddgec_train_path_src, ddgec_train_path_tgt,
			path_vocab_src, path_vocab_tgt, seqrev=config['seqrev'],
			tsv_path=ddgec_train_tsv_path, set_type='ddgec',
			max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
			use_gpu=config['use_gpu'], path_flt=ddgec_train_path_flt)

		vocab_size_enc = len(ddgec_train_set.vocab_src)
		vocab_size_dec = len(ddgec_train_set.vocab_tgt)
		assert vocab_size_enc == vocab_size_dec, \
			'mismatch vocab size: {} - {}'.format(vocab_size_enc, vocab_size_dec)
		vocab_size = vocab_size_enc
		src_word2id = ddgec_train_set.src_word2id
		tgt_word2id = ddgec_train_set.tgt_word2id
		src_id2word = ddgec_train_set.src_id2word

		if config['ddgec_dev_path_src'] and config['ddgec_dev_path_tgt'] and config['ddgec_dev_path_flt']:
			ddgec_dev_path_src = config['ddgec_dev_path_src']
			ddgec_dev_path_tgt = config['ddgec_dev_path_tgt']
			ddgec_dev_path_flt = config['ddgec_dev_path_flt']
			ddgec_dev_tsv_path = config['ddgec_dev_tsv_path']
			ddgec_dev_set = Dataset(ddgec_dev_path_src, ddgec_dev_path_tgt,
				path_vocab_src, path_vocab_tgt, seqrev=config['seqrev'],
				tsv_path=ddgec_dev_tsv_path, set_type='ddgec',
				max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
				use_gpu=config['use_gpu'], path_flt=ddgec_dev_path_flt)
		else:
			ddgec_dev_set = None

	else:
		assert False, 'Not implemented mode {}'.format(config['train_mode'])

	time_st2 = time.time()
	print('data loading time: {}'.format(time_st2 - time_st1))

	# construct model
	seq2seq = Seq2Seq(vocab_size_enc, vocab_size_dec,
					embedding_size_enc=config['embedding_size_enc'],
					embedding_size_dec=config['embedding_size_dec'],
					embedding_dropout=config['embedding_dropout'],
					hidden_size_enc=config['hidden_size_enc'],
					hidden_size_dec=config['hidden_size_dec'],
					num_bilstm_enc=config['num_bilstm_enc'],
					num_unilstm_enc=config['num_unilstm_enc'],
					dd_num_unilstm_dec=config['dd_num_unilstm_dec'],
					dd_hidden_size_att=config['dd_hidden_size_att'],
					dd_att_mode=config['dd_att_mode'],
					dd_additional_key_size=config['dd_additional_key_size'],
					gec_num_bilstm_dec=config['gec_num_bilstm_dec'],
					gec_num_unilstm_dec_preatt=config['gec_num_unilstm_dec_preatt'],
					gec_num_unilstm_dec_pstatt=config['gec_num_unilstm_dec_pstatt'],
					gec_hidden_size_att=config['gec_hidden_size_att'],
					gec_att_mode=config['gec_att_mode'],
					shared_embed=config['shared_embed'],
					dropout=config['dropout'],
					residual=config['residual'],
					batch_first=config['batch_first'],
					max_seq_len=config['max_seq_len'],
					batch_size=config['batch_size'],
					load_embedding_src=config['load_embedding_src'],
					load_embedding_tgt=config['load_embedding_tgt'],
					src_word2id=src_word2id,
					tgt_word2id=tgt_word2id,
					src_id2word=src_id2word,
					hard_att=config['hard_att'],
					add_discriminator=config['add_discriminator'],
					dloss_coeff=config['dloss_coeff'],
					use_gpu=config['use_gpu'],
					ptr_net=config['ptr_net'],
					connect_type=config['connect_type'],
					dd_classifier=config['dd_classifier']).to(device)

	time_st3 = time.time()
	print('model init time: {}'.format(time_st3 - time_st2))

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					restart_dir=config['restart'],
					batch_size=config['batch_size'],
					random_seed=config['random_seed'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					learning_rate=config['learning_rate'],
					eval_with_mask=config['eval_with_mask'],
					scheduled_sampling=config['scheduled_sampling'],
					teacher_forcing_ratio=config['teacher_forcing_ratio'],
					use_gpu=config['use_gpu'],
					ddreg=config['ddreg'],
					max_grad_norm=config['max_grad_norm'],
					loss_shift=config['loss_shift'],
					max_count_no_improve=config['max_count_no_improve'],
					max_count_num_rollback=config['max_count_num_rollback'],
					train_mode=config['train_mode'],
					gec_acc_weight=config['gec_acc_weight'],
					gec_loss_weight=config['gec_loss_weight'],
					dd_loss_weight=config['dd_loss_weight'],
					ddatt_loss_weight=config['ddatt_loss_weight'],
					ddattcls_loss_weight=config['ddattcls_loss_weight'],
					att_scale_up=config['att_scale_up'],
					save_schedule=config['save_schedule'])

	# run training
	t.train_mode = config['train_mode']
	if config['train_mode'] == 'separate':
		seq2seq = t.train_separate(dd_train_set, gec_train_set, seq2seq, num_epochs=config['num_epochs'],
			dd_dev_set=dd_dev_set, gec_dev_set=gec_dev_set)
	elif config['train_mode'] == 'end2end':
		seq2seq = t.train_end2end(ddgec_train_set, seq2seq,
			num_epochs=config['num_epochs'], ddgec_dev_set=ddgec_dev_set)


if __name__ == '__main__':
	main()
