#!/bin/bash
#$ -S /bin/bash
echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3

# train base=joint-* (dsf+i->flt+c); finetune=joint-* (dsf+i->flt+c)

# ----- model ----
fold=10
modeltype=joint-seq2seq-word
tag=errantdd
dd_classifier=False
connect_type='wordhard' # 'wordhard' | 'wordsoft' | 'embed' | 'prob'
shared_embed='context'

script=/home/alta/BLTSpeaking/exp-ytl28/local-ytl/spoken-gec/train.py
restart=None

dd_loss_weight=1.0 # whether or not have dd tags
loss_shift=True
gec_loss_weight=1.0
gec_acc_weight=1.0

ddatt_loss_weight=0.0
ddattcls_loss_weight=0.0
dd_additional_key_size=0
ptr_net=null

train_mode=end2end
checkpoint_every=30
print_every=10
learning_rate=0.001
max_seq_len=32

# ----- BULATS -----
num_epochs=30
batch_size=100
eval_with_mask=True
save=models-finetune/debug/
ddgec_train_path_src=lib/dtal-eval-ytl/dtal-GEM4.gec.txt.nodot.10fd/f$fold/train.txt
ddgec_dev_path_src=lib/dtal-eval-ytl/dtal-GEM4.gec.txt.nodot.10fd/f$fold/dev.txt
# ddgec_train_path_flt=lib/dtal-eval-ytl/mandd-dtal-GEM4.gec.txt.nodot.10fd/f$fold/train.txt #REF DD
# ddgec_dev_path_flt=lib/dtal-eval-ytl/mandd-dtal-GEM4.gec.txt.nodot.10fd/f$fold/dev.txt
ddgec_train_path_flt=models-finetune/v2-bases-lib/dd-errant/dtal.txt.10fd/f$fold/train.txt #ERRANT DD
ddgec_dev_path_flt=models-finetune/v2-bases-lib/dd-errant/dtal.txt.10fd/f$fold/dev.txt

ddgec_train_path_tgt=lib/dtal-eval-ytl/dtal-GEM4.gec.tgt.nodot.10fd/f$fold/train.txt
ddgec_dev_path_tgt=lib/dtal-eval-ytl/dtal-GEM4.gec.tgt.nodot.10fd/f$fold/dev.txt

# ddgec_train_tsv_path=lib/dtal-eval-ytl/dtal-GEM4.tsv.lab.prob.10fd/f$fold/train.tsv
# ddgec_dev_tsv_path=lib/dtal-eval-ytl/dtal-GEM4.tsv.lab.prob.10fd/f$fold/dev.tsv
ddgec_train_tsv_path=None
ddgec_dev_tsv_path=None


$PYTHONBIN $script \
	--ddgec_train_path_src $ddgec_train_path_src \
	--ddgec_train_path_flt $ddgec_train_path_flt \
	--ddgec_train_path_tgt $ddgec_train_path_tgt \
	--ddgec_dev_path_src $ddgec_dev_path_src \
	--ddgec_dev_path_flt $ddgec_dev_path_flt \
	--ddgec_dev_path_tgt $ddgec_dev_path_tgt \
	--path_vocab_src lib/vocab/clctotal+swbd.min-count4.en \
	--path_vocab_tgt lib/vocab/clctotal+swbd.min-count4.en \
	--load_embedding_src lib/embeddings/glove.6B.200d.txt \
	--load_embedding_tgt lib/embeddings/glove.6B.200d.txt \
	--embedding_size_enc 200 \
	--embedding_size_dec 200 \
	--hidden_size_enc 200 \
	--hidden_size_dec 200 \
	--num_bilstm_enc 2 \
	--num_unilstm_enc 0 \
	--hard_att False \
	--dd_num_unilstm_dec 4 \
	--dd_hidden_size_att 10 \
	--dd_att_mode hybrid \
	--connect_type $connect_type \
	--gec_num_bilstm_dec 1 \
	--gec_num_unilstm_dec_preatt 0 \
	--gec_num_unilstm_dec_pstatt 4 \
	--gec_hidden_size_att 10 \
	--gec_att_mode bilinear \
	--shared_embed $shared_embed \
	--random_seed 25 \
	--residual True \
	--batch_first True \
	--eval_with_mask $eval_with_mask \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--learning_rate $learning_rate \
	--max_grad_norm 1.0 \
	--embedding_dropout 0.0 \
	--dropout 0.2 \
	--dd_num_epochs 0 \
	--gec_num_epochs 0 \
	--ddreg False \
	--use_gpu True \
	--seqrev False \
	--train_mode $train_mode \
	--save_schedule roll_back \
	--max_count_no_improve 3 \
	--max_count_num_rollback 5 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--num_epochs $num_epochs \
	--loss_shift $loss_shift \
	--att_scale_up 0 \
	--ptr_net $ptr_net \
	--gec_acc_weight $gec_acc_weight \
	--gec_loss_weight $gec_loss_weight \
	--dd_loss_weight $dd_loss_weight \
	--ddatt_loss_weight $ddatt_loss_weight \
	--ddattcls_loss_weight $ddattcls_loss_weight \
	--dd_additional_key_size $dd_additional_key_size \
	--ddgec_train_tsv_path $ddgec_train_tsv_path \
	--ddgec_dev_tsv_path $ddgec_dev_tsv_path \
	--save $save \
	--restart $restart \
