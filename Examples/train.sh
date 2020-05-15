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

# ============== mode ===============
train_mode=separate
dd_num_epochs=0
gec_num_epochs=0
num_epochs=20

# ============== dd gec connect ===============
dd_classifier=False
connect_type='embed' # 'wordhard' | 'wordsoft' | 'embed' | 'prob'
shared_embed='context' # context | state_tgt | state #only used with rnn+embed
gec_acc_weight=1.0
gec_loss_weight=1.0
dd_loss_weight=1.0
ptr_net=null
ddatt_loss_weight=0.0
ddattcls_loss_weight=0.0
savedir=models-v8/debug/

# ============== schedule ===============
batch_size=100
max_seq_len=32
# print_every=1000
# checkpoint_every=17000
print_every=10
checkpoint_every=30

# ============== dd ===============
dd_train_path_src=lib/swbd+clc_corrupt-new/swbd/train.txt.dsf
dd_train_path_tgt=lib/swbd+clc_corrupt-new/swbd/train.txt
dd_train_tsv_path=lib/swbd+clc_corrupt-new/swbd/train.tsv.prob
# dd_dev_path_src=lib/swbd+clc_corrupt-new/swbd/valid.txt.dsf
# dd_dev_path_tgt=lib/swbd+clc_corrupt-new/swbd/valid.txt
# dd_dev_tsv_path=lib/swbd+clc_corrupt-new/swbd/valid.tsv.prob

# dd_train_path_src=lib/swbd-keepfiller/train.txt.dsf
# dd_train_path_tgt=lib/swbd-keepfiller/train.txt
# dd_train_tsv_path=lib/swbd-keepfiller/train.tsv
# dd_dev_path_src=lib/swbd-keepfiller/valid.txt.dsf
# dd_dev_path_tgt=lib/swbd-keepfiller/valid.txt
# dd_dev_tsv_path=lib/swbd-keepfiller/valid.tsv

dd_dev_path_src=None
dd_dev_path_tgt=None
dd_dev_tsv_path=None

# ============== gec ===============
gec_train_path_src=lib/clc/clc-train.src
gec_train_path_tgt=lib/clc/clc-train.tgt
# gec_dev_path_src=lib/clc/clc-valid.src
# gec_dev_path_tgt=lib/clc/clc-valid.tgt
gec_dev_path_src=None
gec_dev_path_tgt=None

$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/spoken-gec/train.py \
	--dd_train_path_src $dd_train_path_src \
	--dd_train_path_tgt $dd_train_path_tgt \
	--dd_train_tsv_path $dd_train_tsv_path \
	--dd_dev_path_src $dd_dev_path_src \
	--dd_dev_path_tgt $dd_dev_path_tgt \
	--dd_dev_tsv_path $dd_dev_tsv_path \
	--gec_train_path_src $gec_train_path_src \
	--gec_train_path_tgt $gec_train_path_tgt \
	--gec_dev_path_src $gec_dev_path_src  \
	--gec_dev_path_tgt $gec_dev_path_tgt \
	--path_vocab_src lib/vocab/clctotal+swbd.min-count4.en \
	--path_vocab_tgt lib/vocab/clctotal+swbd.min-count4.en \
	--load_embedding_src lib/embeddings/glove.6B.200d.txt \
	--load_embedding_tgt lib/embeddings/glove.6B.200d.txt \
	--embedding_size_enc 200 \
	--embedding_size_dec 200 \
	--dd_classifier $dd_classifier \
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
	--shared_embed context \
	--random_seed 25 \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--residual True \
	--batch_first True \
	--eval_with_mask False \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--learning_rate 0.001 \
	--max_grad_norm 1.0 \
	--embedding_dropout 0.0 \
	--dropout 0.2 \
	--num_epochs $num_epochs \
	--dd_num_epochs $dd_num_epochs \
	--gec_num_epochs $gec_num_epochs \
	--train_mode $train_mode \
	--use_gpu True \
	--seqrev False \
	--loss_shift False \
	--ddreg False \
	--save_schedule roll_back \
	--max_count_no_improve 5 \
	--max_count_num_rollback 5 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--gec_acc_weight $gec_acc_weight \
	--gec_loss_weight $gec_loss_weight \
	--dd_loss_weight $dd_loss_weight \
	--ddatt_loss_weight $ddatt_loss_weight \
	--ddattcls_loss_weight $ddattcls_loss_weight \
	--ptr_net $ptr_net \
	--dd_additional_key_size 0 \
	--save $savedir \
