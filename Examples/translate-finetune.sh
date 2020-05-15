#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3

fold=7
modeltype=joint-seq2seq-word-errantdd

# ----- models ------
ddgec_mode=gec
ckpt=2020_03_09_21_43_41
load=models-finetune/v2-dtal/$modeltype/f$fold/checkpoints/$ckpt
pathout=models-finetune/v2-dtal/$modeltype/f$fold/test
seqlen=165
batch_size=30
test_path_src=lib/dtal-eval-ytl/dtal-GEM4.gec.txt.nodot.10fd/f$fold/test.txt

# ddgec_mode=gec
# ckpt=2020_03_09_11_11_11
# load=models-finetune/v2-nict/$modeltype/f$fold/checkpoints/$ckpt
# pathout=models-finetune/v2-nict/$modeltype/f$fold/test
# seqlen=85
# batch_size=70
# test_path_src=lib/nict-eval/nict.gec.txt.nodot.10fd/f$fold/test.txt

$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v8/translate.py \
    --test_path_src $test_path_src \
    --test_path_tgt $test_path_src \
    --path_vocab_src lib/vocab/clctotal+swbd.min-count4.en \
    --path_vocab_tgt lib/vocab/clctotal+swbd.min-count4.en \
    --load $load \
    --test_path_out $pathout \
    --seqrev False \
    --max_seq_len $seqlen \
    --batch_size $batch_size \
    --use_gpu True \
    --beam_width 1 \
    --ddgec_mode $ddgec_mode \
    --mode 2 \
