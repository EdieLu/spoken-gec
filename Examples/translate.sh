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

# ----- files ------
# fname=test_swbdtrain_dd
# ftst=lib/swbd+clc_corrupt-new/swbd/train.txt.dsf.h7000
# seqlen=90
# fname=test_swbd_dd
# ftst=lib/swbd+clc_corrupt-new/swbd/test.txt.dsf
# seqlen=90
# fname=test_nict
# ftst=lib/nict-eval/nict.gec.txt.nodot
# seqlen=85
# fname=test_dtal
# ftst=lib/dtal-eval-ytl/dtal-GEM4.gec.txt.nodot
# seqlen=165
# fname=test_eval3_segauto
# ftst=lib/eval3-eval/A8-segauto.src.nodot
# seqlen=145

# fname=test_clctrain
# ftst=lib/clc/clc-tail2000.src
# seqlen=140
fname=test_clc
ftst=lib/clc/clc-test.src
seqlen=120
# fname=test_clcvalid
# ftst=lib/clc/clc-valid.src
# seqlen=150

# ----- models ------
# model=models-v8/dd-cls-v002
# ckpt=2020_03_01_12_35_46
# ddgec_mode=dd
# model=models-v8/dd-trs-v003
# ckpt=2020_03_01_01_27_48
# ddgec_mode=dd
model=models-v8/sep-emb-v005
ckpt=2020_04_26_10_48_02
ddgec_mode=gec
# model=models-v8/sep-wordhard-v001
# ckpt=2020_03_05_11_32_33
# ddgec_mode=dd
# model=models-v8/sep-seqtag-prob-v001
# ckpt=2020_03_07_02_18_04
# ddgec_mode=gec
# model=models-v8/sep-seqtag-wordhard-v001
# ckpt=2020_03_05_01_17_57
# ddgec_mode=dd


$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v8/translate.py \
    --test_path_src $ftst \
    --test_path_tgt $ftst \
    --path_vocab_src lib/vocab/clctotal+swbd.min-count4.en \
    --path_vocab_tgt lib/vocab/clctotal+swbd.min-count4.en \
    --load $model/checkpoints/$ckpt \
    --test_path_out $model/$fname/$ckpt/ \
    --seqrev False \
    --max_seq_len $seqlen \
    --batch_size 50 \
    --use_gpu True \
    --beam_width 1 \
    --ddgec_mode $ddgec_mode \
    --mode 2 \
