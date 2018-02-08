#!/bin/bash

DATAFILES="\
../../data/training/*_repair_15min_bess.npy \
../../data/training/*_repair_15min_chp.npy \
../../data/training/*_repair_15min_bess_chp.npy \
"

EXPERIMENTS="\
sacred_norepair.py \
sacred_predict_zero.py \
sacred_predict_training_mean.py \
sacred_ann_fc_repair.py \
sacred_cnn_fc_repair.py \
sacred_rnn_lstm_repair.py \
"

for experiment in $EXPERIMENTS
do
    for datafn in $DATAFILES
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/repair
    done
done
