#!/bin/bash

DATAFILES="\
../../data/training/*_classification_15min_bess_chp.npy \
../../data/training/*_classification_15min_bess.npy \
../../data/training/*_classification_15min_chp.npy \
"

EXPERIMENTS="\
sacred_ann_fc2.py \
sacred_cnn_pool2.py \
sacred_rnn_lstm2.py \
"

for datafn in $DATAFILES
do
    for experiment in $EXPERIMENTS
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/classification
    done
done
