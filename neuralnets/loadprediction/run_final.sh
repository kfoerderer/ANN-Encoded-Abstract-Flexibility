#!/bin/bash

DATAFILES="\
../../data/training/*_price-based_15min_bess_chp.npy \
../../data/training/*_price-based_15min_bess.npy \
../../data/training/*_price-based_15min_chp.npy \
"

EXPERIMENTS="\
sacred_predict_zero.py \
sacred_predict_training_mean.py \
sacred_ann_fc2_prices.py \
sacred_cnn_pool1_prices.py \
sacred_rnn_lstm2_prices.py \
"

for datafn in $DATAFILES
do
    for experiment in $EXPERIMENTS
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/loadprediction
    done
done
