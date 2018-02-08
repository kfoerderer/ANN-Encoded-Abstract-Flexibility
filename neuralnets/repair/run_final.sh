#!/bin/bash

DATAFILES="\
../../data/training/*_repair_only_infeasible_15min_bess.npy \
../../data/training/*_repair_only_infeasible_15min_chp.npy \
../../data/training/*_repair_only_infeasible_15min_bess_chp.npy \
"

EXPERIMENTS="\
sacred_predict_zero.py \
sacred_predict_training_mean.py \
sacred_ann_fc2_representation.py \
sacred_cnn_pool1_representation.py \
sacred_rnn_lstm2_representation.py \
"


for datafn in $DATAFILES
do
    for experiment in $EXPERIMENTS
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/repaironlyinfeasible
    done
done
