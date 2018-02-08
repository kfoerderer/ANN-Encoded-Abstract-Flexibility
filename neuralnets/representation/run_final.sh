#!/bin/bash

DATAFILES="\
../../data/training/*_representation_CS_15min_bess.npy \
../../data/training/*_representation_CS_15min_chp.npy \
../../data/training/*_representation_MT_15min_bess.npy \
../../data/training/*_representation_MT_15min_chp.npy \
"

EXPERIMENTS="\
sacred_predict_training_mean.py \
sacred_ann_fc2_representation.py \
sacred_cnn_pool1_representation.py \
"

for datafn in $DATAFILES
do
    for experiment in $EXPERIMENTS
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/representation
    done
done

DATAFILES="\
../../data/training/2018-01-18T17:57:00_representation_MT_15min_bess.npy \
../../data/training/2018-01-18T17:57:00_representation_MT_15min_chp.npy \
"

EXPERIMENTS="\
sacred_rnn_lstm2_mt_representation.py \
"

for datafn in $DATAFILES
do
    for experiment in $EXPERIMENTS
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/representation
    done
done

DATAFILES="\
../../data/training/2018-01-18T17:57:00_representation_CS_15min_bess.npy \
../../data/training/2018-01-18T17:57:00_representation_CS_15min_chp.npy \
"

EXPERIMENTS="\
sacred_rnn_lstm2_representation.py \
"

for datafn in $DATAFILES
do
    for experiment in $EXPERIMENTS
    do
        python $experiment with dataset_filename="$datafn" -F ../../data/experiments/representation
    done
done