#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

# Try to set reproducible results - sacred does not seem to guarantee that on GPUs as of now
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
#import numpy as np
#import tensorflow as tf
#import random as rn
#import os
#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(1924)
#rn.seed(1924)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#from keras import backend as K
#tf.set_random_seed(1924)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
#

from sacred import Experiment
from sacred.observers import FileStorageObserver

import tempfile, shutil

import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, History

ex = Experiment('sacred_training_mean')
#ex.observers.append(FileStorageObserver.create('data/experiments'))

@ex.config
def my_config():
    dataset_filename = '../../data/training/2018-01-10T23:43:00_price-based_15min_bess_chp.npy'
    test_split = 0.2
    seed = 1924

@ex.capture
def load_data(dataset_filename, test_split):
    data = np.load(dataset_filename)
    train_test_split = int(data.shape[0] * (1.0 - test_split))

    train_x = data[0:train_test_split, 0:-96].astype('float32')
    train_y = data[0:train_test_split, -96:].astype('float32')

    test_x = data[train_test_split:, 0:-96].astype('float32')
    test_y = data[train_test_split:, -96:].astype('float32')

    return (train_x, train_y, test_x, test_y)

@ex.automain
def my_experiment(dataset_filename):
    # Load Data
    (train_x, train_y, test_x, test_y) = load_data()

    # Print some statistics of the data
    print("loaded dataset {}".format(dataset_filename))
    print("training data has shape {} -> {} ; mean {} var {}".format(train_x.shape, train_y.shape, np.mean(train_y), np.std(train_y)))

    train_mean =  np.mean(train_y)
    pred_y = np.tile([train_mean], (test_y.shape[0], test_y.shape[1]))

    rmse = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
    ex.info['rmse'] = rmse
    
    mae = metrics.mean_absolute_error(test_y, pred_y)
    ex.info['mae'] = mae
    
    return rmse
