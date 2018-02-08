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
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, History, EarlyStopping

ex = Experiment()
#ex.observers.append(FileStorageObserver.create('data/experiments'))

@ex.config
def my_config():
    dataset_filename = '../../data/training/2018-01-18T17:57:00_representation_CS_15min_bess.npy'
    test_split = 0.2
    validation_split = 0.1
    epochs = 5000
    batch_size = 1024
    verbose_training = 0
    verbose_modelcheckpointer = 0
    seed = 1924
    scaling_factor_y = 1

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(96, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(96))
    model.compile(loss='mean_squared_error',
                  optimizer='Adam')
    print(model.summary())
    return model

@ex.capture
def evaluate_models(num_seasons, models, test_x_seasons, test_y_seasons, scaling_factor_y):
    pred_y_all = []

    for i in range(num_seasons):
        model = models[i]
        test_x = test_x_seasons[i]
        pred_y = model.predict(test_x) * scaling_factor_y
        pred_y_all.append(pred_y)

    pred_y_all = np.concatenate(pred_y_all, axis=0)
    test_y_all = np.concatenate(test_y_seasons, axis=0)

    rmse = np.sqrt(metrics.mean_squared_error(test_y_all, pred_y_all))
    ex.info['rmse'] = rmse
    
    mae = metrics.mean_absolute_error(test_y_all, pred_y_all)
    ex.info['mae'] = mae
    
    return rmse

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
def my_experiment(dataset_filename, validation_split, epochs, batch_size, verbose_training, verbose_modelcheckpointer, scaling_factor_y):
    # Load Data
    (train_x, train_y, test_x, test_y) = load_data()

    # Print some statistics of the data
    print("loaded dataset {}".format(dataset_filename))
    print("training data has shape {} -> {} ; mean {} var {}".format(train_x.shape, train_y.shape, np.mean(train_y), np.std(train_y)))

    # Split Data
    num_seasons = 1
    train_x_parts = []
    train_y_parts = []
    test_x_parts = []
    test_y_parts = []

    train_x_parts.append(train_x)
    train_y_parts.append(train_y)
    test_x_parts.append(test_x)
    test_y_parts.append(test_y)

    tempdir = tempfile.mkdtemp()

    models = [build_model(input_shape=(train_x_parts[i].shape[1],)) for i in range(num_seasons)]
    checkpointers = [ModelCheckpoint(filepath="{}/bestmodel{}.hdf5".format(tempdir, i), verbose=verbose_modelcheckpointer, save_best_only=True) for i in range(num_seasons)]
    plt.figure(figsize=(12,8))
    histories = []
    for i in range(num_seasons):
        model = models[i]
        checkpointer = checkpointers[i]
        train_x = train_x_parts[i]
        train_y = train_y_parts[i]
        history = model.fit(train_x, train_y/scaling_factor_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=[checkpointer, EarlyStopping(patience=100)],
                            verbose=verbose_training)
        histories.append(history.history)
        plt.plot(history.history['loss'], label='loss_{}'.format(i), ls='dotted')
        plt.plot(history.history['val_loss'], label='val_loss_{}'.format(i))
    plt.legend()
    plt.title(ex.get_experiment_info()['name'] + " on dataset " + dataset_filename)
    plt.savefig(tempdir+'/history.png')
    plt.close()
    ex.info['keras_history'] = histories

    ex.add_artifact(tempdir+'/history.png', name='history.png')
    for i in range(num_seasons):
        ex.add_artifact("{}/bestmodel{}.hdf5".format(tempdir, i), name="model{}.hdf5".format(i))

    shutil.rmtree(tempdir)

    return evaluate_models(num_seasons, models, test_x_parts, test_y_parts)
