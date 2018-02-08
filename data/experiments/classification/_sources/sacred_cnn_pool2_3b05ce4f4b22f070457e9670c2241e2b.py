#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
from sacred.observers import FileStorageObserver

import tempfile, shutil

import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn import metrics

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Conv1D, Concatenate, Flatten, MaxPooling1D, Dropout
from keras.callbacks import ModelCheckpoint, History, EarlyStopping

ex = Experiment()
#ex.observers.append(FileStorageObserver.create('data/experiments'))

@ex.config
def my_config():
    dataset_filename = '../../data/training/2018-01-11T18:58:00_classification_15min_bess_chp.npy'
    test_split = 0.2
    validation_split = 0.1
    epochs = 5000
    batch_size = 1024
    verbose_training = 0
    verbose_modelcheckpointer = 0
    seed = 1924
    rnn_units = 20

@ex.capture
def build_model(input_shape,rnn_units):
    l0 = Input(shape=input_shape)

    l11 = Lambda(lambda x: x[:, :5, 0])(l0)
    l12 = Lambda(lambda x: x[:, 5:, :])(l0)

    x = l12
    x = Conv1D(48, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(48, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(24, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Concatenate()([x, l11])
    x = Dense(12, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=l0, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam')
    model.summary()
    return model

@ex.capture
def evaluate_model(model, test_x, test_y):
    pred_y = model.predict(test_x.reshape((test_x.shape[0], test_x.shape[1], 1)))
    pred_y = (pred_y > 0.5).astype('float32').reshape(pred_y.shape[0])
    report = metrics.classification_report(test_y[:, 0], pred_y, digits=4)
    print(report)
    ex.info['classification_report'] = report
    confusion_matrix = metrics.confusion_matrix(test_y[:, 0], pred_y)
    print(confusion_matrix)
    ex.info['confusion_matrix'] = confusion_matrix

    pred_test_y = np.append(pred_y.reshape(pred_y.shape[0], 1), test_y, axis=1)
    src_0_pred_test_y = pred_test_y[pred_test_y[:, 2] == 0.]
    src_1_pred_test_y = pred_test_y[pred_test_y[:, 2] == 1.]
    print("Source 0:")
    report = metrics.classification_report(src_0_pred_test_y[:, 1], src_0_pred_test_y[:, 0], digits=4)
    print(report)
    ex.info['classification_report_src_0'] = report
    print("Source 1:")
    report = metrics.classification_report(src_1_pred_test_y[:, 1], src_1_pred_test_y[:, 0], digits=4)
    print(report)
    ex.info['classification_report_src_1'] = report

    return metrics.f1_score(test_y[:,0],pred_y)

@ex.capture
def load_data(dataset_filename, test_split):
    data = np.load(dataset_filename)
    train_test_split = int(data.shape[0] * (1.0 - test_split))

    train_x = data[0:train_test_split, 0:-2].astype('float32')
    train_y = data[0:train_test_split, -2:-1].astype('float32')

    test_x = data[train_test_split:, 0:-2].astype('float32')
    test_y = data[train_test_split:, -2:].astype('float32')

    return (train_x, train_y, test_x, test_y)

@ex.automain
def my_experiment(dataset_filename, validation_split, epochs, batch_size, verbose_training, verbose_modelcheckpointer):
    # Load Data
    (train_x, train_y, test_x, test_y) = load_data()

    # Print some statistics of the data
    print("loaded dataset {}".format(dataset_filename))
    print("training data has shape {} ; {} feasible and {} unfeasible".format((train_y==1.0).shape, (train_y==1.0).sum(), (train_y==0.0).sum()))

    tempdir = tempfile.mkdtemp()

    input_shape = (train_x.shape[1],1)
    model = build_model(input_shape)
    checkpointer = ModelCheckpoint(filepath=tempdir+'/bestmodel.hdf5', verbose=verbose_modelcheckpointer, save_best_only=True)
    history = model.fit(train_x.reshape((train_x.shape[0], train_x.shape[1], 1)), train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[checkpointer, EarlyStopping(patience=50)],
                        verbose=verbose_training)
    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title(ex.get_experiment_info()['name'] + " on dataset " + dataset_filename)
    plt.savefig(tempdir+'/history.png')
    plt.close()
    ex.info['keras_history'] = history.history

    ex.add_artifact(tempdir+'/history.png', name='history.png')
    ex.add_artifact(tempdir+'/bestmodel.hdf5', name='model.hdf5')

    shutil.rmtree(tempdir)

    return evaluate_model(model, test_x, test_y)
