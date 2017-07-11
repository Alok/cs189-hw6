#!/usr/bin/env python3
# encoding: utf-8

import math
import random
import csv
import os
import sys
import subprocess
import functools
import itertools

import scipy
from scipy.special import expit
import scipy.io as sio
import matplotlib as plt
import sklearn
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale as normalize
from sklearn.cross_validation import KFold
from neural_net import *


# seed random numbers to make calculation
# deterministic (just a good practice)
# np.random.seed(1)

data = sio.loadmat("./dataset/train.mat")['train_images']
test = sio.loadmat("./dataset/test.mat")['test_images']

# test = np.swapaxes(test,0,2)
# test = test.reshape(784,10000)
# test = test.T
# test = normalize(test).astype('float64')

test = np.reshape(test, (10000, -1)) + 0.0
test = normalize(test)

data = data.reshape(784, 60000)
data = data.reshape(28 * 28, 1, 60000).squeeze().T
data = normalize(data).astype('float64')

labels = sio.loadmat("./dataset/train.mat")['train_labels']

enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()


validation_train_x = create_cross_validation_sets(data, labels)[0][0]
validation_train_y = create_cross_validation_sets(data, labels)[0][1]
validation_test_x  = create_cross_validation_sets(data, labels)[1][0]
validation_test_y  = create_cross_validation_sets(data, labels)[1][1]
assert len(validation_train_x) == len(validation_train_y)
assert len(validation_test_x) == len(validation_test_y)

write_csv = True
# write_csv = False

print("====================== Cross Entropy =======================")
cross_neural_net = NeuralNetwork()
cross_neural_net = cross_neural_net.train(data, labels, backprop_fn=cross_neural_net.cross_entropy_derivative, epsilon =0.001, num_iterations = 5000, batch_size=400)
cross_predictions = unbinarize_predictions(cross_neural_net.predict(test))

if write_csv:
    with open('kaggle_mse.csv','w') as fp:
        kaggle_results = [['id', 'Category']] + [[i+1, digit] for i, digit in enumerate(cross_predictions)]
        a = csv.writer(fp, delimiter = ',')
        a.writerows(kaggle_results)

print("====================== MSE =======================")

mse_neural_net = NeuralNetwork()
mse_neural_net = mse_neural_net.train(data, labels, backprop_fn=mse_neural_net.squared_loss_derivative, epsilon =0.001, num_iterations = 50000, batch_size=400)
mse_predictions = unbinarize_predictions(mse_neural_net.predict(test))


if write_csv:
    with open('kaggle_cross.csv','w') as fp:
        kaggle_results = [['id', 'Category']] + [[i+1, digit] for i, digit in enumerate(mse_predictions)]
        a = csv.writer(fp, delimiter = ',')
        a.writerows(kaggle_results)

valid_mse_neural_net = NeuralNetwork()
valid_mse_neural_net = valid_mse_neural_net.train(validation_train_x, validation_train_y, backprop_fn=valid_mse_neural_net.squared_loss_derivative, epsilon =1e-3, num_iterations = 250000, batch_size=400)
valid_mse_error = valid_mse_neural_net.check_error(validation_test_x,validation_test_y)
print( "mse error on validation set:%f" % valid_mse_error)


valid_cross_neural_net = NeuralNetwork()
valid_cross_neural_net = valid_cross_neural_net.train(validation_train_x, validation_train_y, backprop_fn=valid_cross_neural_net.cross_entropy_derivative, epsilon =0.001, num_iterations = 250000, batch_size=400)
valid_cross_error = valid_cross_neural_net.check_error(validation_test_x,validation_test_y)
print("cross error on validation set:%f" % valid_cross_error)
