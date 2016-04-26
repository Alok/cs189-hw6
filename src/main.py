#!/usr/bin/env python3
# encoding: utf-8

import math
import random
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
from pudb import set_trace
from neural_net import NeuralNetwork

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

data = sio.loadmat("./dataset/train.mat")['train_images']

data = data.reshape(784, 60000)
data = data.reshape(28 * 28, 1, 60000).squeeze().T
# data = normalize(data).astype('float64')

labels = sio.loadmat("./dataset/train.mat")['train_labels']

# vectorize the labels as such:
# if label is 9: return len 10 array such that list[9] = 1 and 0 in every
# other index

enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()

nn = NeuralNetwork(data, labels)
# err = nn.check_error(data)
nn.forward(data)

# cost = nn.squared_loss_cost(data, labels)
# print("cost: {}".format(cost))
# djdv, djdw = nn.squared_loss_cost_derivative(data, labels)
# print("djdv: {}, djdw: {}".format(djdv, djdw))
