#!/usr/bin/env python3
# encoding: utf-8

import math
import time
import random
import os
import pickle
import sys
import subprocess
import functools
import itertools

import scipy
import scipy.stats
import matplotlib as plt
import sklearn
import numpy as np

from scipy.special import expit
from numpy import tanh
from pudb import set_trace
from sklearn.utils import shuffle


# To make unique filenames to save pickles.

def timestamp():
    return str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec)


def benchmark(pred_labels, true_labels):
    errors = [i for i in range(len(pred_labels)) if not np.array_equal(pred_labels[i], true_labels[i])]
    err_rate = len(errors) / len(true_labels)
    return err_rate


def create_cross_validation_sets(data, labels, k=50000):
    shuffled_data, shuffled_labels = shuffle(data, labels)

    t = k

    left_data = shuffled_data[:t]
    right_data = shuffled_data[t:]

    left_labels = shuffled_labels[:t]
    right_labels = shuffled_labels[t:]

    validation_set = (right_data, right_labels)
    training_set = (left_data, left_labels)

    return (training_set, validation_set)


def pickle_obj(obj):
    """saves an object in 2 files, one with a (almost certainly) unique timestamp"""
    pickle_file_unique = open('./pickles/' + timestamp() + '.pickle', 'wb')
    pickle_file = open('./pickles/' + '.pickle', 'wb')
    pickle.dump(obj, pickle_file)
    pickle.dump(obj, pickle_file_unique)
    pickle_file.close()
    pickle_file_unique.close()


def add_bias(arr, bias_term=1):
    return np.insert(arr, len(arr[0]), bias_term, axis=1)

def add_bias_row(arr, bias_term=1):
    return np.insert(arr, len(arr), 1, axis=0)

def s(x, derivative=False):
    if derivative:
        return np.maximum(1e-8, expit(x)) * (np.maximum(1e-8, 1 - expit(x)))
    else:
        return np.maximum(1e-8, expit(x))

def s_prime(x):
    return np.maximum(1e-8, expit(x)) * (np.maximum(1e-8, 1 - expit(x)))


def tanh(x, derivative=False):
    if derivative:
        return (1 - (np.tanh(x)**2))
    else:
        return np.tanh(x)

def tanh_prime(x):
    return (1 - (np.tanh(x)**2))

def unbinarize_predictions(y):
    return [np.argmax(i) for i in y]
