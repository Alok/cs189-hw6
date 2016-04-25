#!/usr/bin/env python3
# encoding: utf-8

import math
import time
import random
import os
import sys
import subprocess
import functools
import itertools

import scipy
import scipy.stats
import matplotlib as plt
import sklearn
import numpy as np

from pudb import set_trace
from sklearn.utils import shuffle


# To make unique filenames to save pickles.

def timestamp():
    return str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec)


def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

def create_cross_validation_sets(data, labels, k = 2/3):
    shuffled_data, shuffled_labels = shuffle(data, labels)

    t = math.floor(len(data) * k)

    left_data     = shuffled_data[:t]
    right_data  = shuffled_data[t:]

    left_labels     = shuffled_labels[:t]
    right_labels  = shuffled_labels[t:]

    validation_set = (right_data, right_labels)
    training_set = (left_data, left_labels)

    return (training_set, validation_set)

def pickle_tree(obj, name):
    """saves an object in 2 files, one with a (almost certainly) unique timestamp"""
    pickle_file_unique = open('./pickles/' + name + timestamp() + '.pickle', 'wb')
    pickle_file = open('./pickles/' + name + '.pickle', 'wb')
    pickle.dump(obj, pickle_file)
    pickle.dump(obj, pickle_file_unique)
    pickle_file.close()
    pickle_file_unique.close()

