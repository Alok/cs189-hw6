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
import scipy.io as sio
import matplotlib as plt
import sklearn
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from pudb import set_trace


data = sio.loadmat("./dataset/train.mat")['train_images']
data      = data.reshape(784,60000)
data      = data.reshape(28*28, 1, 60000).squeeze().T

labels = sio.loadmat("./dataset/train.mat")['train_labels']

def relabel(l = labels):
    """vectorizes labels
    l[2] = 3 -> [0,0,1]
    :l: TODO
    :returns: TODO

    """
    for i,label in enumerate(l):
        x = label[0]
        l[i] = [0] * 10
        l[i][x] = 1
    return l

r = relabel(labels)

enc =  OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
