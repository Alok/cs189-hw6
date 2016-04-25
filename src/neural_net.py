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
import matplotlib as plt
import sklearn
import numpy as np

from pudb import set_trace

# TODO save progress every N iterations by pickling to a file with a timestamp

class NeuralNet(object):


    def __init__(self, data, labels, learning_rate, cost_fn):




