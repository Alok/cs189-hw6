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
from helper import *

# TODO save progress every N iterations by pickling to a file with a timestamp


class NeuralNetwork(object):

    # def __init__(self, X, y, V, W, learning_rate, cost_fn, hidden_activation_fn, output_activation_fn):
    def __init__(self):
        # TODO add bias with np.insert
        self.input_size = 784
        self.output_size = 10
        self.hidden_size = 200
        self.y_hat = None

        i, h, o = self.input_size, self.hidden_size, self.output_size

        self.V = (np.random.randn(h,i) - np.ones((self.hidden_size, self.input_size))) /30000
        self.V = add_bias(self.V)

        self.W = (np.random.randn(o,h) - np.ones((self.output_size, self.hidden_size))) /30000
        self.W = add_bias(self.W)

    def forward(self, X):
        X_biased = add_bias(X)

        self.z2 = np.dot(self.V, X_biased.T)
        self.z2 = add_bias_row(self.z2)

        # TODO issues with tanh, clamp vals?
        self.a2 = tanh(self.z2)
        # self.a2 = add_bias_row(self.a2) # (h+1, n)

        self.z3 = np.dot(self.W, self.a2)
        self.y_hat = s(self.z3)

        return self.y_hat

    def predict(self, X):
        self.y_hat = self.forward(X)
        def convert_prediction(z):
            return 1 if z > .5 else 0

        convert_prediction = np.vectorize(convert_prediction)

        predictions = convert_prediction(self.y_hat)
        return predictions

    def check_error(self, X, y):
        predictions = self.predict(X)
        true_labels = y
        err_rate = benchmark(predictions, true_labels)
        print("err_rate: {}".format(err_rate))
        return err_rate

    def squared_loss_cost(self, X, y):
        self.y_hat = self.forward(X)
        J = .5 * np.sum((y - self.y_hat)**2)
        return J

    def squared_loss_cost_derivative(self, X, y):
        """dJ/dV, dJ/dW"""
        # y: (o, n)
        # a2: (h+1, n)
        # delta3:(o,n)
        # z2: (h+1, n)
        self.y_hat = self.forward(X)

        delta3 = -(y.T - self.y_hat) * s_prime(self.z3)

        dJdW = np.dot(self.a2, delta3.T)

        # TODO add np.multiply on outside
        # delta2 = np.dot(delta3.T, self.W) * tanh_prime(self.z2)
        delta2 = np.dot(self.W.T, delta3) * tanh_prime(self.z2)
        # dJdV = np.dot(X, delta2.T)
        dJdV = np.dot(delta2, X)

        return dJdV, dJdW
