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

    # def __init__(self, X, Y, V, W, learning_rate, cost_fn, hidden_activation_fn, output_activation_fn):
    def __init__(self, X, Y):
        # TODO add bias with np.insert
        self.input_size = 784
        self.output_size = 10
        self.hidden_size = 200
        self.y_hat = None

        self.X = add_bias(X)
        self.Y = Y
        # V -> 200, 784
        # W -> 10, 200

        self.V = (np.random.randn(self.hidden_size, self.input_size) - np.ones((self.hidden_size, self.input_size))) /30000
        self.V = add_bias(self.V)

        self.W = (np.random.randn(self.output_size, self.hidden_size) - np.ones((self.output_size, self.hidden_size))) /30000
        self.W = add_bias(self.W)

    def forward(self, X):
        X_biased = add_bias(X)
        self.z2 = np.dot(X_biased, self.V.T)
        self.z22 = np.dot(self.V, X_biased.T)

        # TODO issues with tanh, clamp vals?
        self.a2 = tanh(self.z2)
        self.a22 = tanh(self.z22)

        self.a2 = add_bias(self.a2)
        self.a22 = add_bias(self.a22)
        print("self.a2: {}".format(self.a2))
        print("self.a22: {}".format(self.a22))
        print(self.a22 == self.a2.T)

        self.z3 = np.dot(self.a2, self.W.T)
        print("self.z3: {}".format(self.z3))
        self.y_hat = s(self.z3)
        print("self.y_hat: {}".format(self.y_hat))

        return self.y_hat

    def predict(self, X):
        if self.y_hat is None:
            self.y_hat = self.forward(X)
        def convert_prediction(z):
            return 1 if z > .5 else 0
        convert_prediction = np.vectorize(convert_prediction)

        predictions = convert_prediction(self.y_hat)
        return predictions

    def check_error(self, X):
        if self.y_hat is None:
            self.y_hat = self.forward(X)
            predictions = self.y_hat
        else:
            predictions = self.predict(X)

        true_labels = self.Y
        err_rate = benchmark(predictions, true_labels)
        print("err_rate: {}".format(err_rate))
        return err_rate

    def squared_loss_cost(self, X, Y):
        self.y_hat = self.forward(X)
        J = .5 * np.sum((Y - self.y_hat)**2)
        return J

    def squared_loss_cost_derivative(self, X, Y):
        """dJ/dV, dJ/dW"""

        self.y_hat = self.forward(X)

        delta3 = np.multiply(-(Y - self.y_hat), s(self.z3, derivative=True))
        dJdW = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W)*tanh(self.z2, derivative=True)
        dJdV = np.dot(X.T, delta2)

        return dJdV, dJdV

