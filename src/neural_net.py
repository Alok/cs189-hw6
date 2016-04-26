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

        self.X = add_bias(X)
        self.Y = Y

        self.V = (np.random.randn(self.hidden_size, self.input_size) - np.ones((self.hidden_size, self.input_size))) /30000
        self.V = add_bias(self.V)

        # self.W = np.random.randn(self.output_size, self.hidden_size)/30000
        self.W = (np.random.randn(self.output_size, self.hidden_size) - np.ones((self.output_size, self.hidden_size))) /30000
        self.W = add_bias(self.W)

    # def forward(self, X):
    #     X_bias = add_bias(X)
    #     self.z2 = np.dot(X_bias, self.V.T)
    #     self.a2 = tanh(self.z2)
    #     self.a2 = add_bias(self.a2)
    #
    #     self.z3 = np.dot(self.a2, self.W.T)
    #
    #     y_hat = s(self.z3)
    #
    #     return y_hat

    def forward(self):
        self.z2 = np.dot(self.X, self.V.T)

        # TODO issues with tanh
        self.a2 = tanh(self.z2)
        print("self.a2 after tanh: {}".format(self.a2))
        self.a2 = add_bias(self.a2)

        self.z3 = np.dot(self.a2, self.W.T)
        # print("self.z3: {}".format(self.z3))
        self.y_hat = s(self.z3)
        # print("y_hat: {}".format(self.y_hat))

        return self.y_hat


    def predict(self):
        predictions = self.forward()

        def convert_prediction(z):
            return 1 if z > .5 else 0

        # predictions = [convert_prediction(i) for i in predictions[j] for j in range(len(predictions))]
        # predictions = [convert_prediction(elem) for elem in row for row in predictions]
        # predictions = [convert_prediction(elem) for row in predictions for elem in row]
        predictions = np.vectorize(convert_prediction)(predictions)
        # predictions = np.array([convert_prediction(elem) for row in predictions for elem in row]).reshape(60000,10)
        print("predictions.shape: {}".format(predictions.shape))
        print("predictions: {}".format(predictions))
        return predictions

    def check_error(self):
        # TODO init self_predictions to be None and then do "if" checks to run predict if it is None else use cached value
        true_labels = self.Y
        print("true_labels.shape: {}".format(true_labels.shape))
        predictions = self.predict()
        err_rate = benchmark(predictions, true_labels)
        print("err_rate: {}".format(err_rate))
        return err_rate


