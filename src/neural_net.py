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

        self.z3 = np.dot(self.W, self.a2)
        self.y_hat = s(self.z3)

        return self.y_hat

    def predict(self, X):
        self.y_hat = self.forward(X).T
        def convert_prediction(z):
            return 1 if z > .5 else 0

        convert_prediction = np.vectorize(convert_prediction)

        predictions = convert_prediction(self.y_hat)
        return predictions

    def check_error(self, X, y):
        predictions = self.predict(X)
        true_labels = y
        err_rate = benchmark(predictions, true_labels)
        return err_rate

    def squared_loss(self, X, y):

        self.y_hat = self.forward(X)
        loss = 0.5 * sum(sum((y.T - self.y_hat)**2))
        return loss

    def cross_entropy_loss(self, X, y):
        self.y_hat = self.forward(X)
        self.y_hat_log = np.log(self.y_hat)

        first_part = np.dot(y, self.y_hat_log)

        one_minus_labels = np.ones(len(y)) - y
        ln_one_minus_y_hat = np.log(np.ones(len(self.y_hat)) - self.y_hat)
        second_part = np.dot(one_minus_labels, ln_one_minus_y_hat)
        return -1 * (first_part + second_part)


    def squared_loss_derivative(self, X, y):
        """dJ/dV, dJ/dW"""
        # y: (o, n)
        # a2: (h+1, n)
        # delta3:(o,n)
        # z2: (h+1, n)
        self.y_hat = self.forward(X)
        X = add_bias(X)

        delta3 = (self.y_hat - y.T) * s_prime(self.z3)

        # dJdW = np.dot(self.a2, delta3.T)
        dJdW = np.dot(delta3, self.a2.T)

        delta2 = np.dot(self.W.T, delta3) * tanh_prime(self.z2)
        # delta2 = np.dot(delta3.T, self.W) * tanh_prime(self.z2)

        dJdV = np.dot(delta2, X)
        dJdV = np.delete(dJdV,-1,axis=0)
        # print("np.unique(dJdV): {}".format(np.unique(dJdV)))

        return dJdV, dJdW

    def cross_entropy_derivative(self, X, y):
        # y: (o, n)
        # a2: (h+1, n)
        # delta3:(o,n)
        # z2: (h+1, n)
        self.y_hat = self.forward(X)

        delta3 = -(y.T - self.y_hat)

        dJdW = np.dot(self.a2, delta3.T)

        # TODO add np.multiply on outside
        delta2 = np.dot(self.W.T, delta3) * tanh_prime(self.z2)

        dJdV = np.dot(delta2, X)
        # print("np.unique(dJdV): {}".format(np.unique(dJdV)))

        return dJdV, dJdW

    def train(self, X, y, backprop_fn, epsilon = .001, num_iterations = 50000):

        # while criteria, train
        for iteration in range(num_iterations):
            if iteration%10000==0:
                pickle_obj(self)
            if iteration%1000==0:
                print("iteration: {}".format(iteration))
                print("squared loss(X, y): {:.1f}".format(self.squared_loss(X, y)))
                print("error: {:g}%".format(100 * self.check_error(X, y)))
            indices = np.random.choice(len(X), 50)
            mini_batch_data = X[indices]
            mini_batch_labels = y[indices]

            dJdV, dJdW = backprop_fn(mini_batch_data, mini_batch_labels)

            self.W = self.W - epsilon * dJdW
            self.V = self.V - epsilon * dJdV
        return self

# kfold and predict on test set
