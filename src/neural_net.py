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
import matplotlib.pyplot as plt
import sklearn
import numpy as np


from pudb import set_trace
from helper import *

# TODO save progress every N iterations by pickling to a file with a timestamp


class NeuralNetwork(object):

    # def __init__(self, X, y, V, W, learning_rate, cost_fn,
    # hidden_activation_fn, output_activation_fn):
    def __init__(self):
        # TODO add bias with np.insert
        self.input_size = 784
        self.output_size = 10
        self.hidden_size = 200
        self.y_hat = None

        i, h, o = self.input_size, self.hidden_size, self.output_size

        self.V = (np.random.randn(h, i) -
                  np.ones((self.hidden_size, self.input_size))) / 30000
        self.V = add_bias(self.V)

        self.W = (np.random.randn(o, h) -
                  np.ones((self.output_size, self.hidden_size))) / 30000
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
        return (-(first_part + second_part))

    def squared_loss_derivative(self, X, y):
        self.y_hat = self.forward(X)
        X = add_bias(X)

        delta3 = (self.y_hat - y.T) * s_prime(self.z3)

        # dJdW = np.dot(self.a2, delta3.T)
        dJdW = np.dot(delta3, self.a2.T)

        delta2 = np.dot(self.W.T, delta3) * tanh_prime(self.z2)
        # delta2 = np.dot(delta3.T, self.W) * tanh_prime(self.z2)

        dJdV = np.dot(delta2, X)
        dJdV = np.delete(dJdV, -1, axis=0)
        # print("np.unique(dJdV): {}".format(np.unique(dJdV)))

        return dJdV, dJdW

    def cross_entropy_derivative(self, X, y):
        self.y_hat = self.forward(X)
        X = add_bias(X)

        delta3 = (self.y_hat - y.T)

        # dJdW = np.dot(self.a2, delta3.T)
        dJdW = np.dot(delta3, self.a2.T)

        delta2 = np.dot(self.W.T, delta3) * tanh_prime(self.z2)
        # delta2 = np.dot(delta3.T, self.W) * tanh_prime(self.z2)

        dJdV = np.dot(delta2, X)
        dJdV = np.delete(dJdV, -1, axis=0)
        # print("np.unique(dJdV): {}".format(np.unique(dJdV)))

        return dJdV, dJdW

    def train(self, X, y, backprop_fn, epsilon=.001, num_iterations=50000, batch_size=400):
        if backprop_fn == self.cross_entropy_derivative:
            loss_fn = self.cross_entropy_loss
        else:
            loss_fn = self.squared_loss
        indep = []
        dep_error = []
        dep_accuracy = []
        # while criteria, train
        for iteration in range(num_iterations):

            if iteration % 10000 == 0:
                pickle_obj(self)
#
            if iteration % 1000 == 0:
                indep.append(iteration)
                print("iteration: {}".format(iteration))
                print("batch size: {}".format(batch_size))
#
                # loss = loss_fn(X, y)
                # print("loss: {:.1f}".format(loss))
                # dep_error.append(loss)

                error_rate = self.check_error(X, y)
                print("error: {:g}%".format(100 * error_rate))
                dep_accuracy.append(error_rate)
                print("\n")

            indices = np.random.choice(len(X), batch_size)
            mini_batch_data = X[indices]
            mini_batch_labels = y[indices]

            dJdV, dJdW = backprop_fn(mini_batch_data, mini_batch_labels)

            self.W = self.W - epsilon * dJdW
            self.V = self.V - epsilon * dJdV

        def plot_pts(indep, dep, title="", x_title="", y_title="", fig_name="error"):
            fig = plt.figure()
            plt.scatter(indep, dep)
            fig.suptitle(title)
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            fig.savefig('./fig/{}_{}.png'.format(fig_name, timestamp()))

        try:
            plot_pts(indep, dep_accuracy,
                     title="iterations vs classification error rate")
            plot_pts(indep, dep_error, title="iterations vs training error",
                     x_title='iterations', y_title='training error')
        except:
            return self

        return self

# kfold and predict on test set
