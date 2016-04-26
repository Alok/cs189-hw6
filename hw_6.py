#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import scipy as sp
from scipy import io
import pickle
import datetime
import sklearn
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.cross_validation import KFold



def sigmoid(x):
    if x < -10:
        return 0.0005
    if x > 10:
        return 0.9995
    value = 1 / (1 + math.exp(-x))
    return value

s = np.vectorize(sigmoid)

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

s_prime = np.vectorize(sigmoidDerivative)

# neural network methods
class NeuralNetwork(object):
    def __init__(self, name, testing=False, sizes=None):
        self.name = name
        self.input_size = 785
        self.hidden_size = 201
        self.output_size = 10
        self.testing = testing
        if testing:
            self.input_size = sizes[0]
            self.hidden_size = sizes[1]
            self.output_size = sizes[2]

        self.V = np.random.rand(self.hidden_size, self.input_size) - np.ones((self.hidden_size, self.input_size))
        self.V = self.V/30000.0

        self.W = np.random.rand(self.output_size, self.hidden_size) - np.ones((self.output_size, self.hidden_size))
        self.W = self.W/30000.0

    def forward(self, inputs):
        if not self.testing and len(inputs) > 1:
            inputs = np.insert(inputs, 784, np.ones(len(inputs)), axis=1)

        self.Vforward = np.dot(self.V, inputs.T)
        self.tanHforward = np.tanh(self.Vforward)
        self.Wforward = np.dot(self.W, self.tanHforward)

        #print "Vforward"
        #print self.Vforward
        #print inputs

        #print "tanHforward"
        #print self.tanHforward

        #print "Wforward"
        #print self.Wforward
        print (s(self.Wforward))
        return s(self.Wforward)

    def meansquaredloss(self, inputs, labels):
        yHat = self.forward(inputs)
        loss = 0.5 * sum(sum((labels.T - yHat)**2))
        return loss

    def backpropMeanSquared(self, inputs, labels):
        yHat = self.forward(inputs)
        if not self.testing:
            inputsWAddedCol = np.insert(inputs, 784, 1, axis=1)
        else:
            inputsWAddedCol = inputs

        sigmoidAndLossDeriv = np.multiply(-(labels.T-yHat), s_prime(self.Wforward))

        dJdW = np.dot(self.tanHforward, sigmoidAndLossDeriv.T)

        tanhDeriv = np.dot(self.W.T, sigmoidAndLossDeriv) * s_prime(self.Vforward)

        dJdV = np.dot(tanhDeriv, inputsWAddedCol)
        print("dJdV: {}".format(dJdV))
        print("np.unique(dJdV): {}".format(np.unique(dJdV)))
        return dJdW, dJdV

    def crossEntropyLoss(self, inputs, labels):
        yHat = self.foward(inputs)
        yHatLn = np.log(yHat)

        firstPart = np.dot(labels, yHatLn)

        oneMinusLabels = np.ones(len(labels)) - labels
        lnOneMinusyHat = np.log(np.ones(len(yHat)) - yHat)
        secondPart = np.dot(oneMinusLabels, lnOneMinusyHat)
        return -1 * (firstPart + secondPart)

    def backpropCrossEntropyLoss(self, inputs, labels):
        yHat = self.forward(inputs)
        if not self.testing:
            inputsWAddedCol = np.insert(input, 784, np.zeros(len(inputs)), axis=1)
        else:
            inputsWAddedCol = inputs
        # to finish


    def trainNeuralNetwork(self, images, labels, epsilon, numIterations, backpropfn):
        for iteration in range(numIterations):
            if iteration%100 == 0:
                self.saveToPickle()
                print ("iteration: " + str(iteration) + " and error: " + str(self.meansquaredloss(images, labels)))
            indices = np.random.choice(len(images), 50)
            mini_batch_data = images[indices]
            mini_batch_labels = labels[indices]
            dJdW, dJdV = backpropfn(mini_batch_data, mini_batch_labels)
            self.W = self.W - epsilon * dJdW.T
            self.V = self.V - epsilon * dJdV

    def predictNeuralNetwork(self, images):
        predictions = self.forward(images)
        return np.argmax(predictions, axis=0)

    def saveToPickle(self):
        filename = './pickles/' + self.name + '.pickle'
        pickle.dump(self, open(filename, "wb" ))


# In[17]:

def getDigits(filename):
    mat_file = sp.io.loadmat(filename)
    labels = mat_file['train_labels']
    enc = OneHotEncoder()
    enc.fit(labels)
    transformedLabels = enc.transform(labels).toarray()
    data = mat_file['train_images']
    data = data.reshape(784, 60000)
    data = data.reshape(28*28, 1, 60000).squeeze().T

    return data, transformedLabels, labels.ravel()


# In[ ]:
loop = False

data, transformedLabels, originalLabels = getDigits("dataset/train.mat")
kf = KFold(len(data), n_folds=2, shuffle=True)
lambdas = [0.01, 0.001, 0.0001]
if loop:
    for lmbda in lambdas:
        incomplete = True
        for train_index, test_index in kf:
            if incomplete:
                new_train_x = np.delete(data, test_index, 0)
                new_train_y = np.delete(transformedLabels, test_index, 0)
                new_test_x = np.delete(data, train_index, 0)
                new_test_y = np.delete(originalLabels, train_index, 0)
                net = NeuralNetwork("testnetwork")
                net.trainNeuralNetwork(new_train_x, new_train_y, 0.001, 5000, net.backpropMeanSquared)
                results = net.predictNeuralNetwork(new_test_x)
                numCorrect = np.sum(results == new_test_y)
                print (lmbda)
                print("numCorrect: {}".format(numCorrect))
                print ("accuracy rate: " + str(numCorrect * 1.0 / 30000))
                incomplete = False


net =  NeuralNetwork('testnetwork')
net.forward(data)
net.backpropMeanSquared(data, transformedLabels)

