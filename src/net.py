#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

class NeuralNetwork(object):
    def __init__(self, X, y, parameters):
        self.X=X
        self.y=y
        #Expect parameters to be a tuple of the form:
        #    ((n_input,0,0), (n_hidden_layer_1, f_1, f_1'), ...,
        #     (n_hidden_layer_k, f_k, f_k'), (n_output, f_o, f_o'))
        self.n_layers = len(parameters)
        self.sizes = [layer[0] for layer in parameters]
        self.fs =[layer[1] for layer in parameters]
        self.fprimes = [layer[2] for layer in parameters]
        self.build_network()

    def build_network(self):
        self.weights=[]
        self.biases=[]
        self.inputs=[]
        self.outputs=[]
        self.errors=[]
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            self.weights.append(np.random.normal(0,1, (m,n)))
            self.biases.append(np.random.normal(0,1,(m,1)))
            self.inputs.append(np.zeros((n,1)))
            self.outputs.append(np.zeros((n,1)))
            self.errors.append(np.zeros((n,1)))
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n,1)))
        self.outputs.append(np.zeros((n,1)))
        self.errors.append(np.zeros((n,1)))

    def feedforward(self, x):
        k=len(x)
        x.shape=(k,1)
        self.inputs[0]=x
        self.outputs[0]=x
        for i in range(1,self.n_layers):
            self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])+self.biases[i-1]
            self.outputs[i]=self.fs[i](self.inputs[i])
        y=self.outputs[-1]
        return y

    def update_weights(self,x,y):
        output = self.feedforward(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(output-y)
        n=self.n_layers-2
        for i in range(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
            self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]
        self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])



    def train(self,n_iter, learning_rate=1):
        self.learning_rate=learning_rate
        n=self.X.shape[0]
        for repeat in range(n_iter):
            index=list(range(n))
            np.random.shuffle(index)
            for row in index:
                x=self.X[row]
                y=self.y[row]
                self.update_weights(x,y)


    def predict_x(self, x):
        return self.feedforward(x)

    def predict(self, X):
        n = len(X)
        m = self.sizes[-1]
        ret = np.ones((n,m))
        for i in range(len(X)):
            ret[i,:] = self.feedforward(X[i])
        return ret

from scipy.special import expit

def logistic(x):
    return 1.0/(1+np.exp(-x))

def logistic_prime(x):
    ex=np.exp(-x)
    return ex/(1+ex)**2

def identity(x):
    return x

def identity_prime(x):
    return 1

def test_regression(plots=False):
    n=200
    X=np.linspace(0,3*np.pi,num=n)
    X.shape=(n,1)
    y=np.sin(X)
    param=((1,0,0),(20, expit, logistic_prime),(20, expit, logistic_prime),(1,identity, identity_prime))
    rates=[0.05]
    predictions=[]
    for rate in rates:
        N=NeuralNetwork(X,y,param)
        N.train(4000, learning_rate=rate)
        predictions.append([rate,N.predict(X)])
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(1,1)
    if plots:
        ax.plot(X,y, label='Sine', linewidth=2, color='black')
        for data in predictions:
            ax.plot(X,data[1],label="Learning Rate: "+str(data[0]))
        ax.legend()

#test_regression(True)

def test_classification(plots=False):
    #Number samples
    n=700

    n_iter=1500
    learning_rate=0.05

    #Samples for true decision boundary plot
    L=np.linspace(0,3*np.pi,num=n)
    l = np.sin(L)

    #Data inputs, training
    X = np.random.uniform(0, 3*np.pi, size=(n,2))
    X[:,1] *= 1/np.pi
    X[:,1]-= 1


    #Data inputs, testing
    T = np.random.uniform(0, 3*np.pi, size=(n,2))
    T[:,1] *= 1/np.pi
    T[:,1] -= 1

    #Data outputs
    y = np.sin(X[:,0]) <= X[:,1]

    #Fitting
    param=((2,0,0),(30, expit, logistic_prime),(30, expit, logistic_prime),(1,expit, logistic_prime))
    N=NeuralNetwork(X,y, param)
    #Training
    N.train(n_iter, learning_rate)
    predictions_training=N.predict(X)
    predictions_training= predictions_training <0.5
    predictions_training= predictions_training[:,0]
    #Testing
    predictions_testing=N.predict(T)
    predictions_testing= predictions_testing <0.5
    predictions_testing= predictions_testing[:,0]

    #Plotting
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(2,1)

    #Training plot
    #We plot the predictions of the neural net blue for class 0, red for 1.
    ax[0].scatter(X[predictions_training,0], X[predictions_training,1], color='blue')
    not_index = np.logical_not(predictions_training)
    ax[0].scatter(X[not_index,0], X[not_index,1], color='red')
    ax[0].set_xlim(0, 3*np.pi)
    ax[0].set_ylim(-1,1)
    #True decision boundary
    ax[0].plot(L,l, color='black')
    #Shade the areas according to how to they should be classified.
    ax[0].fill_between(L, l,y2=-1, alpha=0.5)
    ax[0].fill_between(L, l, y2=1, alpha=0.5, color='red')

    #Testing plot
    ax[1].scatter(T[predictions_testing,0], T[predictions_testing,1], color='blue')
    not_index = np.logical_not(predictions_testing)
    ax[1].scatter(T[not_index,0], T[not_index,1], color='red')
    ax[1].set_xlim(0, 3*np.pi)
    ax[1].set_ylim(-1,1)
    ax[1].plot(L,l, color='black')
    ax[1].fill_between(L, l,y2=-1, alpha=0.5)
    ax[1].fill_between(L, l, y2=1, alpha=0.5, color='red')


#test_classification()
