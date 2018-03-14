# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:04:31 2018
@author: Nazneen Kotwal
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        self.error1 = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.5, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        for k in range(epochs):
            if k % 10000 == 0: print( 'epochs:', k)
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            self.error1.append(error)
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
         
    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    nn = NeuralNetwork([2,2,1])

    mean1 = (0,0)
    mean2 = (1,1)
    mean3 = (1,0)
    mean4 = (0,1)
    cov = ((0.01, 0),(0, 0.01)) 
    sample1 = np.random.multivariate_normal(mean1, cov, 150)
    sample2 = np.random.multivariate_normal(mean2, cov, 150)
    sample3 = np.random.multivariate_normal(mean3, cov, 150)
    sample4 = np.random.multivariate_normal(mean4, cov, 150)

    z1 = np.zeros([150,1])
    z2 = np.ones([150,1])
    # Assigning Classes to the Data Set Created
    a = np.hstack((sample1,z1[:150,]))
    b = np.hstack((sample2,z1[:150,]))
    c = np.hstack((sample3,z2[:150,]))
    d = np.hstack((sample4,z2[:150,]))
    X1 = np.concatenate((a[:100,],b[:100,],c[:100,],d[:100,]))
    Xtest = np.concatenate((a[100:,:2],b[100:,:2],c[100:,:2],d[100:,:2]))
    X = X1[:,:2]
#    y = np.array([0, 1, 1, 0])
    y = X1[:,2]
    y = np.reshape(y,(-1,1))
    nn.fit(X, y)
    for e in Xtest:
#         e = np.reshape(e,(1,2))
         print(e,nn.predict(e))
         
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    plt.scatter(X1[:,0],X1[:,1], c=X1[:,2], cmap=cmap_bold,
    edgecolor='k', s=20) 
    plt.show()     
         