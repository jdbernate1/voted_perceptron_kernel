from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from numpy import linalg
import random
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class VotedPerceptron:
    def __init__(self, n_iter,kernel=gaussian_kernel):
        self.n_iter = n_iter
        self.V = []
        self.C = []
        self.k = 0
        self.kernel = kernel
    
    def fit(self, x, y):
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
        k = 0
        v = [np.ones_like(x)[0]]
        c = [0]
        for epoch in range(self.n_iter): # runs through the data n_iter times
            for i in range(len(x)):
                pred = 1 if self.kernel(v[k], x[i]) > 0 else -1 # checks the sing of v*k
                if pred == y[i]: # checks if the prediction matches the real Y
                    c[k] += 1 # increments c
                else:
                    v.append(np.add(v[k], self.kernel(y[i], x[i])))
                    c.append(1)
                    k += 1
        self.V = v
        self.C = c
        self.k = k

    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for w,c in zip(self.V,self.C):
                s = s + c*np.sign(self.kernel(w,x))
            preds.append(np.sign(1 if s>= 0 else 0))
        return preds





x,y =make_circles(n_samples=1500, noise = 0.05 )
y = np.where(y==0,-1,1)
X_train, y_train = x[:750],y[:750]
X_test, y_test = x[750:], y[750:]
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

clf_votedPerc = VotedPerceptron(n_iter=100)
clf_votedPerc.fit(X_train, y_train)
y_pred = clf_votedPerc.predict(X_test)
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Metrics Report')
print(classification_report(y_test, y_pred))

