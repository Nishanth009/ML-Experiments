# -*- coding: utf-8 -*-
"""
@author: nishanth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
import math
from sklearn.preprocessing import OneHotEncoder


def plot_decision_regions(X, y,W2,W1,b2,b1,res=0.02):
   
   colors = ('red', 'blue')
   cmap = ListedColormap(colors[:len(np.unique(y))])
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),np.arange(x2_min, x2_max, res))
   newx = np.array([xx1.ravel(), xx2.ravel()]).T
   Z = predict(newx,W2,W1,b2,b1)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.xlabel('X1')
   plt.ylabel('X2')
   plt.ylim(xx2.min(), xx2.max())
   plt.scatter(X[y == 0, 0],X[y == 0, 1],alpha=0.8, c='red',
                              marker='s', label=0)
   plt.scatter(X[y == 1, 0],X[y == 1, 1],alpha=0.8, c='blue',
                              marker='x', label=1)
   plt.show()





def computeError(out,y):
    
    error = out-y
    return error,np.sum(error,axis=0)/2.0
    
def predict(X,W2,W1,b2,b1):
    
    neth = X.dot(W1)+b1
    outh = 1/(1+np.exp(-(neth)))
    netO = outh.dot(W2)+b2
    outO = 1/(1+np.exp(-(netO)))
    outO[outO>0.5]=1    
    outO[outO<0.5]=0
    return outO
     
    

def computeWeights(X,Y):
    
    ##INITIALIZE PARAMETERS
    W1 = np.random.randn(2, 100) / np.sqrt(2)
    b1 = np.zeros((1, 100)).reshape(1,-1)
    W2 = np.random.randn(100, 1) / np.sqrt(2)
    b2 = np.zeros((1, 1)).reshape(1,-1)
    alpha = 1e-2
    reg = 1e-3
    iterations = 20000
    error_list = []
    
    for i in range(iterations):
        
        neth = X.dot(W1)+b1
        outh = 1/(1+np.exp(-(neth)))
        netO = outh.dot(W2)+b2
        outO = 1/(1+np.exp(-(netO)))
        error,err = computeError(outO,Y)
        error_list.append(err)
        delta2 = np.multiply(np.multiply(outO,(1-outO)),error)
        dW2 = (outh.T).dot(delta2)
        db2 = np.sum(error, axis=0)     
        delta1 = np.multiply(delta2.dot(W2.T),np.multiply(outh,(1-outh)))
        dW1 = np.dot(X.T,delta1)
        db1 = np.sum(delta1, axis=0)
        dW2 += reg*W2 
        dW1 += reg*W1
        W2 += -alpha*dW2
        b2 += -alpha*db2
        W1 += -alpha*dW1 
        b1 += -alpha*db1

    #print W2,W1,b2,b1
    
    return W2,W1,b2,b1

if __name__=='__main__':
    
    ###CREATE DATASET(2 features)
   
    X, Y = make_moons(200,True,noise=0.20)
    #plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
    #plt.show()
    #y_base = np.array([0,1,1,0],np.float32)
    #y_base = y.flatten()
 
  
   # X = [[0,0],[0,1],[1,0],[1,1]]
    #X = [[1,1],[1,2],[1,3],[1,4],[2,2],[2,3],[3,2],[3,3],[4,1],[4,2],[4,3],[4,4]]
    #Y = np.array([1,1,1,1,0,0,0,0,1,1,1,1]).reshape(-1,1)
    ##MODEL & PREDICT (linear classifier - binary classification problem)
    #X = np.array(X,np.float32)
    
    #print X.shape,Y.shape
    W2,W1,b2,b1 = computeWeights(X,Y.reshape(-1,1))
    plot_decision_regions(X, Y.flatten(),W2,W1,b2,b1,res=0.02)







    
        
