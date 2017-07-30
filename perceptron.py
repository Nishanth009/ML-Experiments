# -*- coding: utf-8 -*-
"""
@author: nishanth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification


def plot_decision_regions(X, y,w0,w1,w2,res=0.02):
   
   colors = ('red', 'blue')
   cmap = ListedColormap(colors[:len(np.unique(y))])
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),np.arange(x2_min, x2_max, res))
   newx = np.array([xx1.ravel(), xx2.ravel()]).T
   A = np.ones(newx.shape[0])
   newx = np.c_[A,newx]
   Z = predict(newx,np.array([w0,w1,w2]).reshape(-1,1))
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
  
def predict(x,W):
    thr=0.0
    pred = np.dot(x,W)
    pred[pred>round(thr,3)]=1
    pred[pred<=round(thr,3)]=0
    return pred


def calc_error(x,y,w1,w2,w0):
    error = 0
    thr = 0.0
    pred = w0+x[0]*w1+x[1]*w2
    if round(pred,3)>round(thr,3):
        out = 1
    else:
        out = 0
    if y!=out:
    	error = y-pred
    return error



def computeWeights(X,Y):
    
    ##INITIALIZE PARAMETERS
    w0 = -0.1
    w1 = 0.001
    w2 = 0.001
    alpha = 0.01
    iterations = 100
    
    for i in range(iterations):
        for xval,yval in zip(X,Y):
            error = calc_error(xval,yval,w1,w2,w0)
            dw0 = error
            dw1 = xval[0]*error
            dw2 = xval[1]*error
            w0+=alpha*dw0
            w1+=alpha*dw1
            w2+=alpha*dw2
    return w0,w1,w2
        

if __name__=='__main__':
    
    ###CREATE DATASET(2 features)
    X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

 
    
    
    ##MODEL & PREDICT (linear classifier - binary classification problem)
    #X = np.array(X,np.float32)
    #Y = np.array(Y,np.float32)
    w0,w1,w2 = computeWeights(X,Y)
    plot_decision_regions(X, Y,w0,w1,w2,res=0.02)







    
        
