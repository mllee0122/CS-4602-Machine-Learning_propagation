import numpy as np
import pandas as pd
from sklearn import datasets #Iris data
from numpy import random
from math import exp

# Processing Iris Dataset
iris = datasets.load_iris()
data = iris['data']
target = iris['target'] # target

# Sigmoid function
def sigmoid(z):
    f = 1/(1+exp(-z))
    return f

# Initial Weights
def initial_weight():
    w1 = np.zeros((4, 3))
    w2 = np.zeros((4, 4))
    w3 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if j != 3:
                w1[i, j] = random.uniform(-0.1, 0.1)
            w2[i, j] = random.uniform(-0.1, 0.1)
            w3[i, j] = random.uniform(-0.1, 0.1)
    return w1, w2, w3        
 
# Forward propagation
def fwd_propagation(x, w1, w2, w3):
    for i in range(4):
        h2[i] = sigmoid((w3[:, i]*x).sum(axis = 0))
        h1[i] = sigmoid((w2[:, i]*h2).sum(axis = 0))
    for i in range(3):
        y[i] = sigmoid((w1[:, i]*h1).sum(axis = 0))
    return h1, h2, y

# Backpropagation
def backpropagation(x, y, t, h1, h2, w1, w2, w3, eta):
    delta1 = np.zeros((3))
    Delta1 = np.zeros((4))
    delta2 = np.zeros((4))
    Delta2 = np.zeros((4))
    delta3 = np.zeros((4))
    
    for i in range(3):
        delta1[i] = y[i]*(1-y[i])*(t[i]-y[i])
    for i in range(4):
        Delta1[i] = (delta1*w1[i, :]).sum(axis = 0)
        delta2[i] = h1[i]*(1-h1[i])*Delta1[i]
    for i in range(4):
        Delta2[i] = (delta2*w2[i, :]).sum(axis = 0)
        delta3[i] = h2[i]*(1-h2[i])*Delta2[i]
    
    # Update Weights
    for i in range(4):
        for j in range(3):
            w1[i, j] += eta*delta1[j]*h1[i]
    for i in range(4):
        for j in range(4):
            w2[i, j] += eta*delta2[j]*h2[i]
    for i in range(4):
        for j in range(4):
            w3[i, j] += eta*delta3[j]*x[i]
    return w1, w2, w3

# main
eta = 0.1
w1, w2, w3 = initial_weight()
MSE = 0
MSE_new = 0
epoch = -1
MSE_diff = 0

while MSE == 0 or MSE_diff > 10**-4:
    epoch += 1
    MSE = MSE_new
    MSE_new = 0.0
    
    for l in range(len(data)):
        x = data[l]
        y = np.zeros((3))
        t = np.zeros((3))
        h1 = np.zeros((4))
        h2 = np.zeros((4))

        # Target
        if target[l] == 0:
            t[0] = 1
        elif target[l] == 1:
            t[1] = 1
        elif target[l] == 2:
            t[2] = 1

        # Forward Propagation
        h1, h2, y = fwd_propagation(x, w1, w2, w3)
        
        if list(y) != list(t):
            # Backrpopagation
            w1, w2, w3 = backpropagation(x, y, t, h1, h2, w1, w2, w3, eta)
            temp = 0.0
            for i in range(3):
                temp += (t[i]-y[i])**2
                MSE_new += temp/3

    # Termination Criterion
    if MSE == 0:
        MSE_diff = 0
    else:
        MSE_diff = (abs(MSE_new-MSE)/MSE)

    print('Epoch', epoch,'MSE\n\t', MSE_new/len(data))

print('\n\nNumber of epochs : ', epoch, '\nMSE Difference : ', MSE_diff)
