#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
git remote set-url origin
Autoencoder

Architecture:
    - 1st and 3rd layers: sigmoid
    - 2nd and 4th layers: no activation function
    - Cost function: MSE
    - Equal number of neurons in 1st and 3rd layers

Features:
    - Early stopping
    - Variable gradient descent learning rate
    - Momentum (set eta > 0 to enable)
    - Update one layer weights individually, then all layers simultaneously
    - Plot MSE vs iteration (uncomment to save plot)
    - Derivatives computed in matrix notation and loop
    
Example code at the end of this file

Authors: Victor Ivamoto - Wesley Santos
Code and calculation of the derivative in GitHub: https://github.com/vivamoto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from sklearn.datasets import load_iris


# ===================================
# Auxiliary Methods
# ===================================
class obj:
    pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ===================================
# Autoencoder train the model
# ===================================
def train(Xd, L1, L2, plot = True):
    """
    Train autoencoder network.
    
    Input
        Xd:         matrix of values to encode
        L1:         number of 1st and 3rd layer neurons
        L2:         number of 2nd layer neurons
    Output
        weights:    object with weights and biases
    """
    N, m = Xd.shape

    np.random.seed(1234)
    weights = obj()
    weights.w1 = np.random.random((m, L1)) / 1000       # 1st layer weights
    weights.w2 = np.random.random((L1, L2))/ 1000       # 2nd layer weights
    weights.w3 = np.random.random((L2, L1))/ 1000       # 3rd layer weights
    weights.w4 = np.random.random((L1, m)) / 1000       # 4th layer weights

    weights.b1 = np.zeros((1, L1))                      # 1st layer bias
    weights.b2 = np.zeros((1, L2))                      # 2nd layer bias
    weights.b3 = np.zeros((1, L1))                      # 3rd layer bias
    weights.b4 = np.zeros((1, m))                       # 4th layer bias

    speed_w1 = np.zeros((weights.w1.shape))             # momentum
    speed_w2 = np.zeros((weights.w2.shape))
    speed_w3 = np.zeros((weights.w3.shape))
    speed_w4 = np.zeros((weights.w4.shape))

    speed_b1 = np.zeros((weights.b1.shape))
    speed_b2 = np.zeros((weights.b2.shape))
    speed_b3 = np.zeros((weights.b3.shape))
    speed_b4 = np.zeros((weights.b4.shape))

    _, y = encode(Xd, weights)             # AE output
    MSE_old = np.mean((Xd - y) ** 2)
    MSE_new = MSE_old
    plotData = pd.DataFrame(columns=['Iteration', 'MSE'])  # Vector of MSE
    lr = 1e-4       # gradient descent learning rate
    wsel = 1        # weight selector (select which weight to update)
    eta = 0.0       # momentum factor. Set to 0 to disable
    run = True      # early stopping: continue until MSE decreases
    ites = 0        # early stopping: count epochs of MSE increases
    it = 0          # number of epochs
    while it < 10000 and run:
        # Uncomment to see MSE evolution
        #print('eta:', eta, 'it:', it, 'Early stop epoch:', ites, 'layer:', wsel, 'MSE:', MSE_new)
        
        # Compute derivatives
        dJdA, dJdB, dJdC, dJdD, dJdB1, dJdB2, dJdB3, dJdB4 = derivative(X, weights)
        # Update the learning rate
        lr = learning_rate(Xd, weights, -dJdA, -dJdB, -dJdC, -dJdD, -dJdB1, -dJdB2, -dJdB3, -dJdB4)

        # Update one layer weights individually, then update all layers simultaneously
        if wsel == 0 or wsel == 1:
            speed_w1 = eta * speed_w1 + lr * dJdA
            speed_b1 = eta * speed_b1 + lr * dJdB1
            weights.w1 = weights.w1 - speed_w1
            weights.b1 = weights.b1 - speed_b1
        if wsel == 0 or wsel == 2:
            speed_w2 = eta * speed_w2 + lr * dJdB
            speed_b2 = eta * speed_b2 + lr * dJdB2
            weights.w2 = weights.w2 - speed_w2
            weights.b2 = weights.b2 - speed_b2
        if wsel == 0 or wsel == 3:
            speed_w3 = eta * speed_w3 + lr * dJdC
            speed_b3 = eta * speed_b3 + lr * dJdB3
            weights.w3 = weights.w3 - speed_w3
            weights.b3 = weights.b3 - speed_b3
        if wsel == 0 or wsel == 4:
            speed_w4 = eta * speed_w4 + lr * dJdD
            speed_b4 = eta * speed_b4 + lr * dJdB4
            weights.w4 = weights.w4 - speed_w4
            weights.b4 = weights.b4 - speed_b4
        
        _, y = encode(Xd, weights)
        MSE_old = MSE_new
        MSE_new = np.mean((Xd - y) ** 2)
        plotData.loc[len(plotData) + 1] = [it, MSE_new]
        #--------------------------
        # Early stopping
        #--------------------------
        # Keep best weights while MSE decreases
        if MSE_new + 1e-6< MSE_old:
            run = True
            ites = 0
            w1 = weights.w1
            w2 = weights.w2
            w3 = weights.w3
            w4 = weights.w4

            b1 = weights.b1
            b2 = weights.b2
            b3 = weights.b3
            b4 = weights.b4
            
        else:
            ites += 1           # count number of increasing MSE
            # Update next layer weights
            if ites == 30 and wsel != 0:
                ites = 0
                wsel += 1
                # Update all weights simultaneously
                if wsel == 5:
                    wsel = 0
            # Exit iteration after 30 epochs of increasing MSE
            # and all weights were updated
            if ites == 30 and wsel == 0:
                run = False
        it += 1
    
    if plot:
        plt.figure()
        plt.plot('Iteration', 'MSE', data=plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Autoencoder')
        plt.legend()
        plt.show()
    
    # Return best weights
    weights.w1 = w1
    weights.w2 = w2
    weights.w3 = w3
    weights.w4 = w4

    weights.b1 = b1
    weights.b2 = b2
    weights.b3 = b3
    weights.b4 = b4

    return weights

# ===================================
# Autoencoder Derivatives - Matrix computation
# ===================================
def derivative(x, weights):
    w1 = weights.w1             # 1st layer weights
    w2 = weights.w2             # 2nd layer weights
    w3 = weights.w3             # 3rd layer weights
    w4 = weights.w4             # 4th layer weights

    b1 = weights.b1             # 1st layer bias
    b2 = weights.b2             # 2nd layer bias
    b3 = weights.b3             # 3rd layer bias
    b4 = weights.b4             # 4th layer bias

    Uin = x @ w1 + b1           # 1st layer input
    u   = sigmoid(Uin)          # 1st layer output (N x L1)
    z   = u @ w2 + b2           # 2nd layer output (N x L2) - No activation
    Vin = z @ w3 + b3           # 3rd layer input
    v   = sigmoid(Vin)          # 3rd layer output   (N x L1)
    y   = v @ w4 + b4           # Autoencoder output (N, m)

    N  = len(x)                 # number of features or predictors
    error = y - x
    dJdB1 = np.mean((((error @ w4.T * v * (1 - v)) @  w3.T) @ w2.T) *  (u * (1 - u)), axis = 0)
    dJdB2 = np.mean((error @ w4.T * v * (1 - v)) @ w3.T, axis=0)
    dJdB3 = np.mean(error @ w4.T * v * (1 - v), axis=0)
    dJdB4 = np.mean(error, axis = 0)
    dJdW1 = 1/N * ((((error @ w4.T * v * (1 - v) @ w3.T) @ w2.T) * (u * (1 - u))).T @ x).T
    dJdW2 = 1/N * ((error @ w4.T * v * (1 - v) @ w3.T).T @ u).T
    dJdW3 = 1/N * ((error @ w4.T * v * (1 - v)).T @ z).T
    dJdW4 = 1/N * (error.T @ v).T

    return dJdW1, dJdW2, dJdW3, dJdW4, dJdB1, dJdB2, dJdB3, dJdB4

# ===================================
# Autoencoder Derivatives - Loop computation
# ===================================
# This is illustrative example of derivative using loops.
# This method is not in use, since matrix computation is faster.
def derivative_loop(x, weights):
    w1 = weights.w1          # 1st layer weights
    w2 = weights.w2          # 2nd layer weights
    w3 = weights.w3          # 3rd layer weights
    w4 = weights.w4          # 4th layer weights

    b1 = weights.b1          # 1st layer bias
    b2 = weights.b2          # 2nd layer bias
    b3 = weights.b3          # 3rd layer bias
    b4 = weights.b4          # 4th layer bias

    m, L1 = w1.shape
    L2 = len(b2)
    N = len(x)

    Uin = x @ w1 + b1           # 1st layer input
    u   = sigmoid(Uin)          # 1st layer output (N x L1)
    z   = u @ w2 + b2           # 2nd layer output (N x L2) - No activation
    Vin = z @ w3 + b3           # 3rd layer input
    v   = sigmoid(Vin)          # 3rd layer output   (N x L1)
    y   = v @ w4 + b4           # Autoencoder output (N, m)

    dJdW1 = np.zeros((m, L1))
    dJdW2 = np.zeros((L1, L2))
    dJdW3 = np.zeros((L2, L1))
    dJdW4 = np.zeros((L1, m))

    dJdB1 = np.zeros((L1))
    dJdB2 = np.zeros((L2))
    dJdB3 = np.zeros((L1))
    dJdB4 = np.zeros((m))

    # Layer 1
    for j in range(L1):
        s1 = np.zeros((N))
        for k in range(L2):
            for l in range(L1):
                for o in range(m):
                    s1 += (y[:, o] - x[:, o]) * w4[l, o] * v[:, l] * (1 - v[:, l]) * w3[k, l] * w2[
                        j, k] * u[:, j] * (1 - u[:, j])
        for i in range(m):
            dJdW1[i, j] = np.mean(s1 * x[:, i], axis=0)
        dJdB1[j] = np.mean(s1)

    # Layer 2
    for k in range(L2):
        s2 = np.zeros((N))
        for l in range(L1):
            for o in range(m):
                s2 += (y[:, o] - x[:, o]) * w4[l, o] * v[:, l] * (1 - v[:, l]) * w3[k, l]
        for j in range(L1):
            dJdW2[j, k] = np.mean(s2 * u[:, j], axis=0)
        dJdB2[k] = np.mean(s2)

    # Layer 3
    for l in range(L1):
        s3 = np.zeros((N))
        for o in range(m):
            s3 += (y[:, o] - x[:, o]) * w4[l, o] * v[:, l] * (1 - v[:, l])
        for k in range(L2):
            dJdW3[k, l] = np.mean(s3 * z[:, k], axis=0)
        dJdB3[l] = np.mean(s3)

    # Layer 4
    for o in range(m):
        s4 = (y[:, o] - x[:, o])
        for l in range(L1):
            dJdW4[l, o] = np.mean(s4 * v[:, l], axis=0)
        dJdB4[o] = np.mean(s4, axis=0)

    return dJdW1, dJdW2, dJdW3, dJdW4, dJdB1, dJdB2, dJdB3, dJdB4


# ===================================
# Autoencoder encode
# ===================================
def encode(Xd, weights):
    """
    Encode and decode Xd.
    
    Input
        Xd:         matrix of values to encode
        weights:    object with weights and biases
    Output:
        z:          enconded values of Xd
        y:          decoded values of z
    """
    w1 = weights.w1
    w2 = weights.w2
    w3 = weights.w3
    w4 = weights.w4

    b1 = weights.b1
    b2 = weights.b2
    b3 = weights.b3
    b4 = weights.b4

    Uin = Xd @ w1 + b1
    u   = sigmoid(Uin)
    z   = u @ w2 + b2           # Code values of X
    Vin = z @ w3 + b3
    v = sigmoid(Vin)
    y = v @ w4 + b4             # Ideally, y should be iqual to X

    # Return the coded input and recovered output
    return z, y


# ===================================
# Autoencoder decode
# ===================================
def decode(z, weights):
    """
    Decode z.

    Input
        z:          matrix of values to decode
        weights:    object with weights and biases
    Output:
        y:          decoded values of z
    """
    w3 = weights.w3
    w4 = weights.w4
    
    b3 = weights.b3
    b4 = weights.b4
    
    Vin = z @ w3 + b3
    v = sigmoid(Vin)
    y = v @ w4 + b4     # decoded values of z
    
    # Return the decoded values of z
    return y

# ===================================
# Autoencoder update the learning rate
# ===================================
# Bisection method, adapted from code from Prof. Clodoaldo Lima
def learning_rate(X, weights, dirW1, dirW2, dirW3, dirW4, dirB1, dirB2, dirB3, dirB4):
    w1 = weights.w1         # 1st layer weights
    w2 = weights.w2         # 2nd layer weights
    w3 = weights.w3         # 3rd layer weights
    w4 = weights.w4         # 4th layer weights

    b1 = weights.b1         # 1st layer bias
    b2 = weights.b2         # 2nd layer bias
    b3 = weights.b3         # 3rd layer bias
    b4 = weights.b4         # 4th layer bias

    epsilon = 1e-3
    hlmin = 1e-3
    lr_l = 0
    lr_u = np.random.random() * 1e-9
    lr_u = np.finfo(float).eps
    wt = obj()
    wt.w1 = w1 + lr_u * dirW1
    wt.w2 = w2 + lr_u * dirW2
    wt.w3 = w3 + lr_u * dirW3
    wt.w4 = w4 + lr_u * dirW4

    wt.b1 = b1 + lr_u * dirB1
    wt.b2 = b2 + lr_u * dirB2
    wt.b3 = b3 + lr_u * dirB3
    wt.b4 = b4 + lr_u * dirB4

    dJdW1, dJdW2, dJdW3, dJdW4, dJdB1, dJdB2, dJdB3, dJdB4 = derivative(X, wt)
    g   = np.concatenate((dJdW1.flatten(), dJdB1.flatten(),
                          dJdW2.flatten(), dJdB2.flatten(),
                          dJdW3.flatten(), dJdB3.flatten(),
                          dJdW4.flatten(), dJdB4.flatten()))
    dir = np.concatenate((dirW1.flatten(), dirB1.flatten(),
                          dirW2.flatten(), dirB2.flatten(),
                          dirW3.flatten(), dirB3.flatten(),
                          dirW4.flatten(), dirB4.flatten()))
    hl = g.T @ dir
    it = 0
    while hl < 0:
        lr_u = 2 * lr_u
        wt.w1 = w1 + lr_u * dirW1
        wt.w2 = w2 + lr_u * dirW2
        wt.w3 = w3 + lr_u * dirW3
        wt.w4 = w4 + lr_u * dirW4

        wt.b1 = b1 + lr_u * dirB1
        wt.b2 = b2 + lr_u * dirB2
        wt.b3 = b3 + lr_u * dirB3
        wt.b4 = b4 + lr_u * dirB4
        dJdW1, dJdW2, dJdW3, dJdW4, dJdB1, dJdB2, dJdB3, dJdB4 = derivative(X, wt)
        g = np.concatenate((dJdW1.flatten(), dJdB1.flatten(),
                            dJdW2.flatten(), dJdB2.flatten(),
                            dJdW3.flatten(), dJdB3.flatten(),
                            dJdW4.flatten(), dJdB4.flatten()))
        hl = g.T @ dir
        it+=1
        
    lr_m = (lr_l + lr_u) / 2
    itmax = ceil(np.log((lr_u - lr_l) / epsilon))
    it = 0
    while abs(hl) > hlmin and it < itmax:
        wt.w1 = w1 + lr_u * dirW1
        wt.w2 = w2 + lr_u * dirW2
        wt.w3 = w3 + lr_u * dirW3
        wt.w4 = w4 + lr_u * dirW4

        wt.b1 = b1 + lr_u * dirB1
        wt.b2 = b2 + lr_u * dirB2
        wt.b3 = b3 + lr_u * dirB3
        wt.b4 = b4 + lr_u * dirB4
        dJdW1, dJdW2, dJdW3, dJdW4, dJdB1, dJdB2, dJdB3, dJdB4 = derivative(X, wt)
        g = np.concatenate((dJdW1.flatten(), dJdB1.flatten(),
                            dJdW2.flatten(), dJdB2.flatten(),
                            dJdW3.flatten(), dJdB3.flatten(),
                            dJdW4.flatten(), dJdB4.flatten()))
        hl = g.T @ dir
        if hl > 0:
            lr_u = lr_m
        elif hl < 0:
            lr_l = lr_m
        else:
            break

        lr_m = (lr_l + lr_u) / 2
        it = it + 1

    return lr_m


# ============================
# Demo with iris dataset
# - Normalize X
# - Use grid search to find best architecture
# - Train and encode with best architecture
# ============================
if __name__ == '__main__':

    # Min-max normalization
    def normalize(X, xmin = 0, xmax = 0):
        if xmin == 0 or xmax == 0:
            xmin = np.min(X)
            xmax = np.max(X)
        return (X - xmin) / (xmax - xmin), xmin, xmax
    
    
    # ============================
    # Prepare dataset
    # ============================
    # Import dataset
    data = load_iris()
    X = data.data
   
    # Create train and test sets
    X_train = np.vstack((X[::3, :], X[1::3, :]))
    X_test = X[2::3, :]
    
    # Normalize X_train and X_test
    X_train, xmin, xmax = normalize(X_train)
    X_test, _, _ = normalize(X_test, xmin, xmax)
    
    # ============================
    # Grid search to build the architecture
    # ============================
    # Perform grid search to find best number of neurons
    # in layers 1 and 2
    plotData = pd.DataFrame(columns = ['L1', 'L2', 'MSE'])  # Vector of MSE
    mse_min = np.inf
    l1_min = 0
    l2_min = 0
    L2 = X.shape[1]         # Number of 2nd layer neurons
    L1 = 2 * L2             # Number of 1st layer neurons
    for l1 in range(1, L1 + 1, 2):
        for l2 in range(1, L2 + 1):
            w = train(X_train, L1 = l1, L2 = l2, plot = True)
            z, x_hat = encode(X_test, w)
            mse = np.mean((x_hat - X_test) ** 2)
            
            # Keep best values
            if mse < mse_min:
                mse_min = mse
                l1_min = l1
                l2_min = l2

            print('L1:', l1, 'L2:', l2, 'MSE:', mse)

    print('Best values.')
    print('MSE:', mse_min)
    print('1st layer neurons:', l1_min)
    print('2nd layer neurons:', l2_min)

    # Encode with best architecture
    w = train(X_train, L1 = l1_min, L2 = l2_min, plot = True)
    z, _ = encode(X_test, w)
    x_hat = decode(z, w)
    mse = np.mean((X_test - x_hat)**2)
    print('Test MSE:', mse)
    
    print(1)
