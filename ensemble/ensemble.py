#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble of Multi Layer Perceptron Networks

- Ensemble learing via negative correlation
- Ensemble Learning Using Decorrelated Neural Networks
- Creating Diversity In Ensembles Using Artiflcial Data (DECORATE)

Reference:

Ensemble learning via negative correlation
Y. Liu, X. Yao
https://doi.org/10.1016/S0893-6080(99)00073-8

Ensemble Learning Using Decorrelated Neural Networks
BRUCE E ROSEN
https://doi.org/10.1080/095400996116820

Creating Diversity In Ensembles Using Artificial Data
Prem Melville and Raymond J. Mooney
http://www.cs.utexas.edu/~ml/papers/decorate-jif-04.pdf

Author: Victor Ivamoto
August, 2020
Code and calculation of the derivative in GitHub: https://github.com/vivamoto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine  # Demo dataset

import neuralnets as nn

class obj:
    pass

# ==========================================
# Initialize parameters
# ==========================================
def ensemble(M = 4, L = 10, maxiter = 1000):
    """
    Creates an ensemble object with empty weights and biases.
    
    Attributes:
        M:          number of networks
        L:          number of hidden layer neurons
        maxiter:    maximum number of iterations
    """
    ens = obj()
    ens.M = M               # Number of networks
    ens.L = L               # Number of hidden layer neurons
    ens.maxiter = maxiter   # Maximum iteration
    
    # Ensemble settings
    ens.A = {}              # 1st layer weights
    ens.B = {}              # 2nd layer weights
    ens.b1 = {}             # 1st layer bias
    ens.b2 = {}             # 2nd layer bias
    
    return ens


# ==========================================
# Auxiliary Methods
# ==========================================
def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# Stable softmax function
def softmax(s, axis = 1):
    max_s = np.max(s, axis = axis, keepdims = True)
    exp = np.exp(s - max_s)
    y = exp / np.sum(exp, axis = axis, keepdims = True)
    return y


# MLP output
def mlp(X, A, B, b1, b2):
    v = X @ A.T + b1  # 1st layer input
    z = sigmoid(v)  # 1st layer output
    u = z @ B.T + b2  # 2nd layer input
    y = softmax(u, axis = 1)  # 2nd layer output
    return y


# ==========================================
# 1. Negative correlation
# ==========================================
# Ref: Ensemble learning via negative correlation
# Y. Liu, X. Yao
# https://doi.org/10.1016/S0893-6080(99)00073-8
# ==========================================
def negcor_train(X, yd, ens, lamb = 0.5, plot = True, pdir = '', DS_name = ''):
    """
    Ensemble learning via negative correlation.
    Trains an ensemble of multi layer perceptron networks, with negative
    correlated errors.

    Input
       X, yd:       training set
       ens:         ensemble object with parameters
       lamb:        lambda parameter. Provides a way to balance the
                    bias-variance-covariance trade-off
       plot:        True to make plot
       pdir:        directory to save the plot. Uncomment to save.
       DS_name:     dataset name used in plot title

     Output
       ens:        ensembe parameters with weights and biases
    """
    
    # Transform yd to 1 of n classes
    yd = np.where(yd == -1, 0, yd)
    if yd.shape[1] == 1:
        yd = np.array([(yd[:, 0] == 1) * 1, (yd[:, 0] == 0) * 1]).T
    
    N, m = X.shape  # Number of observations and features
    nc = yd.shape[1]  # Number of classes
    
    M = ens.M  # Number of networks
    L = ens.L  # Number of hidden layer neurons

    # A and B are weights matrices of 1st and 2nd layers
    # Create A and B matrices with random values
    np.random.seed(1234)
    for i in range(M):
        ens.A[i] = np.random.rand(L, m) * 2 - 1  # 1st layer weights
        ens.B[i] = np.random.rand(nc, L) * 2 - 1  # 2nd layer weights
        ens.b1[i] = np.zeros((1, L))  # 1st layer bias
        ens.b2[i] = np.zeros((1, nc))  # 2nd layer bias
    
    dEdF = np.random.rand(N, nc, M)  # Error derivative
    E = np.ones((M, nc))  # Error of network i
    p = np.zeros((N, nc, M))  # Penalty function
    F = np.zeros((N, nc))  # Ensemble output
    Fi = np.zeros((N, nc, M))  # Output of network i
    
    lr = 2  # Learning rate
    Et = np.mean(E)  # Total error
    plotData = pd.DataFrame(columns = ['Iteration', 'Error'])  # Vector of MSE
    it = 0  # iteration counter
    while abs(Et) > 1e-2 and it < ens.maxiter:
        # ---------------------
        # 1. Compute each network output
        # ---------------------
        for i in range(M):
            A = ens.A[i]
            B = ens.B[i]
            b1 = ens.b1[i]
            b2 = ens.b2[i]
            
            # Compute the derivatives
            dJdA, dJdB, dJdB1, dJdB2 = negcor_derivative(X, dEdF[:, :, i], A, B,
                                                         b1, b2)
            
            # Update learning rate
            #lr = negcor_lr(X, dEdF[:, :, i], A, B, b1, b2, -dJdA, -dJdB, -dJdB1,
            #               -dJdB2)
            
            # Update weight matrices
            A = A - lr * dJdA
            B = B - lr * dJdB
            b1 = b1 - lr * dJdB1
            b2 = b2 - lr * dJdB2
            
            # Keep each network output and weights
            Fi[:, :, i] = mlp(X, A, B, b1, b2)
            
            ens.A[i] = A
            ens.B[i] = B
            ens.b1[i] = b1
            ens.b2[i] = b2
        
        # ---------------------
        # 2. Ensemble output (equation 1)
        # ---------------------
        F = np.mean(Fi, axis = 2, keepdims = False)
        
        # ---------------------
        # 3. Compute penalty function p (equation 3)
        # ---------------------
        for i in range(M):
            s = 0
            for j in range(M):
                if i != j:
                    s += Fi[:, :, j] - F
            p[:, :, i] = (Fi[:, :, i] - F) * s
        
        for i in range(M):
            # ---------------------
            # 4. Compute error function Ei for network i (eq. 2)
            # ---------------------
            E[i] = np.mean(1 / 2 * (Fi[:, :, i] - yd) ** 2 + lamb * p[:, :, i],
                           axis = 0, keepdims = True)
            
            # ---------------------
            # 5. Derivative of E w.r.t. the output of network i (eq. 4)
            # ---------------------
            dEdF[:, :, i] = (1 - lamb) * (Fi[:, :, i] - yd) + lamb * (F - yd)
        
        # Total error is the mean error of all networks
        Et = np.mean(E)
        
        #print('it:', it, 'Error:', Et)
        # Keep plot data
        plotData.loc[len(plotData) + 1] = [it, Et]
        it += 1
    
    if plot:
        plt.figure()
        plt.plot('Iteration', 'Error', data = plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Ensemble via Negative Correlation - ' + DS_name.title())
        if pdir != '':
            plt.savefig(pdir + DS_name + '_negcor_' + str(M) + '_networks.png',
                    dpi = 300, bbox_inches = 'tight')
        plt.show()
    
    # Return the ensemble weights
    
    return ens


# ===================================
# Predict ensemble
# ===================================
def negcor_predict(X_test, nc, ens):
    """
    Ensemble learning via negative correlation.
    Predict from trained ensemble.

    Input
       X_test:       test set
       nc:           number of classes
       ens:          ensemble object
    
    Output
       y_hat:        predicted values, one-hot encoding
    """
    
    N = len(X_test)  # Number of observations
    y_hat = np.zeros((N, nc))  # network output
    
    # 1. Compute each network output
    for i in range(ens.M):
        A = ens.A[i]  # 1st layer weights
        B = ens.B[i]  # 2nd layer weights
        b1 = ens.b1[i]  # 1st layer bias
        b2 = ens.b2[i]  # 2nd layer bias
        
        y_hat += mlp(X_test, A, B, b1, b2)
    
    # 2. Convert softmax output to 0 and 1
    y_hat = np.where(y_hat == np.max(y_hat, axis = 1,
                                     keepdims = True), 1, 0)
    return y_hat


def negcor_derivative(X, dEdF, A, B, b1, b2):
    # Compute each MLP network output
    v = X @ A.T + b1            # 1st layer input
    z = sigmoid(v)              # 1st layer output
    u = z @ B.T + b2            # 2nd layer input
    y = softmax(u, axis = 1)    # 2nd layer output
    
    N = len(X)
    # Compute the derivatives
    dJdA = 1 / N * ((dEdF * y * (1 - y)) @ B).T * ((1 - z) * z).T @ X
    dJdB = 1 / N * (dEdF * y * (1 - y)).T @ z
    dJdB1 = np.mean(((dEdF * y * (1 - y)) @ B).T * ((1 - z) * z).T)
    dJdB2 = np.mean((dEdF * y * (1 - y)).T)
    
    return dJdA, dJdB, dJdB1, dJdB2


def negcor_lr(X, dEdF, A, B, b1, b2, dirA, dirB, dirB1, dirB2):
    np.random.seed(1234)
    epsilon = 1e-3  # precision
    hlmin = 1e-3
    lr_l = 0  # Lower bound learning rate
    lr_u = np.random.rand()  # Upper bound learning rate
    
    # New A and B positions
    An = A + lr_u * dirA
    Bn = B + lr_u * dirB
    b1n = b1 + lr_u * dirB1
    b2n = b2 + lr_u * dirB2
    # Calculate the gradient of new position
    dJdA, dJdB, dJdB1, dJdB2 = negcor_derivative(X, dEdF, An, Bn, b1n, b2n)
    g = np.concatenate((dJdA.flatten(), dJdB1.flatten(),
                        dJdB.flatten(), dJdB2.flatten()))
    d = np.concatenate((dirA.flatten(), dirB1.flatten(),
                        dirB.flatten(), dirB2.flatten()))
    hl = g.T @ d
    
    while hl < 0:
        #
        lr_u *= 2
        # Calculate the new position
        An = A + lr_u * dirA
        Bn = B + lr_u * dirB
        b1n = b1 + lr_u * dirB1
        b2n = b2 + lr_u * dirB2
        # Calculate the gradient of new position
        dJdA, dJdB, dJdB1, dJdB2 = negcor_derivative(X, dEdF, An, Bn, b1n, b2n)
        g = np.concatenate((dJdA.flatten(), dJdB1.flatten(),
                            dJdB.flatten(), dJdB2.flatten()))
        hl = g.T @ d
    
    # lr medium is the average of lrs
    lr_m = (lr_l + lr_u) / 2
    
    # Estimate the maximum number of iterations
    maxiter = np.ceil(np.log((lr_u - lr_l) / epsilon))
    
    it = 0  # Iteration counter
    while abs(hl) > hlmin and it < maxiter:
        An = A + lr_u * dirA
        Bn = B + lr_u * dirB
        b1n = b1 + lr_u * dirB1
        b2n = b2 + lr_u * dirB2
        # Calculate the gradient of new position
        dJdA, dJdB, dJdB1, dJdB2 = negcor_derivative(X, dEdF, An, Bn, b1n, b2n)
        g = np.concatenate((dJdA.flatten(), dJdB1.flatten(),
                            dJdB.flatten(), dJdB2.flatten()))
        
        hl = g.T @ d
        
        if hl > 0:
            # Decrease upper lr
            lr_u = lr_m
        elif hl < 0:
            # Increase lower lr
            lr_l = lr_m
        else:
            break
        # lr medium is the lr average
        lr_m = (lr_l + lr_u) / 2
        # Increase number of iterations
        it += 1
    
    return lr_m


# ==========================================
# 2. Decorrelated Neural Networks
# ==========================================
# Ref:
# Ensemble Learning Using Decorrelated Neural Networks
# BRUCE E ROSEN
# https://doi.org/10.1080/095400996116820
def decorrelated_train(X, yd, ens, lamb, alternate = False, plot = True,
                       pdir = '', DS_name = ''):
    """
    Ensemble Learning Using Decorrelated Neural Networks.
    Trains an ensemble of multi layer perceptron networks, using decorrelated
    neural networks.
    
    Input
       X, yd:       training set. yd is one-hot encoded.
       ens:         ensemble object
       lamb:        The scaling function lambda(t) is either constant or is time
                    dependent. Typically determined by cross-validation.
       alternate:   If False, individual networks are decorrelated with the
                    previously trained network. If True, alternate networks are
                    trained independently of one another yet decorrelate pairs of networks.
       plot:        True to make plot
       pdir:        directory to save the plot. Uncomment to save.
       DS_name:     dataset name used in plot title

    Output
       ens:     ensemble object with weights and biases
    
    """
    
    # One-hot encoding
    yd = np.where(yd == -1, 0, yd)
    if yd.shape[1] == 1:
        yd = np.array([(yd[:, 0] == 1) * 1, (yd[:, 0] == 0) * 1]).T
    
    N, m = X.shape          # Number of instances and features
    nc = yd.shape[1]        # Number of classes
    
    M = ens.M               # Number of networks
    L = ens.L               # Number of hidden layer neurons
    maxiter = ens.maxiter   # Maximum number of iterations
    
    np.random.seed(1234)
    for i in range(M):
        ens.A[i] = np.random.rand(L, m)     # 1st layer weights
        ens.B[i] = np.random.rand(nc, L)    # 2nd layer weights
        ens.b1[i] = np.zeros((1, L))        # 1st layer bias
        ens.b2[i] = np.zeros((1, nc))       # 2nd layer bias
    
    # ------------------------
    # 1. Train the 1st network
    # ------------------------
    y = np.zeros((N, nc, M))  # 2nd layer output
    
    lr = 0.001      # Learning rate
    plotData = pd.DataFrame(
        columns = ['Network', 'Iteration', 'MSE'])  # Vector of MSE
    for j in range(M):
        
        A = ens.A[j]    # 1st layer weights
        B = ens.B[j]    # 2nd layer weights
        b1 = ens.b1[j]  # 1st layer bias
        b2 = ens.b2[j]  # 2nd layer bias

        dJdA = np.ones((L, m))
        dJdB = np.ones((nc, L))
        dJdB1 = np.ones((1, L))
        dJdB2 = np.ones((1, nc))

        it = 0
        while np.linalg.norm(dJdA) + np.linalg.norm(dJdB) + \
                np.linalg.norm(dJdB1) + np.linalg.norm(dJdB2) > 1e-5 and \
                it < maxiter:
            # ------------------------
            # 1. Compute the error and derivative for network j
            # ------------------------
            s1 = 0
            s2 = 0
            for i in range(j):
                # The correlation penalty function P is the product
                # of the jth and ith network error:
                P = (yd - y[:, :, i]) * (yd - y[:, :, j])  # eq 8
                
                # The indicator function d specifies which individual
                # networks are to be decorrelated
                if alternate:
                    # To allow alternate networks to be trained independently
                    # of one another yet decorrelate pairs of networks,
                    # the indicator function can be defined as
                    d = 1 if (i == j - 1) and (i % 2 == 0) else 0  # Equation 10
                else:
                    # To penalize an individual network for being
                    # correlated with the previously trained network,
                    # the indicator function is:
                    d = 1 if (i == j - 1) else 0  # Equation 9
                s1 += lamb * d * P
                s2 += lamb * d * (y[:, :, i] - yd)
            # Derivative of E w.r.t. MLP output y
            dEdy = 2 * (y[:, :, j] - yd) + s2

            # ------------------------
            # 2. Train the network j
            # ------------------------
            # Compute gradients
            dJdA, dJdB, dJdB1, dJdB2 = decorrelated_derivative(X, dEdy, A, B, b1,
                                                               b2)
            # Update learning rate
            #lr = decorrelated_lr(X, dEdy, A, B, b1, b2, -dJdA, -dJdB, -dJdB1,
            #                     -dJdB2)
            # Update weight matrices
            A = A - lr * dJdA
            B = B - lr * dJdB
            b1 = b1 - lr * dJdB1
            b2 = b2 - lr * dJdB2
            
            # Update MLP output
            y[:, :, j] = mlp(X, A, B, b1, b2)
            
            # Update the error
            MSE = np.mean((y[:, :, j] - yd) ** 2)
            # Error function for an individual network j:
            MSE = np.mean((yd - y[:, :, j]) ** 2 + s1)
            
            plotData.loc[len(plotData) + 1] = [j, it, MSE]
            
            #print('it:', it, 'j:', j, 'MSE:', MSE)
            it += 1
        
        # Save the weights in matrix A and B
        ens.A[j] = A
        ens.B[j] = B
        ens.b1[j] = b1
        ens.b2[j] = b2
    
    if plot:
        plt.figure()
        for i in range(M):
            plt.subplot(M + 1, 1, i + 1)
            xrange = np.arane(MSE[:, i])
            plt.plot(xrange, MSE[:, i])
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Ensemble Decorrelated NN - ' + DS_name.title())
        if pdir != '':
            plt.savefig(pdir + DS_name + '_decorrelated_' + str(M) + '_networks.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Uncomment to save parameters to file
    # plotData.to_csv(pdir + DS_name + '_decorrelated_' + str(M) + '_networks.csv', index=False, decimal=',', sep='\t')
    
    # Return the ensemble object with weights
    return ens


# ===================================
# Predict ensemble
# ===================================
def decorrelated_predict(X_test, nc, ens):
    """
    Ensemble Learning Using Decorrelated Neural Networks.
    Predict from trained ensemble.
    
    Input
        X_test:     test set
        nc:         Number of classes
        ens:        ensemble object
    
    Output
        y_hat:      predicted values, one-hot encoding
    """
    
    N = len(X_test)  # Number of observations
    y_hat = np.zeros((N, nc))  # Ensemble predicted output
    u = np.zeros((N, nc, ens.M))  # Ensemble predicted output

    # 1. Compute each network output
    for i in range(ens.M):
        A = ens.A[i]  # 1st layer weights
        B = ens.B[i]  # 2nd layer weights
        b1 = ens.b1[i]  # 1st layer bias
        b2 = ens.b2[i]  # 2nd layer bias

        v = X_test @ A.T + b1  # 1st layer input
        z = sigmoid(v)  # 1st layer output
        u[:,:,i] = z @ B.T + b2  # 2nd layer input

    y_hat = np.mean(u, axis = 2)

    # 2. Convert softmax output to 0 and 1
    y_hat = np.where(y_hat == np.max(y_hat, axis = 1,
                                     keepdims = True), 1, 0)
    return y_hat


def decorrelated_derivative(X, dEdy, A, B, b1, b2):
    v = X @ A.T + b1  # 1st layer input
    z = sigmoid(v)  # 1st layer output
    
    # Compute gradients
    dJdB = dEdy.T @ z
    dJdB2 = np.sum(dEdy, axis = 0, keepdims = True)

    dJdA = (((dEdy @ B) * (z * (1 - z))).T @ X)
    dJdB1 = np.sum((((dEdy @ B) * (z * (1 - z))).T), axis = 1, keepdims = True).T

    return dJdA, dJdB, dJdB1, dJdB2


def decorrelated_lr(X, dEdy, A, B, b1, b2, dirA, dirB, dirB1, dirB2):
    np.random.seed(1234)
    epsilon = 1e-3  # precision
    hlmin = 1e-3
    lr_l = 0  # Lower bound learning rate
    lr_u = np.random.rand()  # Upper bound learning rate
    
    # New A and B positions
    An = A + lr_u * dirA
    Bn = B + lr_u * dirB
    b1n = b1 + lr_u * dirB1
    b2n = b2 + lr_u * dirB2
    # Calculate the gradient of new position
    dJdA, dJdB, dJdB1, dJdB2 = decorrelated_derivative(X, dEdy, An, Bn, b1n,
                                                       b2n)
    g = np.concatenate((dJdA.flatten(), dJdB1.flatten(),
                        dJdB.flatten(), dJdB2.flatten()))
    d = np.concatenate((dirA.flatten(), dirB1.flatten(),
                        dirB.flatten(), dirB2.flatten()))
    hl = g.T @ d
    
    while hl < 0:
        #
        lr_u *= 2
        # Calculate the new position
        An = A + lr_u * dirA
        Bn = B + lr_u * dirB
        b1n = b1 + lr_u * dirB1
        b2n = b2 + lr_u * dirB2
        # Calculate the gradient of new position
        dJdA, dJdB, dJdB1, dJdB2 = decorrelated_derivative(X, dEdy, An, Bn,
                                                           b1n, b2n)
        g = np.concatenate((dJdA.flatten(), dJdB1.flatten(),
                            dJdB.flatten(), dJdB2.flatten()))
        hl = g.T @ d
    
    # lr medium is the average of lrs
    lr_m = (lr_l + lr_u) / 2
    
    # Estimate the maximum number of iterations
    maxiter = np.ceil(np.log((lr_u - lr_l) / epsilon))
    
    it = 0  # Iteration counter
    while abs(hl) > hlmin and it < maxiter:
        An = A + lr_u * dirA
        Bn = B + lr_u * dirB
        b1n = b1 + lr_u * dirB1
        b2n = b2 + lr_u * dirB2
        # Calculate the gradient of new position
        dJdA, dJdB, dJdB1, dJdB2 = decorrelated_derivative(X, dEdy, An, Bn,
                                                           b1n, b2n)
        g = np.concatenate((dJdA.flatten(), dJdB1.flatten(),
                            dJdB.flatten(), dJdB2.flatten()))
        
        hl = g.T @ d
        
        if hl > 0:
            # Decrease upper lr
            lr_u = lr_m
        elif hl < 0:
            # Increase lower lr
            lr_l = lr_m
        else:
            break
        # lr medium is the lr average
        lr_m = (lr_l + lr_u) / 2
        # Increase number of iterations
        it += 1
    
    return lr_m


# ==========================================
# 3. DECORATE
# ==========================================
# Creating Diversity In Ensembles Using Artificial Data
# Prem Melville and Raymond J. Mooney
# http://www.cs.utexas.edu/~ml/papers/decorate-jif-04.pdf
def decorate_train(X_train, y_train, Csize =15, Imax = 50, Rsize = 0.5):
    """
    Creating Diversity In Ensembles Using Artificial Data
    Trains an ensemble of multi layer perceptron networks, adding artificially
    created data.

    Inputs
        X_train:        normalized X values
        y_train:        train set, one-hot encoded
        Csize:          desired ensemble size
        Imax:           maximum number of iterations to build an ensemble
        Rsize:          factor that determines number of artificial examples to generate

    Output
        C:              dictionary with ensemble weights and biases
    """
    i = 0               # step 1
    trials = 1          # step 2
    C = {}
    C[i] = nn.mlp_train(X_train, y_train, L = 10, plot = False)     # step 3
    y_hat = nn.mlp_predict(X_train, C[i])                           # step 5
    acc = np.mean(y_hat == y_train)                                 # step 5
    i = i + 1                                                       # step 4
    while i < Csize and trials < Imax:                              # step 6
        X_ad, y_ad = decorate_create_ad(X_train, y_hat, Rsize)      # step 7, 8
        X_new = np.vstack((X_train, X_ad))                          # step 9
        y_new = np.vstack((y_train, y_ad))                          # step 9
        C[i] = nn.mlp_train(X_new, y_new, L = 10, plot = False)     # step 10, 11
        y_hat = decorate_predict(X_train, C)                        # step 10, 11
        acc_new = np.mean(y_hat == y_train)                         # step 13
        if acc_new >= acc:
            i = i + 1
            acc = acc_new
        else:
            C.pop(i)                # remove component from ensemble
            trials = trials + 1

        #print('trials:', trials, 'it:', i, 'Accuracy:', acc_new)
        
    # Return the ensemble weights
    return C

# create artificial data
def decorate_create_ad(X_train, y_hat, Rsize):
    """
    Artificially creates training data and labels.
    
    Input:
        X_train:        training data
        y_hat:          predicted values by the ensemble
        Rsize:          factor that determines number of artiflcial examples to
                        generate
    Output:
        X_ad, y_ad:     artificially generated training data and labels
    """
    N, nf = X_train.shape   # Number of instances and features
    
    # Create new X values (step 7)
    mu = np.mean(X_train, axis = 0)
    sd = np.std(X_train, axis = 0)
    X_ad = np.random.normal(mu, sd, (int(N * Rsize), nf))

    # Create new labels (step 8)
    error = 0
    if np.sum(np.sum(y_hat, axis = 0) == 0):
        error = 1e-15
    py = np.mean(y_hat, axis = 0) + error       # Py(x)
    py = np.cumsum((1 / py) / np.sum(1 / py))   # Each class cumulative probability
    py = (py * N * Rsize).astype(int)           # cumulative range of each class
    idx = np.random.choice(int(N * Rsize), size = (int(N * Rsize)), replace = False)
    
    nc = y_hat.shape[1]                         # number of classes
    y_ad = np.zeros((int(N * Rsize), nc))
    for i in range(nc):
        if i == 0:
            y_ad[:, i] = (py[i] > idx) * 1
        else:
            y_ad[:,i] = ((py[i] > idx) * 1 + (idx >= py[i-1]) * 1 == 2) * 1

    return X_ad, y_ad

def decorate_predict(X_test, C):
    """
    Ensemble Learning Using Decorrelated Neural Networks.
    Predict from trained ensemble.
    
    Input
        X_test:         test set
        C:              dictionary with ensemble weights and biases
    
    Output
        y_hat:          predicted values, one-hot encoding
    
    """
    py = 0
    for i in range(len(C)):
        py = py + nn.mlp_predict(X_test, C[i])
    
    y_hat = py / len(C)
    y_hat = (np.max(y_hat, axis = 1, keepdims = True) == y_hat)*1

    return y_hat

# ============================
# Demo with iris dataset
# ============================
if __name__ == '__main__':
    # Min-max normalization
    def normalize(X, xmin = 0, xmax = 0):
        if xmin == 0 or xmax == 0:
            xmin = np.min(X)
            xmax = np.max(X)
        return (X - xmin) / (xmax - xmin), xmin, xmax
    
    # ----------------------------
    # Prepare dataset
    # ----------------------------
    # Import dataset
    data = load_wine()
    
    X = data.data
    
    # One-hot enconding: y = {0, 1}
    y = np.zeros((data.target.shape[0], 3))
    y[:, 0] = np.where(data.target == 0, 1, 0)
    y[:, 1] = np.where(data.target == 1, 1, 0)
    y[:, 2] = np.where(data.target == 2, 1, 0)
    
    # Create train and test sets
    X_train = np.vstack((X[::3, :], X[1::3, :]))
    y_train = np.vstack((y[::3, :], y[1::3, :]))
    X_test = X[2::3, :]
    y_test = y[2::3, :]
    
    # Normalize X_train and X_test
    X_train, xmin, xmax = normalize(X_train)
    X_test, _, _ = normalize(X_test, xmin, xmax)
    
    # Intialize parameters
    M = 4                   # Number of networks
    L = 10                  # Number of hidden layer neurons
    maxiter = 1000          # Maximum iteration
    ens = ensemble(M = M, L = L, maxiter = maxiter)        # Initialize parameters

    nc = y_test.shape[1]    # Number of classes
    # ----------------------------
    # Negative correlation
    # ----------------------------
    ens = negcor_train(X_train, y_train, ens, lamb = 0.5)  # Train
    y_hat = negcor_predict(X_test, nc, ens)  # Predict
    print('Neg correlation Accuracy:', np.mean(y_hat == y_test))
    
    # ----------------------------
    # Decorrelataed networks
    # ----------------------------
    ens = decorrelated_train(X_train, y_train, ens = ens, lamb = 1, alternate = True)
    y_hat = decorrelated_predict(X_test, nc, ens)
    print('Decorrelated net Accuracy:', np.mean(y_hat == y_test))
    
    # ----------------------------
    # DECORATE
    # ----------------------------
    C = decorate_train(X_train, y_train, Csize = 15, Imax = 50, Rsize = 0.2)
    y_hat = decorate_predict(X_test, C)
    print('DECORATE Accuracy:', np.mean(y_hat == y_test))

    print(1)
