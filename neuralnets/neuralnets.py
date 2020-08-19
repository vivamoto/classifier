#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Networks
- Single Layer Perceptron (SLP)
- Multi Layer Perceptron (MLP)

SLP architecture:
    - activation function: softmax
    - cost function: cross entropy

MLP architecture:
    - 1st layer: sigmoid
    - 2nd layer: softmax
    - cost function: cross entropy

Features:
    - Early stopping
    - Variable learning rate
    - Plot error vs iteration
    
Parameters:
    X, X_train:     normalized train set
    X_test:         normalized test set
    yd:             desired output with one-hot encondig. yd in [0, 1]
    y:              predicted values
    plot:           plot error vs iteration
    pdir:           directory to save plot (uncomment command to save)
    DS_name:        dataset name used in plot title
    
    maxiter:        maximum number of iterations
    dJtdw, dJtdb:   total cost derivative w.r.t. weights and bias

    lr_m:           optimized learning rate
    
MLP parameters:
    L:              number 1st layer neurons
    mlp:            object with weights and biases
    A, B:           1st and 2nd layers weights matrices
    b1, b2:         1st and 2nd layers biases
    dirA, dirB:     A and B direction of steepest decline
    dirB1, dirB2:   b1 and b2 direction of steepest decline
Cross validation parameters:
    K:          number of cross-validation folds
    l_max:      number of neuros of maximum cross validation accuracy
    
    
    Example in the end of this file
    
Author: Victor Ivamoto
August, 2020
Code and calculation of the derivative in GitHub: https://github.com/vivamoto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine  # Demo dataset

# ==================================
# 1. Auxiliary Methods
# ==================================
class obj:
    pass

def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

# Stable softmax function
def softmax(s, axis=1):
    max_s = np.max(s, axis=axis, keepdims=True)
    exp = np.exp(s - max_s)
    y = exp / np.sum(exp, axis=axis, keepdims=True)
    return y

# ==================================
# 2. Single Layer Perceptron (SLP)
# ==================================
# Single layer perceptron (SLP) with softmax activation
# function and cross entropy cost function
# -----------------------------------
# SLP: Train
# -----------------------------------
def slp_train(X_train, yd, maxiter=1000, plot=True, pdir = '',  DS_name=''):
    """
    Train single layer perceptron network.
    Architecture:
        - softmax activation function
        - cross entropy cost function
    
    Inputs
       X_train:        train set
       yd:             desired output
       plot:           plot error vs iteration
       pdir:           directory to save plot (uncomment command to save)
       DS_name:        dataset name used in plot title
    
    Output
       w:              weight matrix
    """
    np.random.seed(1234)
    
    # Convert Yd to [0, 1]
    yd = np.where(yd == -1, 0, yd)
    if yd.ndim == 1:
        yd = np.array([yd]).T

    # Convert Yd to 1 of n
    if yd.shape[1] == 1:
        yd = np.hstack((yd, np.ones((np.size(yd, 0), 1)) - yd))

    N, m = X_train.shape                # number of instances and features
    nc = np.size(yd, 1)                 # Number of classes
    w = np.random.rand(nc, m)           # Weight matrix
    y = softmax(X_train @ w.T, axis=1)  # SLP output

    # Early stopping settings
    run = True                          # continue iteration while MSE decreases
    ites = 0                            # count epochs of MSE degradation
    MSE_old = np.mean((y - yd) ** 2)
    MSE_new = MSE_old

    it = 0                              # Iteration counter
    alpha = 0.5                         # Gradient descent learning rate
    plotData = pd.DataFrame(columns=['Iteration', 'MSE'])  # Vector of MSE
    while run and it < maxiter:
        it += 1
        # Compute gradient (direction of steepest decline)
        dJdw, dJdb = slp_derivative(X_train, yd, y)
        # Update weight
        w = w - alpha * dJdw
        # Update perceptron output
        y = softmax(X_train @ w.T, axis=1)
        #-----------------------
        # Early stopping
        #-----------------------
        MSE_old = MSE_new
        MSE_new = np.mean((y - yd) ** 2)
        # Keep gradient descent while MSE improves
        if MSE_new < MSE_old:
            ites = 0
            run = True
        else:
            # Exit iteration after 30 bad MSE
            ites += 1
            if ites == 30:
                run = False

        plotData.loc[len(plotData) + 1] = [it, MSE_new]
    # Plot error vs iteration
    if plot:
        plt.figure()
        plt.plot('Iteration', 'MSE', data=plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('SLP - ' + DS_name.title())
        # Uncomment to save plot
        #plt.savefig(pdir + DS_name + '_SLP.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Uncomment to save results to file
    #plotData.to_csv(pdir + '../table/' + DS_name + '_SLP.csv', index=False, decimal=',', sep='\t')

    return w

# -----------------------------------
# SLP: Predict
# -----------------------------------
def slp_predict(X_test, w):
    y_hat = softmax(X_test @ w.T, axis=1).round()
    return y_hat

# -----------------------------------
# SLP: Derivative
# Softmax and cross entropy
# -----------------------------------
def slp_derivative(X, yd, y):
    # Inputs
    #   X:              normalized train set
    #   yd:             desired output coded 1 of n. yd in [0, 1]
    #   y:              predicted values
    # Output
    #   dJtdw, dJtdb:   total cost derivative w.r.t. weights and bias
    
    # Convert Y and Yd to [0, 1]
    yd = np.where(yd == -1, 0, yd)
    y = np.where(y == -1, 0, y)

    N = X.shape[0]  # number of instances

    dJtdb = np.mean((y - yd), axis=0, keepdims=True).T
    dJtdw = ((y - yd).T @ X) / N

    return dJtdw, dJtdb


# ==================================
# 3. Multi Layer Perceptron (MLP)
# ==================================
# Softmax activation function and cross entropy
# cost function
# -----------------------------------
# MLP Train
# -----------------------------------
def mlp_train(X_train, yd, L, maxiter=1000, plot=True, pdir = '', DS_name=''):
    """
    Train multi layer perceptron network with 2 layers.
    1st layer:      sigmoid
    2nd layer:      softmax
    cost function:  cross entropy
    
    Input parameters
        X_train:    normalized train set
        yd:         desired output with 1 of n coding. yd in [0, 1]
        L:          number 1st layer neurons
        maxiter:    maximum number of iterations
        plot:       create MSE vs Iteration plot (boolean)
        pdir:       directory to save plot (uncomment command to save)
        DS_name:    dataset name used in plot title
    
    Output:
        mlp:        object with weights and biases
    """
    # Transform yd to 1 of n classes
    yd = np.where(yd == -1, 0, yd)
    if yd.shape[1] == 1:
        yd = np.array([(yd[:, 0] == 1) * 1, (yd[:, 0] == 0) * 1]).T

    N, m = X_train.shape        # Number of instances and features
    nc = yd.shape[1]            # number of classes and 2nd layer neurons

    # Initialize A and B with random values
    np.random.seed(1234)
    A = np.random.rand(L, m)    # 1st layer weights
    B = np.random.rand(nc, L)   # 2nd layer weights
    b1 = np.zeros(L)            # 1st layer bias
    b2 = np.zeros(nc)           # 2nd layer bias

    # Compute MLP output
    v = X_train @ A.T + b1      # 1st layer input
    z = sigmoid(v)              # 1st layer output
    u = z @ B.T + b2            # 2nd layer input
    y = softmax(u, axis=1)      # 2nd layer output

    # Early stopping settings
    run = True                  # continue iteration while MSE decreases
    ites = 0                    # count epochs of MSE degradation
    MSE_old = np.mean((y - yd) ** 2)
    MSE_new = MSE_old

    lr = 0.1                    # Learning rate
    it = 0                      # Iteration counter
    plotData = pd.DataFrame(columns=['Iteration', 'MSE'])  # Vector of MSE
    while run and it < maxiter:
        # Compute gradients (direction of steepest decline)
        dJdA, dJdB, dJdB1, dJdB2 = mlp_derivative(X_train, yd, A, B, b1, b2)
        # Update the learning rate
        lr = mlp_lr(X_train, yd, A, B, b1, b2, -dJdA, -dJdB, -dJdB1, -dJdB2)

        # Update weight matrices
        A = A - lr * dJdA       # 1st layer weights
        B = B - lr * dJdB       # 2nd layer weights
        b1 = b1 - lr * dJdB1    # 1st layer bias
        b2 = b2 - lr * dJdB2    # 2nd layer bias

        # Update MLP output
        v = X_train @ A.T + b1  # 1st layer input
        z = sigmoid(v)          # 1st layer output
        u = z @ B.T + b2        # 2nd layer input
        y = softmax(u, axis=1)  # 2nd layer output
        #-----------------------
        # Early stopping
        #-----------------------
        MSE_old = MSE_new
        MSE_new = np.mean((y - yd) ** 2)
        # Keep running while MSE improves
        if MSE_new < MSE_old:
            ites = 0
            run = True
        else:
            # Exit iteration after 30 bad MSEs
            ites += 1
            if ites == 30:
                run = False

#        print('it:', it, 'MSE:', MSE_new)
        plotData.loc[len(plotData) + 1] = [it, MSE_new]
        it += 1

    # Plot error vs iteration
    if plot:
        plt.figure()
        plt.plot('Iteration', 'MSE', data=plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('MLP ' + str(L) + ' Neurons - ' + DS_name.title())
        # Uncomment to save plot
        #plt.savefig(pdir + DS_name + '_MLP_' + str(L) + '_neurons.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Uncomment to save parameters to file
    #plotData.to_csv(pdir + '../table/' + DS_name + '_MLP_' + str(L) + '_neurons.csv', index=False, decimal=',', sep='\t')

    # Return object with weights and biases
    mlp = obj()
    mlp.A = A
    mlp.B = B
    mlp.b1 = b1
    mlp.b2 = b2

    return mlp


# -----------------------------------
# MLP Predict
# -----------------------------------
# Predict new values in MLP network
def mlp_predict(X_test, mlp):
    """
    Predict MLP values.

    Input
     X_test:    test set
     mlp:       object with weights and biases

    Output
     y_hat:     predicted values
    """
    v = X_test @ mlp.A.T + mlp.b1   # 1st layer input
    z = sigmoid(v)                  # 1st layer output
    u = z @ mlp.B.T + mlp.b2        # 2nd layer input
    y_hat = softmax(u, axis = 1)    # 2nd layer output
    
    # Convert softmax output to 0 and 1
    y_hat = np.where(y_hat == np.max(y_hat, axis = 1,
                                     keepdims = True), 1, 0)
    return y_hat


# -----------------------------------
# MLP Derivative
# -----------------------------------
def mlp_derivative(X, yd, A, B, b1, b2):
    # Inputs
    #   X, yd:          train set
    #   A, B:           1st and 2nd layers weights matrices
    #   b1, b2:         1st and 2nd layers biases vectors
    # Output
    #   dJdA, dJdB:     A and B derivatives

    N = len(X)          # Number of instances and features
    v = X @ A.T + b1    # 1st layer input
    z = sigmoid(v)      # 1st layer output
    u = z @ B.T + b2    # 2nd layer input
    y = softmax(u)      # 2nd layer output

    # Compute the derivatives
    dJdB = 1 / N * ((y - yd).T @ z)
    dJdA = 1 / N * (((y - yd) @ B) * ((1 - z) * z)).T @ X

    dJdB2 = np.mean((y - yd), axis = 0)
    dJdB1 = np.mean(((y - yd) @ B * ((1 - z) * z)), axis = 0)

    return dJdA,  dJdB, dJdB1, dJdB2


# -----------------------------------
# MLP: Gradient descent learning rate
# -----------------------------------
# Calculate the learning rate using bisection algorithm
# The learning rate is used in MLP for faster convergence
def mlp_lr(X, yd, A, B, b1, b2, dirA, dirB, dirB1, dirB2):
    # Inputs
    #   X, yd:          MLP input and output matrices (train set)
    #   A, B:           1st and 2nd layers weights matrices
    #   b1, b2:         1st and 2nd layers biases
    #   dirA, dirB:     A and B direction of steepest decline
    #   dirB1, dirB2:   b1 and b2 direction of steepest decline
    # Output
    #   lr_m:           optimized learning rate

    np.random.seed(1234)
    epsilon = 1e-3              # precision
    hlmin = 1e-3
    lr_l = 0                    # Lower bound learning rate
    lr_u = np.random.rand()     # Upper bound learning rate

    # New A and B positions
    An = A + lr_u * dirA
    Bn = B + lr_u * dirB
    b1n = b1 + lr_u * dirB1
    b2n = b2 + lr_u * dirB2
    # Calculate the gradient of new position
    dJdA, dJdB, dJdB1, dJdB2 = mlp_derivative(X=X, yd=yd, A=An, B=Bn, b1 = b1n, b2 = b2n)
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
        dJdA, dJdB, dJdB1, dJdB2 = mlp_derivative(X=X, yd=yd, A=An, B=Bn, b1=b1n, b2=b2n)
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
        dJdA, dJdB, dJdB1, dJdB2 = mlp_derivative(X=X, yd=yd, A=An, B=Bn, b1=b1n, b2=b2n)
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


# -----------------------------------
# MLP: Cross validation
# -----------------------------------
# Use cross validation to determine the optimal number
# of neurons in the MLP network
def mlp_cv(X_train, y_train, L=20, maxiter=1000, K=5, plot=True, pdir = '', DS_name=''):
    """
    Find optimal number of neurons using cross validation.
    
    Input
       L:          number of neurons
       maxiter:    maximum number of iterations
       K:          number of cross-validation folds
       plot:       plot error vs iteration
       pdir:       directory to save plot (uncomment command to save)
       DS_name:    dataset name used in plot title
       
     Output
       l_max:      number of neuros of maximum cross validation accuracy
    """
    N, d = X_train.shape
    acc_max = 0
    l_max = 0
    # Data to create plot
    plotData = pd.DataFrame(columns=['L', 'Accuracy'])

    # ------------------------
    # 1. Test several number of neurons using cross-validation
    # ------------------------
    for l in range(1, L, 2):    # l: number of neurons
        acc_vec = np.array([])
        for k in range(K):      # k: number of folds
            # Define K% of rows for validation set
            rv = np.array(range(int(N * k / K),
                                int(N * (k + 1) / K)))

            # Define complementary row numbers for train set
            r = np.setdiff1d(np.array(range(X_train.shape[0])), rv)
            # Create the train set
            X1 = X_train[r]
            y1 = y_train[r]
            # Create the validation set
            X2 = X_train[rv]
            y2 = y_train[rv]

            if y1.shape[1] == 1:
                y1 = np.array([(y1[:, 0] == 1) * 1, (y1[:, 0] == 0) * 1]).T
            if y2.shape[1] == 1:
                y2 = np.array([(y2[:, 0] == 1) * 1, (y2[:, 0] == 0) * 1]).T

            # Train MLP
            mlp = mlp_train(X1, y1, L=l, maxiter=maxiter, plot=False, pdir = pdir, DS_name=DS_name)
            # Predict
            y_hat = mlp_predict(X2, mlp)
            # Compute the accuracy
            acc = (100 * np.mean(y_hat == y2)).round(2)

            # Save the accuracy in a vector
            acc_vec = np.insert(arr=acc_vec, obj=acc_vec.shape, values=acc)
        # Cross-validation accuracy is the
        # mean value of all k-folds
        cv_acc = np.mean(acc_vec)

        # Keep data for plotting
        plotData.loc[len(plotData) + 1] = [l, cv_acc]
        # Keep the best values after running
        # for each K folds,
        if cv_acc > acc_max:
            acc_max = cv_acc    # Best accuracy achieved
            l_max = l           # Number of neurons of best accuracy

    # ------------------------
    # 2. Create plot (# Neurons vs Accuracy)
    # ------------------------
    if plot:
        plt.figure()
        plt.plot('L', 'Accuracy', data=plotData)
        plt.scatter('L', 'Accuracy', data=plotData)
        plt.xlabel('Number of Neurons')
        plt.ylabel('Cross Validation Accuracy (%)')
        plt.title('Cross Validation MLP - ' + DS_name.title())
        # Uncomment to save plot
        #plt.savefig(pdir + DS_name + '_cv_mlp_' + str(L) + '_neurons.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Uncomment to save parameters to file
    #plotData.to_csv(pdir + '../table/' + DS_name + '_cv_mlp_' + str(L) + '_neurons.csv', index=False, decimal=',', sep='\t')

    return l_max

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
        
    #============================
    # Prepare dataset
    #============================
    # Import dataset
    data = load_wine()

    X = data.data

    # One-hot enconding
    y = np.zeros((data.target.shape[0], 3))
    y[:,0] = np.where(data.target == 0, 1, 0)
    y[:,1] = np.where(data.target == 1, 1, 0)
    y[:,2] = np.where(data.target == 2, 1, 0)

    # Create train and test sets
    X_train = np.vstack((X[::3,:], X[1::3,:]))
    y_train = np.vstack((y[::3,:], y[1::3,:]))
    X_test = X[2::3,:]
    y_test = y[2::3,:]

    # Normalize X_train and X_test
    X_train, xmin, xmax = normalize(X_train)
    X_test, _, _ = normalize(X_test, xmin, xmax)
    
    #============================
    # SLP
    #============================
    w = slp_train(X_train, y_train, DS_name = 'Iris')  # Train
    y_hat = slp_predict(X_test, w)                      # Predict
    print('SLP Accuracy:', np.mean(y_hat == y_test))                     # Accuracy

    #============================
    # MLP
    #============================
    mlp = mlp_train(X_train, y_train, L=10, DS_name = 'Iris')  # Train
    y_hat = mlp_predict(X_test, mlp)                            # Predict
    print('MLP Accuracy:', np.mean(y_hat == y_test))                     # Accuracy

    #============================
    # Cross validation to find best number of neurons
    #============================
    # This will takes a few minutes to run
    l = mlp_cv(X_train, y_train)                                # Cross validation
    print('Best number of neurons:', l)
    
    mlp = mlp_train(X_train, y_train, L=l, DS_name = 'Iris')   # Train
    y_hat = mlp_predict(X_test, mlp)                            # Predict
    print('CV MLP Accuracy:', np.mean(y_hat == y_test))       # Accuracy

    
