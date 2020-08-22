#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression for classification

- Linear regression
- Linear regression with regularization
- Logistic Regression

    Example in the end of this file
    
Author: Victor Ivamoto
April, 2020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets as ds   # Demo dataset

#############################################
# Auxiliary Methods
#############################################

# sigmoid function used in
# binary logistic regression
def sigmoid(s):
    sigmoid = np.exp(s) / (1 + np.exp(s))
    return sigmoid


# Stable softmax function
# Ref: Eli Bendersky website
# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
def softmax(s, axis = 1):
    max_s = np.max(s, axis = axis, keepdims = True)
    e = np.exp(s - max_s)
    y =  e / np.sum(e, axis = axis, keepdims = True)
    return y



#############################################
# Linear regression for classification
#############################################
# Train the model
def regression_train(X, y):
    """
    Train linear regression for classification
    
    Input
        X:      train set without bias
        y:      y in {0, 1}, vector for binary classification or one-hot encoded for multiclass
    Output
        w:      regression weights
    """
    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    # Convert y={-1, 1}
    y = np.where(y == 0, -1 ,y)
    # Compute the weights (w) using the formula in lesson 2, slide 31:
    # w = (X_T * X)^-1 * X_T * y
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


# Predict new values
def regression_predict(X, w):
    """
    Predict linear regression for classification
    
    Input
        X:          test set without bias
        w:          weights matrix
    Output
        y_hat:      predicted values
    """
    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    if w.shape[1] == 1:
        y_hat = np.sign(X @ w)
        y_hat = np.where(y_hat == -1, 0, y_hat)
    else:
        y_hat = np.sign(X @ w)
        y_hat = (y_hat == np.max(y_hat, axis = 1, keepdims = True)) * 1

    return y_hat


#############################################
# Logistic Regression
#############################################
# Returns the gradient for both binary and
# multi-class logistic regression.
# Fixed and variable learning rates (lr) available
# Implementation of algorithm in slide 79
def logistic_train(X, y, v=True, binary=True, maxiter=1000, lr=0.1):
    """
    Train logistic regression model.
    
    Input parameters
        X:          matrix of coefficients
        y:          vector of binary outcomes
        v:          True for variable learning rate (lr)
        binary:     True for binary classification (sigmoid),
                    False for multi-class (softmax)
        maxiter:    maximum number of iterations
        lr:         learning rate.
    Output:
        weights vector
    """
    # Insert column of 1s for bias
    X = np.insert(X, 0, 1, axis=1)

    # Step 1: Initial weights with random numbers
    np.random.seed(123)
    if binary:
        y = np.where(y == 0, -1, y)         # y = {-1, 1}
        w = np.random.rand(X.shape[1], 1)
    else:
        y = np.where(y == -1, 0, y)         # y = {0, 1}
        w = np.random.rand(X.shape[1], y.shape[1])

    wes = w
    # Calculate initial gradient (g)
    g = logistic_derivative(X=X, y=y, w=w, binary=binary)
    norm_new = np.linalg.norm(g)
    ites = 0
    cont = True

    # Step 2: For t = 0, 1, 2, ... do
    normagrad = 1e-6  # Maximum gradient norm
    it = 0
    while it < maxiter and norm_new > normagrad and cont:
        # Step 3: Calculate the new gradient
        g = logistic_derivative(X=X, y=y, w=w, binary=binary)
        # Step 4: Calculate learning rate (lr) for variable
        # gradient descent and binary classification
        #if v and binary:
        lr = logistic_lr(X = X, y = y, w = w, dir = -g, binary = binary)
        # Step 5: Update weight (w)
        w = w - lr * g

        # Early stopping
        # Stop iteraction after 30 bad gradient norms
        norm_old = norm_new
        norm_new = np.linalg.norm(g)
        if norm_new < norm_old:
            ites = 0
            cont = True
            wes = w
        else:
            ites += 1
            if ites == 30:
                cont = False

        #print(np.linalg.norm(g))

        # Increase the number of iterations
        it = it + 1

    # Return the weights vector, gradient and error
    return wes, g


# ==================================
# Logistic regression
# ==================================
# Predicted values
def logistic_predict(X, w, binary):
    """
    Predict logistic regression
    Input
        X:          test set
        w:          weights matrix
        binary:     True for binary classification
    Output
        y_hat:      predicted values
    """
    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    if w.ndim == 1:
        w = np.array([w]).T

    # Binary
    if binary:
        # Return 1 if P(y=1|X=x) > 50%, return 0 otherwise
        y_hat = (sigmoid(s=X @ w) >= 0.5) * 1
    
    # Multiclass
    else:
        # Make y_hat prediction with softmax
        y_hat = softmax(X @ w, axis=1)
        # Get the maximum value and transpose
        y_hat = (y_hat == np.max(y_hat, axis=1, keepdims = True)) * 1

    return y_hat


# ==================================
# 3.1 Logistic: Calculate the derivatives
# ==================================
# Return the sigmoid and softmax derivatives
# used in logistic regression
def logistic_derivative(X, y, w, binary=True):
    """
    Logistic regression derivative.
    
    Input definition
        x:          vector of feature coeficients
        y:          vector of outcomes
        w:          vector of weights
        binary:     True for binary classification (sigmoid),
                    False for multi-class (softmax)
    Output
        Gradient of sigmoid or softmax
    """
    if binary:
        # Calculate the sigmoid gradient
        # Aula 2, slide 68
        g =-np.mean(y * X / (1 + np.exp(y * X @ w)), axis=0, keepdims=True).T
    else:
        # Calculate softmax gradient
        # Used in multiclass logistic regression
        # Equation in page 35, definition 4.20
        # Ref: Lecture notes CS480/680–Fall 2018 - University of Waterloo
        # Yaoliang Yu
        # https://cs.uwaterloo.ca/~y328yu/mycourses/480/note06.pdf

        # p: probability of y being 1
        p = softmax(X @ w, axis = 1)

        # Calculate the gradient
        g = ((p - y).T @ X).T / len(X)
        
    return g


# ==================================
# 3.2 Logistic: Variable Learning rate (lr)
# ==================================
# Compute the variable learning rate of gradient
# descent used in binary logistic regression.
# Uses the bisection method.
# Ref: Matlab code from Clodoaldo Lima
def logistic_lr(X, y, w, dir, binary=True):
    """
    Logistic regression learning rate.
    
    Input
        X, y:       train set
        w:          weights matrix
        d:          direction
        binary:     True for binary classification (sigmoid),
                    False for multi-class (softmax)
    Output
        optimal learning rate
    """
    np.random.seed(1234)
    epsilon = 1e-3
    hlmin = 1e-3
    lr_l = 0  # Lower lr
    lr_u = np.random.rand()  # Upper lr

    # New w position
    wn = w + lr_u * dir
    # Calculate the gradient of new position
    dJdW = logistic_derivative(X=X, y=y, w=wn, binary=binary)
    g = dJdW.flatten()
    d = dir.flatten()
    hl = g.T @ d
    while hl < 0:
        lr_u = 2 * lr_u
        # Calculate the new position
        wn = w + lr_u * dir
        # Calculate the gradient of new position
        # f and h aren't used
        dJdW = logistic_derivative(X=X, y=y, w=wn, binary=binary)
        g = dJdW.flatten()
        hl = g.T @ d


    # lr medium is the average of lrs
    lr_m = (lr_l + lr_u) / 2
    # Estimate the maximum number of iterations
    maxiter = np.ceil(np.log((lr_u - lr_l) / epsilon))
    # Iteration counter
    it = 0
    while abs(hl) > hlmin and it < maxiter:
#        print('maxiter:', maxiter, 'it:', it)
        # Calculate new position
        wn = w + lr_m * dir
        # Calculate the gradient of the new position
        # Note: f and h aren't used
        dJdW = logistic_derivative(X=X, y=y, w=wn, binary=binary)
        g = dJdW.flatten()
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
        it = it + 1
    return lr_m


# ==================================
# 4. Linear Regression with Regularization
# ==================================
# 4.1 Regularization: Cross-validation error
# ==================================
# Error function for linear regression
def error(X, y, w, l, q=1):
    """
    Input values:
        X:      matrix of coefficients
        y:      vector of outputs
        w:      vector of weights
        l:      lambda (scalar)
        q:      1 for lasso, 2 for ridge regression
    Output
        error
    """
    # Equation in slide 43
    # Least squares
    E = np.mean((X @ w - y) ** 2) + l * np.sum(abs(w) ** q, axis = 0)

    # Return the error
    return E


#############################################
# Linear Regression with Regularization
#############################################
# This function calculates the regularization coefficient lambda
# We choose a grid of lambda values, and compute the cross-validation error
# for each value of lambda. We then select the tuning parameter value
# for which the cross-validation error is smallest. Finally, the model
# is re-fit using all of the available observations and the selected
# value of the tuning parameter
# Reference:
# G. James, D. Witten, T. Hastie, R. Tibshirani:  An Introduction to Statistical Learning
def regularization(X, y):
    """
    Linear regression for classification with regularizaton.
    Perform grid search with cross validation to find the best regularization factor.

    Input
        X:      train set
        y:      train set, one-hot encoded
    Output
        Return the best values: weight, lambda, accuracy and data frame
    """
    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis = 1)
    if type(y) is not np.ndarray:
        y = np.array(y)
    # convert y = {-1, 1}
    y = np.where(y == 0, -1, y)
    
    lmin = 0  # Initial regularization factor (lambda) value
    Emin = 1.  # Initial error value (100%)
    N = X.shape[1]  # Number of attributes
    I = np.eye(N)  # Identity matrix
    
    # Test several values of lambda(l) and pick the lambda
    # that miminizes the error (slide 41)
    
    # Step 1: choose a grid of lambda values
    for l in range(1000):
        
        # Step 2: Compute the cross-validation error for each lambda
        E = regularization_cv(X, y, l)
        
        # Step 3: Select lambda for smallest cross-validation error
        if E < Emin:
            Emin = E
            lmin = l
        
        # Dataframe to create chart
        if l == 0:
            df = pd.DataFrame(data = {'Lambda': l, 'Error': [E]})
        else:
            df = df.append({'Lambda': l, 'Error': E}, ignore_index = True)
    
    # Step 4: Re-fit the model with all samples and best lambda
    w = np.linalg.inv((lmin / N) * I + X.T @ X) @ X.T @ y
    
    # Return the best values: weight, lambda, accuracy and data frame
    return w, Emin, lmin, df


# ==================================
# 4.3 Regularization: k-Fold Cross-Validation
# Function used to tune regularization parameter
# ==================================
# The objective is estimate the parameter lambda that
# results in the lowest error in the validation set.
# We use k-fold cross validation for this: we split
# the training set into training and validation sets
# with 90% and 10% of the original training set size.
# Reference:
# G. James, D. Witten, T. Hastie, R. Tibshirani:  An Introduction to Statistical Learning
def regularization_cv(X, y, l, K=10):
    """
    Find best value of regularization factor (lambda) with cross validation
    Input
        X:      train set
        y:      train set, one-hot encoded
        l:      lambda, regularization factor
        K:      number of folds
    Output
        cross-validation error
    """
    N = X.shape[0]      # Number of observations
    d = X.shape[1]      # Number of attributes + 1
    I = np.eye(d)       # Identity matrix
    E = np.array([])    # Error array

    # k-fold cross validation
    for k in range(K):
        # Define K% of rows for validation set
        rv = np.array(range(int(N * k / K),
                            int(N * (k + 1) / K)))

        # Define complementary row numbers for train set
        r = np.setdiff1d(np.array(range(X.shape[0])), rv)

        # Create the train set
        X_train = X[r]
        y_train = y[r]

        # Create the validation set
        X_val = X[rv]
        y_val = y[rv]

        # Weight with regularization (slide 43)
        w = np.linalg.inv((l / N) * I + X_train.T @ X_train) @ X_train.T @ y_train

        # Calculate the cross-validation error
        err = regularization_cv_error(X=X_val, y=y_val, w=w)
        # err = error(X = X_val, y = y_val, w = w, l = l, q = 2)

        # Update the error vector
        E = np.insert(arr=E, obj=E.shape, values=err)

    # Return the cross-validation error
    return np.mean(E)


# ==================================
# 4.2 Regularization: Cross validation Error
# ==================================
# Compute the cross-validation error
# for binary and multi-class.
# The error is used to select the best regularization parameter.
# Reference:
# G. James, D. Witten, T. Hastie, R. Tibshirani:  An Introduction to Statistical Learning
def regularization_cv_error(X, y, w):
    """
    Cross validation error
    Input
        X:      matrix of coefficients
        y:      matrix or vector of outcomes
        w :     matrix or vector of weights
    """
    try:
        # Multi-class regression
        if y.shape(1) > 1:
            y_hat = pd.DataFrame(X @ w)
            for i in range(y_hat.shape[0]):
                y_hat.iloc[i] = (y_hat.iloc[i] == y_hat.iloc[i].max()) * 1
            y_hat.replace(0, -1, inplace=True)
    except:
        # Binary regression
        y_hat = np.sign(X @ w)

    E = np.mean(y_hat != y)
    return E



# ==================================
# 4.5 Regularization: Plot results
# ==================================
# Plot lambda x error from regularization
def plot(df, title, pdir = ''):
    """
    Plot lambda x error from regularization
    Input
        df:         dataframe with values to be plotted
        title:      chart title
        pdir:       directory to save the plot if not empty
    """
    x = df.Lambda
    y = df.Error

    plt.figure()
    plt.plot(x, y)  # Plot some data on the (implicit) axes.

    # Lables and title
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.title(title)
    if pdir != '':
        plt.savefig(pdir + title.replace(' ', '_') + '.png', dpi=300, bbox_inches='tight')
    plt.show()


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
    data = ds.load_breast_cancer()
    data = ds.load_wine()
    
    X = data.data
    
    # Binary
    if len(data.target_names) == 2:
        binary = True
        y = np.array([data.target]).T
    # Multiclass
    else:
        binary = False
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

    #================================
    # Linear regression
    #================================
    w = regression_train(X_train, y_train)
    y_hat = regression_predict(X_test, w)
    acc = np.mean(y_hat == y_test)
    print('Linear regression accuracy:', acc)
    
    #================================
    # Linear regression with regularization
    #================================
    w, Emin, lmin, df = regularization(X_train, y_train)
    y_hat = regression_predict(X_test, w)
    acc = np.mean(y_hat == y_test)
    print('Linear regression with regularization accuracy:', acc)

    #================================
    # Logistic regression
    #================================
    w, g = logistic_train(X_train, y_train, binary = binary)
    y_hat = logistic_predict(X_test, w, binary = binary)
    acc = np.mean(y_hat == y_test)
    print('Logistic regression accuracy:', acc)

    print(1)
