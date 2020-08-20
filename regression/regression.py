#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stable softmax function
def softmax(s, axis = 1):
  max_s = np.max(s, axis = axis, keepdims = True)
  e = np.exp(s - max_s)
  y =  e / np.sum(e, axis = axis, keepdims = True)
  return y
# ==================================
# 2. Linear regression for classification
# ==================================
# Train the model
def fit_reg(X, y):
    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    # Compute the weights (w) using the formula in lesson 2, slide 31:
    # w = (X_T * X)^-1 * X_T * y
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


# Predict new values
def fx_reg(X, w):
    # Predict new values with the optimized weights and
    # X values from the test set

    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    if w.shape[1] == 1:
        y_hat = np.round(X @ w)
    else:
        y_hat = X @ w
        if type(y_hat) is not pd.DataFrame:
            y_hat = pd.DataFrame(y_hat)
        for i in range(y_hat.shape[0]):
            y_hat.iloc[i] = (y_hat.iloc[i] == y_hat.iloc[i].max()) * 1
    y_hat = np.array(y_hat)

    return y_hat


# ==================================
# 3. Logistic regression
# ==================================
# Predicted values
def fx_logistic(X, w, binary):
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
        y_hat = softmax(X @ w, axis=0)
        # Get the maximum value and transpose
        y_hat = ((y_hat == np.max(y_hat, axis=0)) * 1).T

    return y_hat


# ==================================
# 3.1 Logistic: Calculate the derivatives
# ==================================
# Return the sigmoid and softmax derivatives
# used in logistic regression
def calc_derivative(X, y, w, binary=True):
    # Input definition
    # x: vector of feature coeficients
    # y: vector of outcomes
    # w: vector of weights
    # binary: True for binary classification (sigmoid),
    #         False for multi-class (softmax)
    # Output: Gradient of sigmoid or softmax
    if binary:
        # Calculate the sigmoid gradient
        #N = X.shape[0]
        # Aula 2, slide 68
        #s = 0.
        #for n in range(N):
        #    s = s + (y[n] * X[n]) / (1 + np.exp(y[n] * w.T @ X[n]))
        #g1 = -s / N
        g =-np.mean(y * X / (1 + np.exp(y * X @ w)), axis=0, keepdims=True).T
    else:
        # Calculate softmax gradient
        # Used in multiclass logistic regression
        # Formula in page 35, definition 4.20
        # CS480/680–Fall 2018 - University of Waterloo

        # p: probability of y being 1
        p = softmax(w.T @ X.T, axis=1)

        # Calculate the gradient
        g = X.T @ (p - y.T).T

        N = X.shape[0]
        g1 = 0.
        for i in range(N):
            x_i = np.array([X[i]]).T
            y_i = np.array([y[i]]).T
            # p_i = scipy.special.softmax(w.T @ x_i, axis = 1)
            # Vou testar com minha função, se não der certo volta a linha comentada
            # p_i = scipy.special.softmax(w.T @ x_i)
            p_i = softmax(w.T @ x_i, axis=1)

            # Calculate the gradient
            g1 = g1 + x_i @ (p_i - y_i).T

    return g


# ==================================
# 3.2 Logistic: Variable Learning rate (lr)
# ==================================
# Compute the variable learning rate of gradient
# descent used in binary logistic regression
def calc_lr(X, y, w, d, binary=True):
    # d = direction
    # binary: True for binary classification (sigmoid),
    #         False for multi-class (softmax)
    np.random.seed(1234)
    epsilon = 1e-3
    hlmin = 1e-3
    lr_l = 0  # Lower lr
    lr_u = np.random.rand()  # Upper lr

    # New w position
    wn = w + lr_u * d
    # Calculate the gradient of new position
    g = calc_derivative(X=X, y=y, w=wn, binary=binary)
    hl = g.T @ d
    while hl < 0:
        lr_u = 2 * lr_u
        # Calculate the new position
        wn = w + lr_u * d
        # Calculate the gradient of new position
        # f and h aren't used
        g = calc_derivative(X=X, y=y, w=wn, binary=binary)
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
        wn = w + lr_m * d
        # Calculate the gradient of the new position
        # Note: f and h aren't used
        g = calc_derivative(X=X, y=y, w=wn, binary=binary)
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
# 3.3 Logistic: sigmoid function used in
# binary logistic regression
# ==================================
def sigmoid(s):
    sigmoid = np.exp(s) / (1 + np.exp(s))
    return sigmoid


# ==================================
# 3.4 Logistic: Gradient Descent
# ==================================
# Returns the gradient for both binary and
# multi-class logistic regression.
# Fixed and variable learning rates (lr) available
# Implementation of algorithm in slide 79
# +++++++++++++++++++++++++++++++++
def logistic(X, y, v=True, binary=True, maxiter=1000, lr=0.1):
    # Input parameters
    # X: matrix of coefficients
    # y: vector of binary outcomes
    # v: True for variable learning rate (lr)
    # binary: True for binary classification (sigmoid),
    #         False for multi-class (softmax)
    # maxiter: maximum number of iterations
    # lr: learning rate.
    # Output: weights vector

    X = np.insert(X, 0, 1, axis=1)
    y = np.where(y == 0, -1, y)

    normagrad = 1e-10  # Maximum gradient norm

    np.random.seed(123)
    # Step 1: Initial weights with random numbers
    if binary:
        w = np.random.rand(X.shape[1], 1) * 100
    else:
        w = np.random.rand(X.shape[1], y.shape[1]) * 100

    # Calculate initial gradient (g)
    g = calc_derivative(X=X, y=y, w=w, binary=binary)
    norm_new = np.linalg.norm(g)
    ites = 0
    cont = True
    # Step 2: For t = 0, 1, 2, ... do
    lr = 10
    t = 0
    while t < maxiter and np.linalg.norm(g) > normagrad and cont:
        # Increase the number of iterations
        t = t + 1
        # Step 3: Calculate the new gradient
        g = calc_derivative(X=X, y=y, w=w, binary=binary)
        # Step 4: Calculate learning rate (lr) for variable
        # gradient descent and binary classification
        if v and binary:
            lr = calc_lr(X = X, y = y, w = w, d = -g, binary = binary)
        # Step 5: Update weight (w)
        w = w - lr * g

        norm_old = norm_new
        norm_new = np.linalg.norm(g)
        if norm_new < norm_old:
            ites = 0
            wes = w
        else:
            ites += 1
            if ites == 30:
                cont = False

        print(np.linalg.norm(g))

    # Return the weights vector, gradient and error
    return wes, g


# ==================================
# 4. Linear Regression with Regularization
# ==================================
# 4.1 Regularization: Cross-validation error
# ==================================
# Error function for linear regression
def error(X, y, w, l, q=1):
    # Input values:
    # X: matrix of coefficients
    # y: vector of outputs
    # w: vector of weights
    # l: lambda (scalar)
    # q: 1 for lasso, 2 for ridge regression
    # Output: error

    # Formula in slide 43
    # Least squares
    y = np.array(y)
    E = 0.
    N = X.shape[0]  # Number of observations (rows in X matrix)
    d = X.shape[1]  # Number of columns in X matrix
    for i in range(N):
        s = 0.
        for j in range(d):
            s = s + X[i, j] * w[j]
        E += (s - y[i]) ** 2

    # Calculate regularization term
    M = w.shape[0]
    reg = 0.
    for j in range(M):
        reg = reg + abs(w[j]) ** q

    # Total error
    E = E / N + l * reg

    # E = np.mean((X_train @ w - y_train) ** 2) + np.sum(abs(w) ** q)
    # np.mean((X_train @ np.array([w[:,i]]).T - np.array([y_train[:,i]]).T) ** 2) + np.sum(abs(np.array([w[:,i]]).T))
    E = np.mean((X @ w - y) ** 2) + l * np.sum(abs(w) ** q)

    # Return the error
    return np.mean(E)


# ==================================
# 4.2 Regularization: Cross validation Error
# ==================================
# Compute the cross-validation error
# for binary and multi-class.
# The error is used to select the best regularization parameter.
# Reference:
# G. James, D. Witten, T. Hastie, R. Tibshirani:  An Introduction to Statistical Learning
def cv_error(X, y, w):
    # Input paramters
    # X: matrix of coefficients
    # y: matrix or vector of outcomes
    # w : matrix or vector of weights
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
def cross_validation(X, y, l, K=10):
    # Input parameters:
    # X: trainning set matrix with predictors (features)
    # y: trainning set vector with outcomes
    # l: lambda, regularization tuning parameter
    # K: number of folds

    N = X.shape[0]  # Number of observations
    d = X.shape[1]  # Number of attributes + 1
    I = np.eye(d)  # Identity matrix
    E = np.array([])  # Error array

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
        err = cv_error(X=X_val, y=y_val, w=w)
        # err = error(X = X_val, y = y_val, w = w, l = l, q = 2)

        # Update the error vector
        E = np.insert(arr=E, obj=E.shape, values=err)

    # Return the cross-validation error
    return np.mean(E)


# ==================================
# 4.4 Linear Regression - Regularization
# Return weights with regularization
# ==================================
# This function calculates the regularization coefficient lambda
# We choose a grid of lambda values, and compute the cross-validation error
# for each value of lambda. We then select the tuning parameter value
# for which the cross-validation error is smallest. Finally, the model
# is re-fit using all of the available observations and the selected
# value of the tuning parameter
# Reference:
# G. James, D. Witten, T. Hastie, R. Tibshirani:  An Introduction to Statistical Learning
#
def regularization(X, y):
    # Input parameters:
    # X: matrix of features coefficients in train set
    # y: vector of outcomes in train set.

    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    if type(y) is not np.ndarray:
        y = np.array(y)

    lmin = 0  # Initial lambda value
    Emin = 1.  # Initial error value (100%)
    N = X.shape[1]  # Number of attributes
    I = np.eye(N)  # Identity matrix

    # Test several values of lambda(l) and pick the lambda
    # that miminizes the error (slide 41)

    # Step 1: choose a grid of lambda values
    for l in np.arange(0, 1000):

        # Step 2: Compute the cross-validation error for each lambda
        E = cross_validation(X, y, l)

        # Step 3: Select lambda for smallest cross-validation error
        if E < Emin:
            Emin = E
            lmin = l

        # Dataframe to create chart
        if l == 0:
            df = pd.DataFrame(data={'Lambda': l, 'Error': [E]})
        else:
            df = df.append({'Lambda': l, 'Error': E}, ignore_index=True)

    # Step 4: Re-fit the model with all samples and best lambda
    w = np.linalg.inv((lmin / N) * I + X.T @ X) @ X.T @ y

    # Return the best values: weight, lambda, accuracy and data frame
    return w, Emin, lmin, df


# ==================================
# 4.5 Regularization: Plot results
# ==================================
# Plot lambda x error from regularization
def plot(df, pdir, title):
    # Input:
    # df = dataframe with values to be plotted
    # title = chart title
    x = df.Lambda
    y = df.Error

    plt.figure()
    plt.plot(x, y)  # Plot some data on the (implicit) axes.

    # Lables and title
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.title(title)
    plt.savefig(pdir + title.replace(' ', '_') + '.png', dpi=300, bbox_inches='tight')
    plt.show()
