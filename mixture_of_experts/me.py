#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Mixture of Experts
 
 Simple architecture with no hierarchy of experts.
 Two setups:
    1. Linear models for experts and gating with softmax output
    2. Linear models for experts and gating with normalized gaussian output

 Input: time series one step ahead
 Goal:  predict new values

 Main methods:
 - best_lag:        Create charts to select the best lag length
 - cv_experts:      Cross validation for choosing the number of experts
 - final_ME:        Train and test ME

 Example in the end of this file

 Author: Victor Ivamoto, with code from C.A.M. Lima and B. Kemmer
 Reference before each method
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

##################################################
# Step 1
# Create 3 plots to select best lag in time series dataset
##################################################
def best_lag(ts, pdir):
    """
    Create 3 plots to select best lag in time series dataset.
    
    Args:
        ts:         time series dataset
        pdir:       directory to save plots, if not empty

    Returns:
        - time series x lagged time series plot
        - correlation heatmap plot
        - correlation line chart
    """

    #-------------------------
    # 1. Plot time series x lagged time series
    #-------------------------
    end = 50
    X = ts[:end]
    plt.figure()
    for lag in range(12):
        X1 = ts[lag:end+lag]
        plt.subplot(4, 3, 1 + lag)
        plt.plot(np.arange(len(X)), X, np.arange(len(X1)), X1)
        plt.title("Lag: " + str(lag))
    plt.show()

    #-------------------------
    # 2. Create several lagged time series
    #-------------------------
    # Ref: Bruno Kemmer
    df = pd.DataFrame(ts)
    df.columns = ['y']

    lags = 25
    for i in range(1, lags + 1):
        df['y_' + str(i)] = df['y'].shift(i)
    df = df.dropna()

    #-------------------------
    # 3 Correlation heatmap plot
    #-------------------------
    # Ref: Bruno Kemmer
    plt.figure()
    plt.title('Correlation among variables and delays.')
    sns.heatmap(df.corr())
    # Uncomment to save plot
    #plt.savefig(pdir + 'corr_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()

    #-------------------------
    # 4 Correlation line chart
    #-------------------------
    # Ref: Bruno Kemmer
    xrange = np.arange(lags + 1)
    plt.figure()
    plt.plot(xrange, df.corr()['y'])
    plt.axhline(y=0.1, color='y', linestyle='--', lw=.5)
    plt.axhline(y=-0.1, color='y', linestyle='--', lw=.5)
    plt.axhline(y=0.05, color='r', linestyle='--', lw=.5)
    plt.axhline(y=-0.05, color='r', linestyle='--', lw=.5)
    plt.axvline(x=20, color='b', linestyle='--', lw=0.5)
    plt.title('Correlation Between Input and Delayed Input')
    plt.xlabel('Lag')
    plt.ylabel('Y Correlation')
    if pdir != '':
        plt.savefig(pdir + 'corr2.png', dpi=300, bbox_inches='tight')
    plt.show()

    return

##################################################
# Step 2
# Cross validation
##################################################
# Select best number of experts using cross validation
# Plot the number of experts vs likelihood
# Ref: Committee machines: a unified approach using support vector machines
# C. A. M. Lima
def cv_experts(ts, lag = 20, gating = 'linear', pdir = ''):
    """
    Select best number of experts using cross validation
    Plot the number of experts vs likelihood

     Input
        ts:         time series dataset
        lag:        time lag used to build the (x, y) dataset
        gating:     gating type 'linear' or 'gaussian'
    Output:
        best number of experts
    """

    kfolds  = 5         # number of cross-validation folds
    bestm   = 1         # best number of experts
    lik_max = 0         # best likelihood

    #------------------------------------------
    # 1. Convert time series to  (X, Y) dataset
    #------------------------------------------
    # for each Y value, keep 'lag' values of X
    N = ts.shape[0]
    y = ts[lag: N]
    y = np.array([y]).T
    X = np.zeros((N-lag, lag))
    for i in range(lag, N):
        X[i-lag,:] = ts[i - lag:i]

    N = X.shape[0]

    #------------------------------------------
    # 2. Find best number of experts using cross validation
    # ------------------------------------------
    # 2.1 Estimate number of experts (eq 3.102)
    plotData = pd.DataFrame(columns=['Number of Experts', 'Likelihood'])
    #mmax = int(np.ceil((N - lag) / (2 * (lag + 1))))
    for m in range(2, 10):

        # 2.2 Cross validation
        lik_vec = np.array([])
        for k in range(kfolds):
            # Define K% of rows for validation set
            rv = np.array(range(int(N * k / kfolds),
                                int(N * (k + 1) / kfolds)))

            # Define complementary row numbers for train set
            r = np.setdiff1d(np.array(range(X.shape[0])), rv)

            # Create the train and validation sets
            X_train = X[r]
            y_train = y[r]
            X_val   = X[rv]
            y_val   = y[rv]

            # Insert column of 1s for bias
            X_train = np.insert(X_train, 0, 1, axis=1)
            X_val   = np.insert(X_val, 0, 1, axis=1)

            # Train the model with mixture of experts (ME)
            wexp, wgat, var, alpha, gamma, sigma = ME(X_train, y_train, m = m, gating = gating, add_bias = False)

            # 2.4 Compute gating and expert outputs
            if gating == 'linear':
                ygat = softmax(X_val @ wgat.T)              # gating output
            elif gating == 'gaussian':
                p_xv = prob(X_val, alpha, gamma, sigma)     # P(x|v) - slide 129
                ygat = gating_output(X_val, p_xv, alpha)    # gating output - slide 129
            yexp = calc_saida_esp(X_val, wexp)              # expert output

            # Calculate the likelihood
            p   = calc_prob_exp(yexp, y_val, var)
            lik = likelihood(ygat, p)

            # Update the likelihood vector
            lik_vec = np.insert(arr = lik_vec, obj = lik_vec.shape, values = lik)

        # Cross-validation likelihood is the
        # mean value of all k-folds
        cv_lik = np.mean(lik_vec)

        # Keep best cross validation values
        lik_max = cv_lik if m == 1 else lik_max     # 1st iteration likelihood
        if cv_lik > lik_max:
            lik_max = cv_lik    # highest likelihood
            bestm = m           # best number of experts

        # Keep data for plot
        plotData.loc[len(plotData) + 1] = [m, lik]

        #print("Number of experts: ", str(m), "CV Likelihood: ", cv_lik)

    #------------------------------------------
    # 3 Plot
    #------------------------------------------
    title = gating.title() + ' Gating - ' + str(lag) + ' Lags'
    plt.figure()
    plt.plot('Number of Experts', 'Likelihood', data=plotData)
    plt.xlabel('Number of Experts')
    plt.ylabel('Likelihood')
    plt.title(title)
    if pdir != '':
        plt.savefig(pdir + gating + '_' + str(lag) + '_cv_experts.png', dpi=300, bbox_inches='tight')
    plt.show()
    return bestm

##################################################
# Step 3
# Mixture of experts
##################################################
# Train and predict the model, then creates a plot
def final_ME(ts, m, lag = 20, gating = 'linear', pdir = ''):
    """
    Train and predict the model, then creates a plot.
    
    Args:
        ts:         time series dataset
        m:          number of experts
        lag:        time series lag
        gating:     gating type: 'linear' or 'gaussian'
        pdir:       directory to save plot

    Returns:
        plots
    """

    #------------------------------------------
    # 1. Create (X, Y) dataset
    #------------------------------------------
    # Convert time series in (X, Y) dataset
    # for each Y, use 'lag' number of values for X
    N = ts.shape[0]
    y = ts[lag: N]
    y = np.array([y]).T
    X = np.zeros((N-lag, lag))
    for i in range(lag, N):
        X[i-lag,:] = ts[i - lag:i]

    N = X.shape[0]

    #------------------------------------------
    # 2. Create train and test sets
    #------------------------------------------
    # Create train and test sets (80% train, 20% test)
    mid = int(np.floor(0.8 * N))
    X_train = X[0:mid,:]
    y_train = y[0:mid]
    X_test  = X[mid:N,:]
    y_test  = y[mid:N]

    # Insert column of 1s for bias
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test  = np.insert(X_test, 0, 1, axis=1)

    #------------------------------------------
    # 3 Train and predict the model
    #------------------------------------------
    wexp, wgat, var, alpha, gamma, sigma = ME(X_train, y_train, m = m,
                                              gating = gating, add_bias = False)

    # Predict ME output
    if gating == 'linear':
        ygat = softmax(X_test @ wgat.T)                     # gating output
    elif gating == 'gaussian':
        p_xv = prob(X_test, alpha, gamma, sigma)            # P(x|v)
        ygat = gating_output(X_test, p_xv, alpha)           # gating output
    yexp = calc_saida_esp(X_test, wexp)                     # expert output
    y_hat = np.sum(ygat * yexp, axis = 1, keepdims=True)    # ME output

    # Compute performance metrics
    p = calc_prob_exp(yexp, y_test, var)
    lik  = likelihood(ygat, p)
    # Eror
    error = y_test - y_hat
    # Mean Absolute Error - MAE
    mae = np.mean(abs(ygat - y_hat), axis = 0)
    # Mean Squared Error - MSE
    mse =  np.mean((ygat - y_hat)**2, axis = 0)
    # Root Mean Squared Error - RMSE
    rmse = np.sqrt(mse)

    result = pd.DataFrame(data = {'Expert': np.arange(1,m+1),
                                  'Likelihood': lik,
                                  'MAE': mae,
                                  'MSE': mse,
                                  'RMSE': rmse})

    #result.to_csv(pdir + 'result_' + gating + '.csv', index=False, sep = '\t')
    #------------------------------------------
    # 4 Plot predicted and real values
    #------------------------------------------
    xrange = np.arange(N-mid)
    plt.figure()
    plt.plot(xrange, y_test, xrange, y_hat)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.suptitle('Real vs Predicted Time Series')
    plt.title(gating.title() + ' Gating, ' + str(lag) + ' Lags, ' + str(m) + ' Experts')
    if pdir != '':
        plt.savefig(pdir + gating + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    xrange = np.arange(y_test.size)

    # Plot gates
    plt.figure(figsize=(9, 12))
    for i in range(m):
        plt.subplot(m+1,1,i+1)
        plt.plot(xrange, ygat[:,i])
        plt.ylabel('Gate ' + str(i+1))

    plt.xlabel('Test set')
    plt.suptitle(gating.title() + ' Gating, ' + str(lag) + ' Lags, ' + str(m) + ' Experts')
    if pdir != '':
        plt.savefig(pdir + 'gate_' + gating + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot experts
    plt.figure(figsize=(9, 12))
    for i in range(m):
        plt.subplot(m+1,1,i+1)
        plt.plot(xrange, yexp[:,i])
        plt.ylabel('Expert ' + str(i+1))

    plt.xlabel('Test set')
    plt.suptitle(gating.title() + ' Gating, ' + str(lag) + ' Lags, ' + str(m) + ' Experts')
    if pdir != '':
        plt.savefig(pdir + 'expert_' + gating + '.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Plot data, output and error
    plt.figure(figsize=(9, 12))
    plt.subplot(311)
    plt.plot(xrange, y_test)
    plt.ylabel('Data')

    plt.subplot(312)
    plt.plot(xrange, y_hat)
    plt.ylabel('Output')

    plt.subplot(313)
    plt.plot(xrange, error)
    plt.ylabel('Error')

    plt.xlabel('Test set')
    plt.suptitle(gating.title() + ' Gating, ' + str(lag) + ' Lags, ' + str(m) + ' Experts')
    if pdir != '':
        plt.savefig(pdir + 'data_' + gating + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    return


def ME(X, Yd, m = 4, gating = 'linear', add_bias = False):
    """
    Train mixture of experts.
    
    Input
        X:          input matrix
        Yd:         desired ME output
        m:          number of experts
        add_bias:   if True, add column of ones in X matrix

    Returns
        wexp:       expert weights
        wgat:       gating weights
        var:        variance
        alpha:      gaussian gating parameter
        gamma:      gaussian gating parameter
        sigma:      gaussian gating parameter
    """
    # Dimensions:
    # wexp (ns x ne x m)
    # wgat (m x ne)
    # yexp (N, ns, m)
    # ygat (N x m)
    # g = ygat = (N x m)
    # p (N x m)
    # h (N x m) ?

    # Add column with 1s for the bias
    if add_bias:                        # add column of 1s in X matrix for bias
        X = np.insert(X, 0, 1, axis=1)
    if Yd.ndim == 1:                    # convert row vector to column vector
        Yd = np.array([Yd]).T
    N, ne = X.shape                     # number of instances and features
    eps = np.finfo(float).eps           # small number to avoid nan and division by zero

    ns    = Yd.shape[1]                 # number of ME outputs
    var   = np.ones(m)                  # adaptative variance
    wexp  = np.random.rand(ns, ne, m)   # expert weight
    wgat  = np.random.rand(m, ne)       # linear gating weight
    ygat  = softmax(X @ wgat.T)         # gating output (a priori probability)
    yexp  = calc_saida_esp(X, wexp)     # expert output

    j = random.sample(list(np.arange(N)), N)
    alpha = 1/m * np.ones((m,1))        # gaussian gating parameter
    gamma = X[j[:m],:]                  # gaussian gating parameter
    sigma = np.zeros((ne, ne, m))       # gaussian gating parameter
    for i in range(m):
        sigma[:,:,i] = np.eye(ne, ne)   # gaussian gating parameter


    p = calc_prob_exp(yexp, Yd, var)    # P(y|X,theta)
    p_xv = np.zeros((N, m))             # P(X|v)
    lik_old = 0
    lik_new = likelihood(ygat, p)
    it = 0
    maxiter = 100
    while abs(lik_new - lik_old) > 1e-3 and it < maxiter:
        #print('lik = ', lik_new, "diff: ", lik_new - lik_old)
        # Linear gating
        if gating == 'linear':
            # Step E - Estimate H
            p = calc_prob_exp(yexp, Yd, var)    # P(y|X,theta)
            h = prob_h(ygat, p)
            # Step M: optimize gating and expert outputs
            for i in range(m):
                wexp[:,:,i], var[i] = atualiza_exp(X, Yd, h[:, i], var[i], wexp[:,:,i], ns)
            wgat = atualiza_gat_linear(X, h, wgat) # gating weights
            ygat = softmax(X @ wgat.T)      # gating output
            yexp = calc_saida_esp(X, wexp)  # expert output

            lik_old = lik_new
            lik_new = likelihood(ygat, p)
        # Gaussian gating
        else:
            # Step E:
            ygat = prob(X, alpha, gamma, sigma)
            p = calc_prob_exp(yexp, Yd, var)
            h = alpha.T * ygat * p / np.sum(eps + alpha.T * ygat * p, axis = 1, keepdims=True)
            # Step M: optimize gating and expert outputs
            for i in range(m):
                wexp[:,:,i], var[i] = atualiza_exp(X, Yd, h[:, i], var[i], wexp[:,:,i], ns)
            alpha = np.mean(h, axis =0, keepdims = True).T
            gamma = h.T @ X / np.sum(h + eps, axis=0, keepdims=True).T
            for i in range(m):
                dif = X - gamma[i]
                sigma[:,:,i] = (dif / np.sum(h[:,i]) * h[:,i:i+1]).T @ dif

            #ygat = gating_output(X, p_xv, alpha)    # gating output
            yexp = calc_saida_esp(X, wexp)          # expert output

            lik_old = lik_new
            lik_new = likelihood(ygat, p)

            if it > 15 and gating == 'gaussian':
                if lik_new < lik_old:
                    print("Doesn't converge.")
                elif lik_new < lik_old + 5:
                    print("Converge.")
                break

        #print('lik = ', lik_new, "diff: ", abs(lik_new - lik_old))
        it += 1

    return wexp, wgat, var, alpha, gamma, sigma


# Update gating output
# Ref: Aula 09 - Comite de Maquinas
# C.A.M.Lima
def atualiza_gat_linear(X, h, w):
    # Use gradient ascent to update the gating weights
    # until the gradient is near zero, i.e. reach a local maximum.
    Y = X @ w.T                 # gating is linear combination of input
    g = softmax(Y)              # gating output (a priori probability)
    dQdE = (h - g)              # slide 129
    dQdw = dQdE.T @ X
    lr = 0.1                    # learning rate
    it = 0                      # iteration counter
    maxiter = 1500              # maximum iterations
    while np.linalg.norm(dQdw) > 1e-3 and it < maxiter:
        it = it + 1
        w = w + lr * dQdw       # update weights array
        Y = X @ w.T             # update gating output
        g = softmax(Y)          # gating output (a priori probability)
        dQdE = (h - g)          # Q derivative w.r.t. gating output
        dQdw = dQdE.T @ X       # Q derivative w.r.t. weights
    return w

1# Ref: Aula 09 - Comite de Maquinas - slide 113
# C.A.M.Lima
def atualiza_exp(X, Yd, h, var, w, ns):
    """
    Use gradient ascent to update the expert weights until the gradient is
    near zero, i.e. reach a local maximum.
    
    Args:
        X:      trian set
        Yd:     train set, desired output
        h:      posterior probability
        var:    variance
        w:      expert weights
        ns:     number of outputs

    Returns:

    """

    h = np.array([h]).T
#    y = X @ w.T
#    dQdu = (h / var) * (Yd - y)     # slide 129
#    dQdw = dQdu.T @ X               # slide 129
#    lr = 0.05                       # learning rate
#    it = 0                          # iteration counter
#    maxiter = 1000                    # maximum iterations
#    while np.linalg.norm(dQdw) > 1e-3 and it < maxiter:
#        it = it + 1
#        # compute learning rate
#        #lr = calc_lr(X, w, d = dQdw, h = h, Yd = Yd, var = var)
#        w = w + lr * dQdw               # update weights
#        y = X @ w.T                     # update expert output
#        dQdu = (h / var) * (Yd - y)     # slide 129
#        dQdw = dQdu.T @ X
#        dQdw = np.sum(dQdu * X, axis = 0, keepdims=True)

    # Weights update - Least squares
    Q = (h / var)
    invdesign = np.linalg.pinv((X * Q).T @ X)
    w = (invdesign @ (X * Q).T @ Yd).T

    # Compute variance - slide 129
    var = (1 / ns) * h.T @ (Yd - X @ w.T) ** 2 / np.sum(h, axis = 0)
    var = np.maximum(var, 1e-6)   # limit variance size to avoid nan and division by zero

    return w, var


# Posterior probability
# Ref: Aula 09 - Comite de Maquinas - slide 113
# C.A.M.Lima
def likelihood(g, p):
    return np.sum(np.log(np.sum(g * p, 1)), axis = 0)


# Posterior probability
# Ref: Aula 09 - Comite de Maquinas - slide 129
# C.A.M.Lima
def prob_h(g, p):
    # g = ygat = (N x m)
    # p (N x m)
    # h (N x m)
    num = g * p
    den = np.sum(num, axis = 1, keepdims = True)
    h = num / den
    return h


# Expert probability P(y|x,theta)
# Ref: Aula 09 - Comite de Maquinas - slide 129
# C.A.M.Lima
def calc_prob_exp(yexp, Yd, var):
    ns = Yd.shape[1]

    p = np.exp(-(Yd - yexp)**2 /(2*var)) / ((2*np.pi*var) ** (ns / 2))

    return p

# Stable version of softmax prevents overflow for large values of s
# Ref: Eli Bendersky Website
# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
def softmax(s, axis = 1):
    # ygat (N x m)
    max_s = np.max(s, axis=axis, keepdims=True)
    e = np.exp(s - max_s)
    ygat = e / np.sum(e, axis=axis, keepdims=True)

    return ygat

# Ref: code from C.A.M.Lima
# Expert output - linear function
def calc_saida_esp(X, wexp):
    # yexp (N, ns, m)
    N = X.shape[0]
    ns, ne, m = wexp.shape
    yexp = np.zeros((N, m))
    for i in range(m):
        yexp[:,i:i+1] = X @ wexp[:,:,i].T
    return yexp

def calc_saida_esp1(X, y, w, h, var):
    for i in range(m):
        sumh = np.sum(h[:, i])
        Q = (h[:, i] / (var[i] ** 2))
        invdesign = np.linalg.pinv((X * Q).T @ X)
        w[:,:,i] = (invdesign @ (X * Q).T @ Y).T

        modelerr = y - X @ w
        smallnumber = 0.0001 # razoes numericas
        var[i] = smallnumber + np.sqrt(np.sum(h[:, i].T @ (modelerr**2 ))/sumh)

    return w, var
# Gaussian gating output
# Ref: Aula 09 - Comite de Maquinas - slide 124
# C.A.M.Lima
def gating_output(X, p_xv, alpha):
    return alpha.T * p_xv / (p_xv @ alpha)


# Gaussian gating probability P(x|v) and output
# Ref: Aula 09 - Comite de Maquinas - slide 124
# C.A.M.Lima
def prob(X, alpha, gamma, sigma):
    # Input
    # X: train data
    # alpha, gamma, sigma: gating parameters
    N, ne = X.shape             # number of instances and features
    m = len(gamma)              # number of experts
    eps = np.finfo(float).eps   # Very small number - avoid dividion by zero
    dist = np.zeros((N,1))
    p_xv = np.zeros((N, m))     # P(x|v)
    gate = np.zeros((N, m))     # gating output
    for i in range(m):
        dif = (X - gamma[i:i+1, :])
        inv = np.linalg.pinv(sigma[:, :, i])
        for t in range(N):
            dist[t,:] = dif[t:t + 1, ] @ inv @ dif[t:t + 1, ].T
        p_xv[:, i:i+1] = np.exp(-dist/2) / (eps + ((2 * np.pi) ** (ne / 2)) *
                                            np.sqrt(np.linalg.det(sigma[:, :, i]) + eps))
        gate[:, i:i + 1] = alpha[i] * p_xv[:, i:i + 1]

#    gate[:,i:i+1] = alpha[i] * p_xv[:,i:i+1] / (p_xv @ alpha)
    gate = gate / np.sum(gate, axis = 1, keepdims = True)
    
    return gate

# Compute gradient descent learning rate
# Use of bissection method
# Ref: C. A. M. Lima.
def calc_lr(X, w, d, h, Yd, var):
    # Inputs
    # X:
    # w: weights
    # d: direction to objective function minimum
    # h: ME posteriori probability
    # Yd: desired output
    # var: variance
    np.random.seed(1234)
    epsilon = 1e-3          #
    hlmin = 1e-3            #
    lr_l = 0  # Lower lr    # lower learning rate
    lr_u = np.random.rand() * 1e-12  # Upper learning rate

    # New w position
    wn = w + lr_u * d
    y = X @ wn.T
    # Calculate new position gradient
    g = exp_derivative(X, Yd, y, h, var)

    hl = g @ d.T
    while hl < 0:
        # Double upper limit until convergence
        lr_u = 2 * lr_u
        # Calculate the new position
        wn = w - lr_u * d
        y = X @ wn.T
        # Calculate the gradient of new position
        g = exp_derivative(X, Yd, y, h, var)
        hl = g @ d.T

    # lr medium is the average of learning rates
    lr_m = (lr_l + lr_u) / 2
    # Estimate the maximum number of iterations
    maxiter = np.ceil(np.log((lr_u - lr_l) / epsilon))
    it = 0
    while abs(hl) > hlmin and it < maxiter:
        # Calculate new position
        wn = w - lr_m * d
        y = X @ wn.T

        # Calculate the gradient of the new position
        g = exp_derivative(X, Yd, y, h, var)

        hl = g @ d.T
        if hl > 0:
            lr_u = lr_m     # Decrease upper lr
        elif hl < 0:
            lr_l = lr_m     # Increase lower lr
        else:
            break
        # lr medium is the lr average
        lr_m = (lr_l + lr_u) / 2
        # Increase number of iterations
        it = it + 1

    return lr_m

# Expert derivative
def exp_derivative(X, Yd, y, h, var):
    dQdu = (h / var) * (Yd - y)  # slide 129
    dQdw = dQdu.T @ X
    return dQdw

# Gating derivative
def gat_derivative(X, y, h):
    g = softmax(y)      # gating output (a priori probability)
    dQdE = (h - g)      # Q derivative w.r.t. gating output
    dQdw = dQdE.T @ X   # Q derivative w.r.t. weights
    return dQdw


#============================
# Demo
#============================
if __name__ == '__main__':

    # ------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------
    ts = np.loadtxt("data/train_set.txt")

    # ------------------------------------------------
    # 2. Create charts to select the best time series lag
    # ------------------------------------------------
    best_lag(ts, pdir = '')
    lag = 6

    # ------------------------------------------------
    # 3. Cross validation for choosing the number of experts
    # ------------------------------------------------
    cv_experts(ts = ts,  lag = lag, gating = 'linear', pdir = '')
    cv_experts(ts = ts,  lag = lag, gating = 'gaussian', pdir = '')

    # ------------------------------------------------
    # 4. Train and test ME
    # ------------------------------------------------
    final_ME(ts = ts, m = 6, lag = lag, gating = 'linear',   pdir = '')
    final_ME(ts = ts, m = 2, lag = lag, gating = 'gaussian', pdir = '')
    print(1)
