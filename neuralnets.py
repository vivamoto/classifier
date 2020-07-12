import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================
# 3.3 Logistic: Theta is the sigmoid function used in
# binary logistic regression
# ==================================
def sigmoid_unstable(s):
    # Slide 46
    sigmoid = np.exp(s) / (1 + np.exp(s))
    return sigmoid


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid(s):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (s >= 0)
    neg_mask = (s < 0)
    z = np.zeros_like(s)
    z[pos_mask] = np.exp(-s[pos_mask])
    z[neg_mask] = np.exp(s[neg_mask])
    top = np.ones_like(s)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


# +++++++++++++++++++++++++++++++++
# 5. Neural networks
# - Single Layer Perceptron (SLP)
# - Multi Layer Perceptron (MLP)
# +++++++++++++++++++++++++++++++++
# ==================================
# 5.1 Single Layer Perceptron (SLP)
# ==================================
# Single layer perceptron (SLP) with softmax activation
# function and cross entropy cost function
def SLP(X, yd, error_type=2, maxiter=1000, plot=True, DS_name='', pdir = ''):
    # Input:
    # X: features matrix
    # yd: desired output
    # error_type: 1 = mean squared error
    #             2 = cross entropy
    # plot: plot error vs iteration
    # Output:
    # w: weight matrix
    np.random.seed(1234)

    # Convert Yd to [0, 1]
    yd = np.where(yd == -1, 0, yd)
    if yd.ndim == 1:
        yd = np.array([yd]).T

    # Convert Yd to 1 of n
    if yd.shape[1] == 1:
        yd = np.hstack((yd, np.ones((np.size(yd, 0), 1)) - yd))

    # N: number of instances
    # m: number of features
    N, m = X.shape
    nc = np.size(yd, 1)  # Number of classes
    w = np.random.rand(nc, m)  # Weight matrix
    y = softmax(X @ w.T, axis=1)
    if (error_type == 1):
        error = y - yd
        MSE = 1 / N * np.sum(error ** 2)  # Mean Squared Error
    elif (error_type == 2):
        error = - np.sum(yd * np.log(y + 1e-6))
        MSE = error / N

    it = 0  # Iteration counter
    alpha = 0.5  # Learning rate
    plotData = pd.DataFrame(columns=['Iteration', 'MSE'])  # Vector of MSE
    while ((MSE > 1e-5) & (it < maxiter)):
        it += 1
        # Compute direction of steepest decline
        dJdw, dJdb = SLP_derivative(X, yd, y)
        # Update weight
        w = w - alpha * dJdw
        # Update perceptron output
        y = softmax(X @ w.T, axis=1)
        if (error_type == 1):
            error = y - yd
            MSE = 1 / N * np.sum(error ** 2)
        elif (error_type == 2):
            # Cross-entropy error function
            error = - np.sum(yd * np.log(y + 1e-6))
            MSE = error / N

        plotData.loc[len(plotData) + 1] = [it, MSE]
    # Plot error vs iteration
    if plot:
        plt.plot('Iteration', 'MSE', data=plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('SLP - ' + DS_name)
        plt.savefig(pdir + DS_name + '_SLP.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Save parameters to file
    plotData.to_csv(pdir + DS_name + '_SLP.csv', index=False, decimal=',', sep='\t')

    return w


# Stable softmax function
def softmax(s, axis=1):
    max_s = np.max(s, axis=axis, keepdims=True)
    e = np.exp(s - max_s)
    y = e / np.sum(e, axis=axis, keepdims=True)
    return y


def calc_grad_soft(X, yd, w, nc, m, N, error_type):
    # This function is not in use
    yin = X @ w.T
    max_yin = np.max(yin)
    y = np.exp(yin - max_yin) / (np.sum(np.exp(yin - max_yin), 1) * np.ones((1, N))).T
    if (error_type == 1):
        erro = y - yd
    elif (error_type == 2):
        # Cross-entropy error function
        erro = yd * np.log10(y)

    dJdw = np.zeros((nc, m))
    for n in range(N):
        for i in range(nc):
            for j in range(m):
                for k in range(nc):
                    if (i == k):
                        if (error_type == 1):
                            dJdw[i, j] = dJdw[i, j] + erro[n, k] * (1 - y[n, k]) * y[n, i] * X[n, j]
                        elif (error_type == 2):
                            # dJdw(i,j)  = dJdw(i, j) + (yd(n,k)  *    log10(y(n,k)) / y(n,k)) * (1 - y(n, k)) * y(n, i) * X(n,j);
                            dJdw[i, j] = dJdw[i, j] + (yd[n, k] / y[n, k]) * (1 - y[n, k]) * y[n, i] * X[n, j]
                    else:
                        if (error_type == 1):
                            dJdw[i, j] = dJdw[i, j] + erro[n, k] * (0 - y[n, k]) * y[n, i] * X[n, j]
                        elif (error_type == 2):
                            dJdw[i, j] = dJdw[i, j] + (yd[n, k] / y[n, k]) * (0 - y[n, k]) * y[n, i] * X[n, j]
    dJdw = 1 / N * dJdw
    return dJdw


# -----------------
# 5.2 SLP Derivative
# Softmax and cross entropy
# -----------------
def SLP_derivative(X, yd, y):
    # Convert Y and Yd to [0, 1]
    yd = np.where(yd == -1, 0, yd)
    y = np.where(y == -1, 0, y)

    N = X.shape[0]  # number of instances

    dJtdb = np.mean((y - yd), axis=0, keepdims=True).T
    dJtdw = ((y - yd).T @ X) / N

    return dJtdw, dJtdb


# +++++++++++++++++++++++++++++++++
# 5.3 Multi Layer Perceptron (MLP)
# +++++++++++++++++++++++++++++++++
# Softmax activation function and cross entropy
# cost function
# 1st layer: sigmoid
# 2nd layer: softmax
# cost function: cross entropy
def mlp(X, yd, L, maxiter=1000, plot=True, DS_name='', pdir = ''):
    # Input
    # X: X_train
    # yd: desired y (y_train)
    # L: number 1st layer neurons
    # maxiter: maximum number of iterations

    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)

    # Transform yd to 1 of n classes
    yd = np.where(yd == -1, 0, yd)
    if yd.shape[1] == 1:
        yd = np.array([(yd[:, 0] == 1) * 1, (yd[:, 0] == 0) * 1]).T

    # N: number of instances
    # m: number of features
    # nc: number of classes and 2nd layer neurons
    N, m = X.shape
    nc = yd.shape[1]

    # A and B are weight matrices of 1st and 2nd layers
    # Initialize A and B with random values
    np.random.seed(1234)
    A = np.random.rand(L, m)
    B = np.random.rand(nc, L)

    v = X @ A.T  # 1st layer input
    z = sigmoid(v)  # 1st layer output
    u = z @ B.T  # 2nd layer input
    y = softmax(u, axis=1)  # 2nd layer output

    MSE = 1 / N * np.sum((y - yd) ** 2)  # Mean Squared Error

    it = 0  # Iteration counter
    lr = 0.5  # Learning rate
    plotData = pd.DataFrame(columns=['Iteration', 'MSE'])  # Vector of MSE
    while (MSE > 1e-5) and (it < maxiter):
        it += 1
        # Compute direction of steepest decline
        dJdA, dJdB = MLP_derivative(X, yd, A, B)
        # Update learning rate
        # lr = calc_lr_mlp(X, yd, A, B, -dJdA, -dJdB)
        # lr = calc_alfa(X, yd, A, B, -dJdA, -dJdB)
        # Update weight matrices
        A = A - lr * dJdA
        B = B - lr * dJdB
        # Update MLP output
        v = X @ A.T  # 1st layer input
        z = sigmoid(v)  # 1st layer output
        u = z @ B.T  # 2nd layer input
        y = softmax(u, axis=1)  # 2nd layer output
        # Update the error
        MSE = 1 / N * np.sum((y - yd) ** 2)  #
        plotData.loc[len(plotData) + 1] = [it, MSE]
    # Plot error vs iteration
    if plot:
        plt.plot('Iteration', 'MSE', data=plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('MLP ' + str(L) + ' Neurons - ' + DS_name)
        plt.savefig(pdir + DS_name + '_MLP_' + str(L) + '_neurons.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Save parameters to file
    plotData.to_csv(pdir + DS_name + '_MLP_' + str(L) + '_neurons.csv', index=False, decimal=',', sep='\t')

    return A, B


# ==================================
# 5.3.1 MLP: Cross validation
# ==================================
def cv_mlp(X_train, y_train, L=20, maxiter=1000, K=5, plot=True, DS_name='', pdir = ''):
    # Input
    # L: number of neurons
    # maxiter: maximum number of iterations
    # K: number of cross-validation folds
    # plot: plot chart if True
    # DS_name: dataset name
    # Output
    # l_max: number of neuros of maximum cross validation accuracy
    N, d = X_train.shape
    acc_max = 0
    l_max = 0
    # Data to create plot
    plotData = pd.DataFrame(columns=['L', 'Accuracy'])

    # ------------------------
    # 1. Test several number of neurons
    #    using cross-validation
    # ------------------------
    for l in range(1, L, 2):
        acc_vec = np.array([])
        for k in range(K):
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
            A, B = mlp(X1, y1, L=l, maxiter=maxiter, plot=False, DS_name=DS_name)
            # Predict
            y_hat = fx_mlp(X2, A, B)
            # Compute the accuracy
            acc = (100 * np.mean(y_hat == y2)).round(2)

            y_hat = fx_mlp(X1, A, B)
            # Compute the accuracy
            acc = (100 * np.mean(y_hat == y1)).round(2)

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
            acc_max = cv_acc  # Best accuracy achieved
            l_max = l  # Number of neurons of best accuracy

    # ------------------------
    # 2. Create plot (# Neurons vs Accuracy)
    # ------------------------
    if plot:
        plt.scatter('L', 'Accuracy', data=plotData)
        plt.xlabel('Number of Neurons')
        plt.ylabel('Cross Validation Accuracy (%)')
        plt.title('CV MLP - ' + DS_name)
        plt.savefig(pdir + DS_name + '_cv_mlp_' + str(L) + '_neurons.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Save parameters to file
    plotData.to_csv(pdir + DS_name + '_cv_mlp_' + str(L) + '_neurons.csv', index=False, decimal=',', sep='\t')

    return l_max


# Predict new values in MLP network
def fx_mlp(X, A, B, add_bias=True):
    # Input
    # X, y: test sets
    # A, B: weight matrices
    # Output:
    # y_hat: predicted values
    # acc: predicted accuracy

    # Insert bias column
    if add_bias:
        X = np.insert(X, 0, 1, axis=1)

    v = X @ A.T  # 1st layer input
    z = sigmoid(v)  # 1st layer output
    u = z @ B.T  # 2nd layer input
    y_hat = softmax(u, axis=1)  # 2nd layer output
    # Convert softmax output to 0 and 1
    y_hat = np.where(y_hat == np.max(y_hat, axis=1,
                                     keepdims=True), 1, 0)
    return y_hat


# ==================================
# 5.3.2 MLP Derivative
# ==================================
def MLP_derivative(X, yd, A, B):
    # Inputs
    # X, yd:
    # A, B: weights matrices
    # Output
    # dJdA, dJdB: A and B derivatives

    # N: number of instances
    # m: number of features
    N, m = X.shape
    nc = yd.shape[1]  # Number of classes
    v = X @ A.T  # 1st layer input
    z = sigmoid(v)  # 1st layer output
    u = z @ B.T  # 2nd layer input
    y = softmax(u)  # 2nd layer output

    # Compute the derivatives
    dJdB = 1 / N * ((y - yd).T @ z)
    dJdA = 1 / N * (((y - yd) @ B) * ((1 - z) * z)).T @ X

    return dJdA, dJdB


# ==================================
# 5.3.3 MLP: Gradient descent learning rate
# ==================================
# Calculate the learning rate using bissection algorithm
# The learning rate is used in MLP for faster convergence
def calc_lr_mlp(X, yd, A, B, dirA, dirB):
    # Inputs
    # X, yd: MLP input and output matrices (train set)
    # A, B: MLP weights matrices
    # dirA, dirB: A and B direction of steepest decline
    # Output
    # lr_m: optimized learning rate

    np.random.seed(1234)
    epsilon = 1e-3
    hlmin = 1e-3
    lr_l = 0  # Lower lr
    lr_u = np.random.rand()  # Upper lr

    # New A and B positions
    An = A + lr_u * dirA
    Bn = B + lr_u * dirB
    # Calculate the gradient of new position
    dJdA, dJdB = MLP_derivative(X=X, yd=yd, A=An, B=Bn)
    g = np.concatenate((dJdA.flatten('F'), dJdB.flatten('F')))
    d = np.concatenate((dirA.flatten('F'), dirB.flatten('F')))
    hl = g.T @ d

    while hl < 0:
        #
        lr_u *= 2
        # Calculate the new position
        An = A + lr_u * dirA
        Bn = B + lr_u * dirB
        # Calculate the gradient of new position
        dJdA, dJdB = MLP_derivative(X=X, yd=yd, A=An, B=Bn)
        g = np.concatenate((dJdA.flatten('F'), dJdB.flatten('F')))
        hl = g.T @ d

    # lr medium is the average of lrs
    lr_m = (lr_l + lr_u) / 2

    # Estimate the maximum number of iterations
    itmax = np.ceil(np.log((lr_u - lr_l) / epsilon))

    it = 0  # Iteration counter
    while np.any(hl) > hlmin and it < itmax:
        An = A + lr_u * dirA
        Bn = B + lr_u * dirB
        # Calculate the gradient of new position
        dJdA, dJdB = MLP_derivative(X=X, yd=yd, A=An, B=Bn)

        g = np.concatenate((dJdA.flatten('F'), dJdB.flatten('F')))
        hl = g.T @ d

        if np.any(hl) > 0:
            # Decrease upper lr
            lr_u = lr_m
        elif np.any(hl) < 0:
            # Increase lower lr
            lr_l = lr_m
        else:
            break
        # lr medium is the lr average
        lr_m = (lr_l + lr_u) / 2
        # Increase number of iterations
        it += 1
    return lr_m


# +++++++++++++++++++++++++++++++++
# MLP Ensemble
# +++++++++++++++++++++++++++++++++
# Ref: Ensemble learning via negative correlation
# Y. Liu, X. Yao
def mlp_ensemble(X, yd, M, L, lamb, ens, itmax=10000, plot=True, DS_name='', pdir = ''):
    # Input
    # X, yd: training set
    # M: number of networks
    # L: number of hidden layer neurons
    # lamb: adjust the penalty strength. 0<= lamb <= 1
    # itmax: maximum number of iterations
    # plot: True to make plot
    # DS_name: dataset name used in plot title
    # Output
    # ens: ensembe parameters with weights

    # Insert bias column
    X = np.insert(X, 0, 1, axis=1)

    # Transform yd to 1 of n classes
    yd = np.where(yd == -1, 0, yd)
    if yd.shape[1] == 1:
        yd = np.array([(yd[:, 0] == 1) * 1, (yd[:, 0] == 0) * 1]).T

    N, m = X.shape  # Number of observations and features
    nc = yd.shape[1]  # Number of classes

    # A and B are weights matrices of 1st and 2nd layers
    # Create A and B matrices with random values
    np.random.seed(1234)
    for i in range(M):
        ens['A_' + str(i)] = np.random.rand(L, m)  # 1st layer weights
        ens['B_' + str(i)] = np.random.rand(nc, L)  # 2nd layer weights

    dEdF = np.random.rand(N, nc, M)  # Error derivative
    E = np.ones((M, nc))  # Error of network i
    p = np.zeros((N, nc, M))  # Penalty function
    F = np.zeros((N, nc))  # Ensemble output
    Fi = np.zeros((N, nc, M))  # Output of network i

    plotData = pd.DataFrame(columns=['Iteration', 'Error'])  # Vector of MSE
    it = 0  # iteration counter
    while abs(np.mean(E)) > 1e-5 and it < itmax:
        # ---------------------
        # 1. Compute each network output
        # ---------------------
        for i in range(M):
            A = ens['A_' + str(i)]
            B = ens['B_' + str(i)]

            # Compute each MLP network output
            v = X @ A.T  # 1st layer input
            z = sigmoid(v)  # 1st layer output
            u = z @ B.T  # 2nd layer input
            y = softmax(u)  # 2nd layer output

            # Compute the derivatives
            dJdA = 1 / N * (((dEdF[:, :, i] * y * (1 - y))) @ B).T * ((1 - z) * z).T @ X
            dJdB = 1 / N * (dEdF[:, :, i] * y * (1 - y)).T @ z

            # Update learning rate
            lr = 0.5  # Learning rate
            # lr = calc_lr_mlp(X, yd, A, B, -dJdA, -dJdB)
            # lr = calc_alfa(X, yd, A, B, -dJdA, -dJdB)

            # Update weight matrices
            A = A - lr * dJdA
            B = B - lr * dJdB

            # Compute each MLP network output
            v = X @ A.T  # 1st layer input
            z = sigmoid(v)  # 1st layer output
            u = z @ B.T  # 2nd layer input
            y = softmax(u)  # 2nd layer output

            # Keep each network output and weights
            Fi[:, :, i] = y
            ens['A_' + str(i)] = A
            ens['B_' + str(i)] = B

        # ---------------------
        # 2. Ensemble output (equation 1)
        # ---------------------
        F = np.mean(Fi, axis=2, keepdims=False)

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
            E[i:i + 1] = np.mean(1 / 2 * (Fi[:, :, i] - yd) ** 2 + lamb * p[:, :, i], axis=0, keepdims=True)

            # ---------------------
            # 5. Derivative of E w.r.t. the output of network i (eq. 4)
            # ---------------------
            dEdF[:, :, i] = (1 - lamb) * (Fi[:, :, i] - yd) + lamb * (F - yd)

        # Keep plot data
        plotData.loc[len(plotData) + 1] = [it, np.mean(E)]
        it += 1

    if plot:
        plt.plot('Iteration', 'Error', data=plotData)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Ensemble via Negative Correlation - ' + DS_name)
        plt.savefig(pdir + DS_name + '_negcor_' + str(M) + '_networks.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Return the ensemble weights

    return ens


# ===================================
# Predict ensemble
# ===================================
def fx_mlp_ensemble(X, y, ens, M):
    # N:  Number of observations
    # nc: Number of classes

    N, nc = y.shape

    y_hat = np.zeros((N, nc))  # Output of network
    # 1. Compute each network output
    for i in range(M):
        y_hat += fx_mlp(X, ens['A_' + str(i)],
                        ens['B_' + str(i)], add_bias=True)

    # 2. Convert softmax output to 0 and 1
    y_hat = np.where(y_hat == np.max(y_hat, axis=1,
                                     keepdims=True), 1, 0)
    return y_hat


# Ref:
# Ensemble Learning Using Decorrelated Neural Networks
# BRUCE E ROSEN
# https://doi.org/10.1080/095400996116820
def ensemble_Rosen(X, yd, M, L, lamb, ens, alternate=False, plot=True, DS_name='', pdir = ''):
    # Input
    # X, yd: training set
    # M: number of networks
    # L: number of hidden layer neurons
    # lamb: The scaling function lambda(t) is either constant or
    #        is time dependent.
    # Add column x0 = 1 for the bias

    # Insert column of 1s for the bias
    X = np.insert(X, 0, 1, axis=1)

    # Transform yd to 1 of n classes
    yd = np.where(yd == -1, 0, yd)
    if yd.shape[1] == 1:
        yd = np.array([(yd[:, 0] == 1) * 1, (yd[:, 0] == 0) * 1]).T

    N, m = X.shape
    nc = yd.shape[1]

    np.random.seed(1234)
    for i in range(M):
        ens['A_' + str(i)] = np.random.rand(L, m)  # 1st layer weights
        ens['B_' + str(i)] = np.random.rand(nc, L)  # 2nd layer weights

    dJdA = np.ones((L, m))
    dJdB = np.ones((nc, L))

    # ------------------------
    # 1. Train the 1st network
    # ------------------------
    y = np.zeros((N, nc, M))  # 2nd layer output

    lr = 0.001  # Learning rate
    maxiter = 10000
    plotData = pd.DataFrame(columns=['Network', 'Iteration', 'MSE'])  # Vector of MSE
    for j in range(M):

        A = ens['A_' + str(j)]  # 1st layer weights
        B = ens['B_' + str(j)]  # 2nd layer weights
        MSE = 1
        it = 0
        while np.linalg.norm(dJdA) + np.linalg.norm(dJdB) > 1e-5 and it < maxiter:
            it += 1
            # ------------------------
            # 1. Compute the error and derivative for network j
            # ------------------------
            s1 = 0
            s2 = 0
            for i in range(j - 1):
                # The correlation penalty function P is the product
                # of the jth and ith network error:
                P = (yd - y[:, :, i]) * (yd - y[:, :, j])  # equation 8

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
                s2 += lamb * d * (yd - y[:, :, i])
            # Error function for an individual network j:
            MSE = 1 / N * np.sum((yd - y[:, :, j]) ** 2 + s1)
            # Derivative of E w.r.t. MLP output y
            dEdy = - np.sum(2 * (yd - y[:, :, j]) + s2, axis=0, keepdims=True)
            # ------------------------
            # 2. Train the network j
            # ------------------------
            v = X @ A.T  # 1st layer input
            z = sigmoid(v)  # 1st layer output
            u = z @ B.T  # 2nd layer input
            y[:, :, j] = softmax(u, axis=1)  # 2nd layer output
            # Compute direction of steepest decline
            dJdA = ((dEdy * (y[:, :, j] * (1 - y[:, :, j])) @ B) * (z * (1 - z))).T @ X
            dJdB = (dEdy * (y[:, :, j] * (1 - y[:, :, j]))).T @ z

            # Update learning rate
            # lr = calc_lr_mlp(X, yd, A, B, -dJdA, -dJdB)
            # lr = calc_alfa(X, yd, A, B, -dJdA, -dJdB)
            # Update weight matrices
            A = A - lr * dJdA
            B = B - lr * dJdB
            # Update MLP output
            v = X @ A.T  # 1st layer input
            z = sigmoid(v)  # 1st layer output
            u = z @ B.T  # 2nd layer input
            y[:, :, j] = softmax(u, axis=1)  # 2nd layer output
            # Update the error
            MSE = 1 / N * np.sum((y[:, :, j] - yd) ** 2)  #
            plotData.loc[len(plotData) + 1] = [j, it, MSE]

        # Save the weights in matrix A and B
        ens['A_' + str(j)] = A
        ens['B_' + str(j)] = B

    if plot:
        for i in range(M):
            plt.plot('Iteration', 'MSE', data=plotData)
            plt.xlabel('Iteration')
            plt.ylabel('Mean Squared Error')
            plt.title('Ensemble Decorrelated NN - ' + DS_name)
            plt.savefig(pdir + DS_name + '_decorrelated_' + str(M) + '_networks.png', dpi=300, bbox_inches='tight')
            plt.show()

    # Save parameters to file
    plotData.to_csv(pdir + DS_name + '_SLP.csv', index=False, decimal=',', sep='\t')

    # Return the weights
    return ens


# ===================================
# Predict ensemble
# ===================================
def fx_ensemble_Rosen(X, y, ens, M):
    # N:  Number of observations
    # nc: Number of classes
    N, nc = y.shape

    nc = 2 if y.shape[1] == 1 else nc
    y_hat = np.zeros((N, nc))  # Output of network
    # 1. Compute each network output
    for i in range(M):
        y_hat += fx_mlp(X, ens['A_' + str(i)],
                        ens['B_' + str(i)], add_bias=True)

    # 2. Convert softmax output to 0 and 1
    y_hat = np.where(y_hat == np.max(y_hat, axis=1,
                                     keepdims=True), 1, 0)
    return y_hat


def softmax_der(x):
    S = softmax(x)
    if S.ndim == 0:
        S_vector = S.reshape(S.shape[0], 1)
    else:
        S_vector = S
    S_matrix = np.tile(S_vector, S.shape[0])
    der = np.diag(S) - S_matrix * S_matrix.T
    return der


# Softmax derivative
def sm_derivative(y):
    N, nc = y.shape
    d = np.zeros((N, nc))
    for i in range(nc):
        for k in range(nc):
            d[i] += y[k] * ((i == k) - y[i])

    return d


def sm_dir(S):
    S_vector = S.reshape(S.shape[0], 1)
    S_matrix = np.tile(S_vector, S.shape[0])
    S_dir = np.diag(S) - (S_matrix * np.transpose(S_matrix))
    return S_dir

# +++++++++++++++++++++++++++++++++
# Mixture of Experts
# +++++++++++++++++++++++++++++++++
# Aula 2020-06-04 - 2:33
# https://drive.google.com/file/d/1H3_kMR29pcZR67AH8gWhTvl92iv2nzU0/view
#Treinar todas as redes em paralelo junto com a gating
#1. inicializa os pesos da rede
#2. calcula H
#3. calcula saida do especialista
##3. calcula v
# calcula g
#clacula h
#atualiza gating
#atualiza tetha
#reculcula v e g, h
# Implementar duas estratégias de mistura de especialista para problema de classificação.
# i) modelos lineares para rede gating e especialista
# ii) gaussianas normalizadas para rede gating e modelos lineares para rede especialista
# Verificar o desempenho dos modelos na predição de serie temporal chaveada e na classificação de dados gaussianos. Ambos os conjuntos de dados encontram-se disponível no tidia.

#==================================
# ME: Linear Model for Gating and Experts
#==================================
import regression as rg
def lmme(X, yd, m):
    # Inputs:
    # X, yd: training set
    # m: number of experts

    # Add column x0 = 1 for the bias
    X = np.insert(X, 0, 1, axis=1)
    N, ne = X.shape
    nc = yd. shape[1]

    # Compute the weights (w) using the formula in lesson 2, slide 31:
    # w = (X_T * X)^-1 * X_T * y
    w = np.linalg.inv(X.T @ X) @ X.T @ y

    # tetha: gating weights (tetha)
    # v:     expert weights
    # u:     expert output
    # roh:   learning rate

    roh = 0.5  # Learning rate
    # Initialize gating and expert weights
    v = np.random.rand(nf, m)       # Gating weights
    tetha = np.random.rand(nf, m)   # Expert weights
    u = X @ tetha       # Expert output
    g = softmax(X @ v)  # Gate output
    p = np.exp(-0.5 * (y[t] - mu[t, i]).T @ (y[t] - mu[t, i]))
    s = np.sum(g[t, :] * np.exp(-0.5 * (y[t] - mu[t, :]).T @ (y[t] - mu[t, :]), axis=1)
    h = (g[t, i] * p / s)

    # Update weights
    tetha = tetha + roh *  h[t, i] * (y[t] - mu[t, i]) * X[t, :].T
    v = v + roh * (h[t, i] - g[t, i]) * X[t, :].T
    np.linalg.inv(X.T @ X) @ X.T @ y

    while likelyhood_new - likelyhood_old > 1e-3:
        # Step E
        sigma = (1 / d) * (np.sum(h * np.linalg(y - u) ** 2, axis=0)) / (np.sum(h, axis=0))
        P = (1 / (2 * np.pi * sigma) ** (d / 2))) * np.exp(-np.linalg.norm(y - u[i, j] * X) / (2 * sigma ** 2))

        # Step M

