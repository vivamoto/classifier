import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# +++++++++++++++++++++++++++++++++
# 6. SVM
# This section defines:
# - Traditional SVM
# - LS-SVM
# - TW-SVM
# +++++++++++++++++++++++++++++++++
# ==================================
# 6.1 SVM: Kernel
# ==================================
# Compute kernel matrix K(x1, x2)
def kernel(x1, x2, t='RBF', b=1, c=1, d=2, sigma=3):
    # t = kernel type
    #   linear : linear
    #   poly   : polynomial
    #   rbf    : radial base gaussian function (RFB)
    #   erbf   : radial base exponencial
    #   tan    : hyperbolic tangent
    #   fourier: Fourier series
    #   lspline: linear splines
    # b: constant multiplier for hyperbolic tangent
    # c: constant sum for polynomial, tangent and linear splines
    # d: polynomial and linear splines power indicator
    # sigma: free paramter for RBF and exponential base
    # One problem with the polynomial kernel is that it may suffer
    # from numerical instability:
    # when x1Tx2 + c < 1, K(x1, x2) = (x1Tx2 + c)^d tends to zero with increasing d,
    # whereas when x1Tx2 + c > 1, K(x1, x2) tends to infinity
    if x1.ndim == 1:
        x1 = np.array([x1]).T

    if x2.ndim == 1:
        x2 = np.array([x2]).T

    m1 = x1.shape[0]
    m2 = x2.shape[0]
    K = np.zeros((m1, m2))

    for k in range(m1):
        for l in range(m2):

            # Linear kernel
            if t == 'Linear':
                K[k, l] = x1[k] @ x2[l]

            # Polynomial kernel
            elif t == 'Polynomial':
                K[k, l] = (x1[k] @ x2[l] + c) ** d

            # Radial base gaussian function (RBF)
            elif t == 'RBF':
                K[k, l] = np.exp(-(x1[k] - x2[l]) @ (x1[k] - x2[l]) / (2 * sigma ** 2))

            # Radial base exponential function
            elif t == 'erbf':
                K[k, l] = np.exp(-np.abs(x1[k] - x2[l]) / (2 * sigma ** 2))
            # Hyperbolic tangent
            elif t == 'tan':
                K[k, l] = np.tanh(b * (x1[k] @ x2[l]) + c)

            # Fourier series
            elif t == 'fourier':
                K[k, l] = np.sin(N + 1 / 2) * (x1[k] - x2[l]) / np.sin((x1[k] - x2[l]) / 2)
            # Linear splines
            elif t == 'lspline':
                K[k, l] = c + x1[k] * x2[l] + x1[k] * x2[l] * min(x1[k], x2[l]) + 1 / 2 * (x1[k] + x2[l]) * min(x1[k],
                                                                                                                x2[
                                                                                                                    l]) ** d

    return K


# =======================
# 6.1 SVM: Prediction
# =======================
def fx_svm(X, X_train, y_train, alpha, SV, t, b, c, d, sigma):
    # Predict SVM
    # Input values:
    # X:
    # X_train, y_train: matrix used for training
    # alpha: Lagrange multipliers
    # SV: support vectors
    # Output:
    #
    if y_train.ndim == 1:
        y_train = np.array([y_train]).T

    y_train = np.where(y_train == 0, -1, y_train)
    y_train = y_train[SV[:, 0]]
    X_train = X_train[SV[:, 0]]
    alpha = alpha[SV[:, 0]]

    # Compute the bias
    K = kernel(X_train, X, t=t, b=b, c=c, d=d, sigma=sigma)
    SV_neg = y_train < 0
    SV_pos = y_train > 0
    bias = (-1 / 2) * (np.max(K[SV_neg[:, 0], :].T @ alpha[SV_neg]) + np.min(K[SV_pos[:, 0], :].T @ alpha[SV_pos]))
    bias = y_train - np.sum(alpha * y_train * K, axis=1, keepdims=True)
    bias = np.mean(bias)

    # Predict new values
    y_hat = np.sign(np.sum(alpha * y_train * K, axis=0, keepdims=True) + bias).T
    y_hat = np.where(y_hat == -1, 0, y_hat)

    return y_hat


# ==================================
# 6.1.2 SVM: Train
# ==================================

def svm(X, y, C=None, t='RBF', b=1, c=1, d=2, sigma=3):
    # Train SVM model
    # Input:
    # X, y: input values
    # C:
    # t: kernel type (see kernel function definition)
    # b, c, d, sigma: kernel parameters (see kernel funciton definition)
    # Output:
    # SV: support vectors
    # alpha: Lagrange multipliers
    # t, b, c, d, sigma: kernel parameters

    y = np.where(y == 0, -1, y)
    N, m = X.shape
    # Compute the Gram matrix
    K = kernel(X, X, t=t, b=b, c=c, d=d, sigma=sigma)
    # Construct P, q, A, b, G, h matrices for CVXOPT
    P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
    P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
    q = cvxopt.matrix(np.ones(N) * -1, tc='d')
    A = cvxopt.matrix(y, (1, N), tc='d')
    b = cvxopt.matrix(0.0, tc='d')
    if C is None or C == 0:  # hard-margin SVM
        G = cvxopt.matrix(np.diag(np.ones(N) * -1), tc='d')
        h = cvxopt.matrix(np.zeros(N), tc='d')
    else:  # soft-margin SVM
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(N) * -1), np.eye(N))), tc='d')
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)), tc='d')
    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    # Lagrange multipliers
    alpha = np.asarray(solution['x'])
    # Support vectors have non zero lagrange multipliers
    SV = alpha > 1e-5  # some small threshold

    return SV, alpha, t, b, c, d, sigma


# +++++++++++++++++++++++++++++++++
# 6.2 LS-SVM
# +++++++++++++++++++++++++++++++++
# Reference:
# Committee machines: a unified approach using support
# vector machines
# Clodoaldo Aparecido de Moraes Lima

def fx_lssvm(X_test, X, y, alpha, bias, kernel_type='Polynomial', b=1, c=1, d=2, sigma=3):
    y = np.where(y == 0, -1, y)

    y_hat = np.sign(np.sum(alpha * y * kernel(X, X_test, t=kernel_type, b=b, c=c, d=d, sigma=sigma), axis=0,
                           keepdims=True) + bias).T
    y_hat = np.where(y_hat == -1, 0, y_hat)

    return y_hat


def lssvm(X, y, C=10, kernel_type='Polynomial', b=1, c=1, d=2, sigma=3):
    y = np.where(y == 0, -1, y)

    N = X.shape[0]
    nc = y.shape[1]
    K = kernel(X, X, t=kernel_type, b=b, c=c, d=d, sigma=sigma)

    # 3. Compute omega
    omega = np.zeros((N, N), int)
    for k in range(K.shape[0]):
        for l in range(K.shape[1]):
            omega[k, l] = y[k] * y[l] * K[k, l]

    # 4. Build Matrix A and vector b
    I = np.eye(omega.shape[0])
    ZZCI = omega + C ** -1 * I

    # 4.1 Build matrix A
    A11 = np.zeros((1, 1))  # Element A(1,1)
    A1 = np.hstack((A11, -y.T))  # Row 1
    A2 = np.hstack((y, ZZCI))  # Row 2

    # Build matrix A
    A = np.vstack((A1, A2))

    # 4.2 Output vector b
    b = np.vstack((np.zeros((1, 1)), np.ones((N, 1))))

    # 5. Solve the linear equation Ax = b
    x = np.linalg.solve(A, b)

    bias = x[0]
    alpha = x[1:len(x)]

    return alpha, bias


# +++++++++++++++++++++++++++++++++
# 6.3 TSVM
# +++++++++++++++++++++++++++++++++
# Ref:
# 1. Twin Support Vector Machines for Pattern Classification
# Jayadeva - Reshma Khemchandani - Suresh Chandra
# In IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,
# VOL. 29, NO. 5, MAY 2007
#
# 2. Twin Support Vector Machines - Models, Extensions and Applications
# Jayadeva - Reshma Khemchandani - Suresh Chandra
# Springer
# ==================================
# 6.3.1 TWSVM: Prediction
# ==================================
# Paper documentation
def fx_twsvm(X, Ct, z1, z2, kernel_type='Polynomial', b=1, c=1, d=2, sigma=3):
    # eq 3.33
    u1 = z1[:-1]
    b1 = z1[-1:]
    u2 = z2[:-1]
    b2 = z2[-1:]

    K_XCt = kernel(X, Ct, kernel_type, b=b, c=c, d=d, sigma=sigma)

    # Define the two surfaces (eq 3.21)
    surface1 = K_XCt @ u1 + b1
    surface2 = K_XCt @ u2 + b2

    # Calculate the distance from X to each surface
    dist1 = abs(surface1)  # class 1
    dist2 = abs(surface2)  # class -1

    # A new data point x ∈ Rn is assigned to class r (r = 1, 2),
    # depending on which of the two planes given by (3.19) it lies closer to.

    y_hat = np.argmax(np.hstack((dist1, dist2)), 1)  # eq 3.20

    if y_hat.ndim == 1:
        y_hat = np.array([y_hat]).T

    return y_hat


# ==================================
# 6.3.2 TWSVM: Prediction
# ==================================
# Prof. Clodoaldo explanation
def fx_twsvm_clodoaldo(X, Ct, z1, z2, kernel_type='Polynomial', b=1, c=1, d=2, sigma=3):
    N = X.shape[0]

    u1 = z1[:-1]
    b1 = z1[-1:]
    u2 = z2[:-1]
    b2 = z2[-1:]

    K_XCt = kernel(X, Ct, kernel_type, b=b, c=c, d=d, sigma=sigma)

    # Define the two surfaces: equation 33
    surface1 = K_XCt @ u1 + b1
    surface2 = K_XCt @ u2 + b2

    # Calculate the distance from X to each surface
    dist1 = abs(surface1) / np.linalg.norm(u1)
    dist2 = abs(surface2) / np.linalg.norm(u2)

    # Initialize y_hat = 2
    y_hat = np.ones((X.shape[0], 1)) * 2
    pos = (surface1 >= 0) & (surface2 >= 0)  # class  1
    neg = (surface1 <= 0) & (surface2 <= 0)  # class -1
    mid = (surface1 <= 0) & (surface2 >= 0)  # X between both surfaces
    y_hat[pos] = 1  # class  1
    y_hat[neg] = 0  # class -1
    y_hat[mid] = np.where(dist1[mid] < dist2[mid], 1, 0)

    return y_hat


# ==================================
# 6.3.3 TWSVM: Solve quadratic problem
# ==================================
def solveQP(P, q, x0, C=None):
    # References:
    # "Quadratic Programming with Python and CVXOPT"
    # https://cvxopt.org/
    # https://web.archive.org/web/20140429090836/http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/
    # https://sandipanweb.wordpress.com/2018/04/23/implementing-a-soft-margin-kernelized-support-vector-machine-binary-classifier-with-quadratic-programming-in-r-and-python/
    # Solve SVM QP optimization problem:
    # min(x) 1/2 * x.T P x + q.T x
    # s.t. Gx<=h
    #      Ax=b
    # Input parameters of CVXOPT library
    N = q.shape[0]
    # construct P, q, G, h matrices for CVXOPT
    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    if C is None or C == 0:  # hard-margin SVM
        G = cvxopt.matrix(np.diag(np.ones(N) * -1), tc='d')
        h = cvxopt.matrix(np.zeros((N, 1)), tc='d')
    else:  # soft-margin SVM
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(N) * -1), np.eye(N))), tc='d')
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)), tc='d')
    # solve QP problem
    x = cvxopt.solvers.qp(P, q, G, h)
    return x


# TWSVM: Solve quadratic problem
# Optimize equations 45 and 47
def minimize(alpha, M):
    g = -np.sum(alpha) + 0.5 * alpha.T @ M @ alpha
    return g


# ==================================
# 6.3.4 TWSVM: Train
# ==================================
# A, B: matrices of points belonging to classes 1 and -1
# e1, e2: vectors of ones of appropriate dimensions
# I: identity matrix of size (n+1) x (n+1)
# epsilon x I: regularization term
def twsvm(X, y, c1=1, c2=1, kernel_type='RBF', optimizer=2, b=1, c=1, d=2, sigma=3):
    # Input:
    # X, y: matrix. y = {-1, 1}
    # c1: alpha max value
    # kernel_type: 1 = linear
    #              2 = polynomial
    #              3 = rbf
    # optimizer: 1 = CVXOPT
    #            2 = Scipy
    # Output:
    # Ct, z1 and z2: parameters for prediction
    #
    y = np.where(y == 0, -1, y)
    if y.ndim == 1:
        y = np.array([y]).T

    A = X[y[:, 0] == 1]
    B = X[y[:, 0] == -1]

    Ct = np.vstack((A, B))

    m1 = A.shape[0]
    m2 = B.shape[0]

    e1 = np.ones((m1, 1))
    e2 = np.ones((m2, 1))

    K_ACt = kernel(A, Ct, kernel_type, b=b, c=c, d=d, sigma=sigma)
    K_BCt = kernel(B, Ct, kernel_type, b=b, c=c, d=d, sigma=sigma)

    # equation 3.33
    S = np.hstack((K_ACt, e1))
    R = np.hstack((K_BCt, e2))
    P1 = R @ np.linalg.pinv(S.T @ S) @ R.T

    # equation 3.36
    L = np.hstack((K_ACt, e1))
    J = np.hstack((K_BCt, e2))
    P2 = L @ np.linalg.pinv(J.T @ J) @ L.T

    # Initial values for alpha and gamma
    alpha0 = np.zeros((np.size(R, 0), 1))
    gamma0 = np.zeros((np.size(L, 0), 1))

    # Lagrange multipliers
    if optimizer == 1:
        # CVXOPT algorithm:
        alpha = solveQP(P1, -e2, x0=alpha0, C=c1)  # eq 3.35
        gamma = solveQP(P2, -e1, x0=gamma0, C=c2)  # eq 3.36

        alpha = np.ravel(alpha['x'])
        gamma = np.ravel(gamma['x'])
    else:
        # Scipy optimize
        b1 = optimize.Bounds(0, c1)
        b2 = optimize.Bounds(0, c2)
        alpha = optimize.minimize(minimize, x0=alpha0, args=P1, method='L-BFGS-B', bounds=b1).x
        gamma = optimize.minimize(minimize, x0=gamma0, args=P2, method='L-BFGS-B', bounds=b2).x

    if alpha.ndim == 1:
        alpha = np.array([alpha]).T

    if gamma.ndim == 1:
        gamma = np.array([gamma]).T

    # Equation 3.34
    epsilon = 1e-16
    I = np.eye(len(S.T @ S))
    z1 = -np.linalg.pinv(S.T @ S + epsilon * I) @ R.T @ alpha

    I = np.eye(len(J.T @ J))
    z2 = np.linalg.pinv(J.T @ J + epsilon * I) @ L.T @ gamma

    return Ct, z1, z2


# +++++++++++++++++++++++++++++++++
# 6.4 Multiclass SVM, TWSVM, LSSVM
# +++++++++++++++++++++++++++++++++
# The best methods are one vs one and DDAG
# Reference:
# A Comparison of Methods for Multi-class Support Vector Machines
# Chih-Wei Hsu and Chih-Jen Lin
def svm_mc(X_train, y_train, X_test, y_test, c1=1, c2=1, model='svm', method='ovo', kernel_type='RBF', optimizer=2, C=1,
           b=1, c=1, d=2, sigma=3):
    # Input
    # model: svm, twsvm, lssvm
    # method: ono = one vs one
    #         ova = one vs all
    # kernel_type: linear, poly, rbf, erbf, tan, fourier, lspline
    # Optimizer: 1 = CVXOPT,
    #            2 = scipy optimize
    # C: SVM soft margin
    # TWSVM parameters (see reference)
    # c1: soft margin 1
    # c2: soft margin 2
    # Kernel parameters (see function definition)
    # b: constant multiplier
    # c: constant adder
    # d: Polynomial degree
    # sigma: RBF kernel sigma
    # Output
    # y_hat: predicted values

    if type(y_train) is not np.ndarray:
        y_train = np.array(y_train)

    # ---------------
    # One vs One
    # ---------------
    if method == 'ovo':
        # Number of instances and classes
        N, nc = y_test.shape

        # i is the positve class
        # j is the negative class
        acc = np.ones((int(nc * (nc - 1) / 2)))
        X2 = X_test
        k = 0
        for i in range(nc):
            for j in range(i + 1, nc):
                # Create the train and test sets
                # X1, y1: train sets
                # X2, y2: test  sets
                pos = y_train[:, i] == 1  # index of positive class
                neg = y_train[:, j] == 1  # index of negative class

                X1 = np.vstack((X_train[pos], X_train[neg]))

                y1 = np.vstack((y_train[pos], y_train[neg]))
                y1 = np.array([y1[:, i]]).T

                if model == 'svm':
                    # Train the model
                    SV, alpha, t, b, c, d, sigma = svm(X1, y1,
                                                       C=C, t=kernel_type,
                                                       b=b, c=c, d=d, sigma=sigma)
                    # Predict
                    y = fx_svm(X2, X1, y1, alpha, SV, t, b, c, d, sigma)

                # TW-SVM
                elif model == 'twsvm':
                    # Train the model
                    Ct, z1, z2 = twsvm(X1, y1, c1=c1, c2=c2,
                                       kernel_type=kernel_type,
                                       optimizer=optimizer, b=b, c=c, d=d, sigma=sigma)
                    # Predict
                    y = fx_twsvm(X2, Ct, z1, z2, kernel_type=kernel_type, b=b, c=c, d=d, sigma=sigma)

                # LS-SVM
                elif model == 'lssvm':
                    # Train the model
                    alpha, bias = lssvm(X1, y1, C=C, kernel_type=kernel_type, b=b, c=c, d=d, sigma=sigma)
                    # Predict
                    y = fx_lssvm(X2, X1, y1, alpha, bias, kernel_type, b=b, c=c, d=d, sigma=sigma)

                # Keep predictions in a matrix
                if k == 0:
                    y_hat_temp = y
                else:
                    y_hat_temp = np.hstack((y_hat_temp, y))
                k = k + 1

        # Build y_hat matrix
        # Sum number of votes for each class
        # i is the postive clsas: sum the results
        # j is the negative class: sum the inverse of results
        # k is nc * (nc - 1) predictions
        y_hat = np.zeros((N, nc))
        for n in range(N):
            k = 0
            for i in range(nc):
                for j in range(i + 1, nc):
                    # Sum of positive class
                    y_hat[n, i] = y_hat[n, i] + y_hat_temp[n, k]
                    # Sum of negative class (invert the result)
                    y_hat[n, j] = y_hat[n, j] + int(np.logical_not(y_hat_temp[n, k]))
                    k = k + 1

        # Y_hat has the majority of votes
        y_hat = (y_hat == np.max(y_hat, axis=0)) * 1

    # ---------------
    # One vs All
    # ---------------
    elif method == 'ova':

        # Number of instances and classes
        N, nc = y_test.shape

        X1 = X_train
        X2 = X_test
        acc = np.ones((nc))
        for i in range(nc):
            # Create the train and test sets
            # X1, y1: train
            # X2, y2: test
            y1 = y_train[:, i]
            y1 = np.array([y1]).T

            # Traditional SVM
            if model == 'svm':
                SV, alpha, t, b, c, d, sigma = svm(X1, y1,
                                                   C=C, t=kernel_type,
                                                   b=b, c=c, d=d, sigma=sigma)

                y = fx_svm(X2, X1, y1, alpha,
                           SV, t, b, c, d, sigma)
            # TW-SVM
            elif model == 'twsvm':
                # Train the model
                Ct, z1, z2 = twsvm(X1, y1, c1=c1, c2=c2,
                                   kernel_type=kernel_type,
                                   optimizer=optimizer, b=b, c=c, d=d, sigma=sigma)
                # Predict
                y = fx_twsvm(X2, Ct, z1, z2, kernel_type=kernel_type, b=b, c=c, d=d, sigma=sigma)
            # LS-SVM
            elif model == 'lssvm':
                # Train the model
                alpha, bias = lssvm(X1, y1, C=C, kernel_type=kernel_type, b=b, c=c, d=d, sigma=sigma)
                # Predict
                y = fx_lssvm(X2, X1, y1, alpha, bias, kernel_type, b=b, c=c, d=d, sigma=sigma)

            # Save predictions in a matrix
            if i == 0:
                y_hat = y
            else:
                y_hat = np.hstack((y_hat, y))

    y_hat = np.where(y_hat == -1, 0, y_hat)
    return y_hat


# ==================================
# 6.5 Tune SVM, TWSVM, LSSVM
# ==================================
# Tune SVM, TWSVM and LSSVM with RBF kernel using grid search
# with cross validation.
# Ref: A Practical Guide to Support Vector Classification.
# C.-W.. Hsu, C.-C. Chang, C.-J. Lin (2016).
# https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
def tune_rbf(X_train, y_train, X_test, y_test, model='svm', K=5, plot=False, DS_name='', pdir = ''):
    # Input
    # X_train, y_train: trainning sets
    # X_test, y_test: test sets
    # model: svm, twsvm or lssvm
    # K: number of folds
    # Output:
    # y_hat: predicted value with best C and sigma

    # N: Number of observations
    # d: Number of attributes + 1
    N, d = X_train.shape
    nc = y_train.shape[1]  # Number of classes
    I = np.eye(d)  # Identity matrix
    E = np.array([])  # Error array
    c1 = 1
    c2 = 1
    C = 1
    b = 1  # kernel parameter
    c = 1  # kernel parameter
    d = 2  # kernel parameter
    C_max = None  # Keep max C  (SVM soft margin)
    c1_max = None  # Keep max c1 (TWSVM soft margin)
    c2_max = None  # Keep max c2 (TWSVM soft margin)
    s_max = None  # Keep max sigma (rbf kernel parameter)
    acc_max = 0  # Keep max accuracy

    C_vals = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_vals = [0.01, 0.1, 1, 10, 100]
    C_vals = [2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15]
    gamma_vals = [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1, 2 ** 3]
    # ----------------------------
    # 1.1 TWSVM - not in use
    # ----------------------------
    if model == 'twsvm-X':
        # plotData: data to create plot
        plotData = pd.DataFrame(columns=['C1', 'C2', 'Sigma', 'Accuracy'])
        # Grid search over several values of C and sigma
        for C in C_vals:  # assume c1 = c2 = C
            for gamma in gamma_vals:
                sigma = np.sqrt(0.5 / gamma)
                acc_vec = np.array([])  # Accuracy array
                # k-fold cross validation
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
                    # Binary classification
                    if nc == 1:
                        # Train the model
                        Ct, z1, z2 = twsvm(X1, y1, c1=c1, c2=c2,
                                           kernel_type='RBF',
                                           optimizer=2, b=b, c=c, d=d, sigma=sigma)
                        # Predict
                        y_hat = fx_twsvm(X2, Ct, z1, z2, kernel_type='RBF', b=b, c=c, d=d, sigma=sigma)

                    # Multi-class
                    else:
                        y_hat = svm_mc(X1, y1, X2, y2,
                                       model=model, method='ovo', c1=c1, c2=c2,
                                       kernel_type='RBF', optimizer=2,
                                       C=C, b=b, c=c, d=d, sigma=sigma)

                    # Compute the accuracy
                    acc = (100 * np.mean(y_hat == y2)).round(2)
                    # Update the accuracy vector
                    acc_vec = np.insert(arr=acc_vec, obj=acc_vec.shape, values=acc)
                # Cross-validation accuracy is the
                # mean value of all k-folds
                cv_acc = np.mean(acc_vec)
                plotData.loc[len(plotData) + 1] = [c1, c2, sigma, cv_acc]
                # Keep the best values after running
                # for each K folds,
                if cv_acc > acc_max:
                    acc_max = np.mean(acc_vec)
                    c1_max = c1
                    c2_max = c2
                    s_max = sigma
    # ----------------------------
    # 1.2 SVM and LSSVM, binary and multi-class
    # ----------------------------
    else:
        # plotData: data to create plot
        plotData = pd.DataFrame(columns=['C', 'Sigma', 'Accuracy'])
        # Grid search over several values of C and sigma
        for C in C_vals:
            for gamma in gamma_vals:
                sigma = np.sqrt(1 / (2 * gamma))
                acc_vec = np.array([])  # Accuracy array

                # k-fold cross validation
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
                    # Binary classification
                    if nc == 1:
                        # Traditional SVM
                        if model == 'svm':
                            # Train the model
                            SV, alpha, t, b, c, d, sigma = svm(X1, y1, C=C,
                                                               t='RBF', b=1,
                                                               c=1, d=2,
                                                               sigma=sigma)
                            # Predict
                            y_hat = fx_svm(X2, X1, y1, alpha, SV, t='RBF', b=b, c=c, d=d, sigma=sigma)

                        # LS-SVM
                        elif model == 'lssvm':
                            # Train the model
                            alpha, bias = lssvm(X1, y1, C=C, kernel_type='RBF', b=b, c=c, d=d, sigma=sigma)
                            # Predict
                            y_hat = fx_lssvm(X2, X1, y1, alpha, bias, kernel_type='RBF', b=b, c=c, d=d, sigma=sigma)
                        elif model == 'twsvm':
                            # Train the model
                            Ct, z1, z2 = twsvm(X1, y1, c1=C, c2=C,
                                               kernel_type='RBF',
                                               optimizer=2, b=b, c=c, d=d, sigma=sigma)
                            # Predict
                            y_hat = fx_twsvm(X2, Ct, z1, z2, kernel_type='RBF', b=b, c=c, d=d, sigma=sigma)

                    # Multi class classification
                    else:
                        y_hat = svm_mc(X1, y1, X2, y2,
                                       c1=C, c2=C,
                                       model=model, method='ovo',
                                       kernel_type='RBF', optimizer=2,
                                       C=C, b=b, c=c, d=d, sigma=sigma)

                    # Compute the accuracy
                    acc = (100 * np.mean(y_hat == y2)).round(2)
                    # Update the accuracy vector
                    acc_vec = np.insert(arr=acc_vec, obj=acc_vec.shape, values=acc)
                # Cross-validation accuracy is the
                # mean value of all k-folds
                cv_acc = np.mean(acc_vec)
                plotData.loc[len(plotData) + 1] = [C, sigma, cv_acc]

                # Keep the best values after running
                # for each K folds,
                if cv_acc > acc_max:
                    acc_max = cv_acc
                    C_max = C
                    s_max = sigma
    # ----------------------------
    # 2. Use the best parameter C and sigma to train
    #   the whole training set
    # ----------------------------
    if nc == 1:
        # Traditional SVM
        if model == 'svm':
            # Train the model
            SV, alpha, t, b, c, d, sigma = svm(X_train, y_train, C=C_max,
                                               t='RBF', b=1,
                                               c=1, d=2,
                                               sigma=s_max)
            # Predict
            y_hat = fx_svm(X_test, X_train, y_train, alpha, SV, t='RBF', b=b, c=c, d=d, sigma=s_max)
        # LS-SVM
        elif model == 'lssvm':
            # Train the model
            alpha, bias = lssvm(X_train, y_train, C=C_max, kernel_type='RBF', b=b, c=c, d=d, sigma=s_max)
            # Predict
            y_hat = fx_lssvm(X_test, X_train, y_train, alpha, bias, kernel_type='RBF', b=b, c=c, d=d, sigma=s_max)
        elif model == 'twsvm':
            # Train the model
            Ct, z1, z2 = twsvm(X_train, y_train, c1=C_max, c2=C_max,
                               kernel_type='RBF',
                               optimizer=2, b=b, c=c, d=d, sigma=s_max)
            # Predict
            y_hat = fx_twsvm(X_test, Ct, z1, z2, kernel_type='RBF', b=b, c=c, d=d, sigma=s_max)
    # Multi-class
    else:
        y_hat = svm_mc(X_train, y_train, X_test, y_test,
                       c1=C_max, c2=C_max,
                       model=model, method='ovo',
                       kernel_type='RBF', optimizer=2,
                       C=C_max, b=b, c=c, d=d, sigma=s_max)
    # ----------------------------
    # 3. Plot CV Accuracy, C and Sigma
    # ----------------------------
    if plot:
        if model in ['svm', 'lssvm']:
            plt.scatter('C', 'Sigma', c='Accuracy', data=plotData)
            plt.xlabel('Soft margin C')
            plt.ylabel('Sigma')
            plt.title(model.upper() + ' - ' + DS_name)
        elif model == 'twsvm':
            plt.scatter('C1', 'C2', s='Sigma', c='Accuracy', data=plotData)
            plt.xlabel('Soft margin C1')
            plt.ylabel('Soft margin C2')
            plt.title('TWSVM Tuning - ' + DS_name)
        plt.savefig(pdir + DS_name + '_' + model + '_tuning.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ----------------------------
    # 4. Save tested parameters to file
    # ----------------------------
    plotData.to_csv(pdir + DS_name + "_" + model + "_tuning.csv", index=False, decimal=',', sep='\t')

    return y_hat


# Train SVM with linear, polynomial and RBF kernels
# Tune RBF kernel using grid search and cross validation
def multi_svm(X_train, y_train, X_test, y_test, DS_name, results, tune=True):
    # Input
    # DS_name: dataset name, e.g. 'Diabetes'
    # results: dataframe with test results
    # ----------------------------
    # 1. Train SVM with linear, polynomial and rbf kernels
    # ----------------------------
    for kernel_type in ['Linear', 'Polynomial', 'RBF']:
        # Train
        SV, alpha, t, b, c, d, sigma = svm(X_train, y_train, C=1,
                                           t=kernel_type, b=1,
                                           c=1, d=2,
                                           sigma=5)
        # Predict
        y_hat = fx_svm(X_test, X_train, y_train,
                       alpha, SV, t=kernel_type,
                       b=b, c=c, d=d, sigma=5)
        # Accuracy
        acc = (100 * np.mean(y_hat == y_test)).round(2)
        results = results.append({'Dataset': DS_name,
                                  'Method': 'SVM',
                                  'kernel': kernel_type,
                                  'Accuracy': acc},
                                 ignore_index=True)
    # ----------------------------
    # 2. Tune RBF kernel using grid search and cross validation
    # ----------------------------
    if tune:
        y_hat = tune_rbf(X_train, y_train, X_test, y_test, model='svm', K=5, DS_name=DS_name)
        acc = (100 * np.mean(y_hat == y_test)).round(2)

        results = results.append({'Dataset': DS_name,
                                  'Method': 'SVM',
                                  'kernel': 'Tuned RBF',
                                  'Accuracy': acc},
                                 ignore_index=True)
    return results


def multi_twsvm(X_train, y_train, X_test, y_test, DS_name, results, tune=True):
    # DS_name: dataset name, e.g. 'Diabetes'
    # results: dataframe with test results
    # ----------------------------
    # 1. Train TWSVM with linear, polynomial and rbf kernels
    # ----------------------------
    for kernel_type in ['Linear', 'Polynomial', 'RBF']:
        # Train
        Ct, z1, z2 = twsvm(X_train, y_train, c1=1, c2=1,
                           kernel_type=kernel_type,
                           optimizer=2, b=1, c=1, d=2, sigma=5)
        # Predict
        y_hat = fx_twsvm(X_test, Ct, z1, z2, kernel_type=kernel_type, b=1, c=1, d=2, sigma=5)
        # Accuracy
        acc = (100 * np.mean(y_hat == y_test)).round(2)
        results = results.append({'Dataset': DS_name,
                                  'Method': 'TWSVM',
                                  'kernel': kernel_type,
                                  'Accuracy': acc},
                                 ignore_index=True)
    # ----------------------------
    # 2. Tune RBF kernel using grid search and cross validation
    # ----------------------------
    if tune:
        y_hat = tune_rbf(X_train, y_train, X_test, y_test, model='twsvm', K=5, DS_name=DS_name)
        acc = (100 * np.mean(y_hat == y_test)).round(2)
        results = results.append({'Dataset': DS_name,
                                  'Method': 'TWSVM',
                                  'kernel': 'Tuned RBF',
                                  'Accuracy': acc},
                                 ignore_index=True)
    return results


def multi_lssvm(X_train, y_train, X_test, y_test, DS_name, results, tune=True):
    # DS_name: dataset name, e.g. 'Diabetes'
    # results: dataframe with test results
    # ----------------------------
    # 1. Train LSSVM with linear, polynomial and rbf kernels
    # ----------------------------
    for kernel_type in ['Linear', 'Polynomial', 'RBF']:
        # Train the model
        alpha, bias = lssvm(X_train, y_train, C=1, kernel_type=kernel_type, b=1, c=1, d=2, sigma=5)
        # Predict
        y_hat = fx_lssvm(X_test, X_train, y_train, alpha, bias, kernel_type=kernel_type, b=1, c=1, d=2, sigma=5)
        # Accuracy
        acc = (100 * np.mean(y_hat == y_test)).round(2)
        results = results.append({'Dataset': DS_name,
                                  'Method': 'LSSVM',
                                  'kernel': kernel_type,
                                  'Accuracy': acc},
                                 ignore_index=True)
    # ----------------------------
    # 2. Tune RBF kernel using grid search and cross validation
    # ----------------------------
    if tune:
        y_hat = tune_rbf(X_train, y_train, X_test, y_test, model='lssvm', K=5, DS_name=DS_name)
        acc = (100 * np.mean(y_hat == y_test)).round(2)
        results = results.append({'Dataset': DS_name,
                                  'Method': 'LSSVM',
                                  'kernel': 'Tuned RBF',
                                  'Accuracy': acc},
                                 ignore_index=True)
    return results


# Multi-class SVM, LSSVM and TWSVM training
def multi_mc(X_train, y_train, X_test, y_test, results, tune=True):
    # DS_name: dataset name, e.g. 'Diabetes'
    # results: dataframe with test results
    # ----------------------------
    # 1. Train LSSVM with linear, polynomial and rbf kernels
    # ----------------------------
    for model in ['svm', 'twsvm', 'lssvm']:
        for kernel_type in ['Linear', 'Polynomial', 'RBF']:
            for method in ['ovo', 'ova']:
                # Train the model
                y_hat = svm_mc(X_train, y_train, X_test, y_test,
                               c1=1, c2=1, model=model,
                               method=method, kernel_type=kernel_type,
                               optimizer=2, C=1, b=1,
                               c=1, d=2, sigma=5)
                acc = (100 * np.mean(y_hat == y_test)).round(2)

                results = results.append({'Dataset': 'Iris',
                                          'class': 'overall',
                                          'Method': model.upper(),
                                          'kernel': kernel_type,
                                          'ovo-ova': method,
                                          'Accuracy': acc},
                                         ignore_index=True)

                # Accuracy for each class
                acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)

                results = results.append({'Dataset': 'Iris',
                                          'class': 'setosa',
                                          'Method': model.upper(),
                                          'kernel': kernel_type,
                                          'ovo-ova': method,
                                          'Accuracy': acc[0]},
                                         ignore_index=True)

                results = results.append({'Dataset': 'Iris',
                                          'class': 'versicolor',
                                          'Method': model.upper(),
                                          'kernel': kernel_type,
                                          'ovo-ova': method,
                                          'Accuracy': acc[1]},
                                         ignore_index=True)

                results = results.append({'Dataset': 'Iris',
                                          'class': 'virginica',
                                          'Method': model.upper(),
                                          'kernel': kernel_type,
                                          'ovo-ova': method,
                                          'Accuracy': acc[2]},
                                         ignore_index=True)

    # ----------------------------
    # 2. Tune RBF kernel using grid search and cross validation
    # ----------------------------
    if tune:
        for model in ['svm', 'twsvm', 'lssvm']:
            y_hat = tune_rbf(X_train, y_train, X_test, y_test, model=model, K=5, DS_name='Iris')
            acc = (100 * np.mean(y_hat == y_test)).round(2)
            results = results.append({'Dataset': 'Iris',
                                      'class': 'overall',
                                      'Method': model.upper(),
                                      'kernel': 'Tuned RBF',
                                      'Accuracy': acc},
                                     ignore_index=True)
            # Accuracy for each class
            acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)

            results = results.append({'Dataset': 'Iris',
                                      'class': 'setosa',
                                      'Method': model.upper(),
                                      'kernel': 'Tuned RBF',
                                      'Accuracy': acc[0]},
                                     ignore_index=True)

            results = results.append({'Dataset': 'Iris',
                                      'class': 'versicolor',
                                      'Method': model.upper(),
                                      'kernel': 'Tuned RBF',
                                      'Accuracy': acc[1]},
                                     ignore_index=True)

            results = results.append({'Dataset': 'Iris',
                                      'class': 'virginica',
                                      'Method': model.upper(),
                                      'kernel': 'Tuned RBF',
                                      'Accuracy': acc[2]},
                                     ignore_index=True)

    return results
