#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Support Vector Machines (SVM)

Normalize X using min-max or z-score before predicting
All methods converts Y ∈ [-1, 1] and returns Y ∈ [0, 1]
Use 1-of-N coding of Y for multiclass

AVailable models:
   - traditional SVM,
   - Twin SVM (TW-SVM),
   - Least Squares SVM (LS-SVM)
Binary and Multiclass:
   - One versus One
   - One versus All
Kernel types:
   - linear, polynomial, rbf, erbf, hyperbolic tangent (tanh), linear splines
QP solver options:
   - cvxopt, SciPy optimize
RBF kernel tuning:
   - soft margin and sigma

Commonly used parameters
	X_train, y_train: Train sets.
	X_test, y_test:   Test sets.
	model:            One of 'svm', 'lssvm', 'twsvm'.
	y_hat:            Predicted values.
	kernel:           One of 'linear', 'poly', 'rbf', 'erbf', 'tanh', 'lspline'
	K:                Number of cross validation folds
	tune:             (boolean) tune RBF kernel parameters soft margin and sigma.
	plot:             (boolean) create plots.
	pdir:             Plot directory.
	DS_name:          Dataset name used in plot title and filename.
	                  Do not include spaces.

Author: Victor Ivamoto
August, 2020
References within the code
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import cvxopt

cvxopt.solvers.options['show_progress'] = False

import sklearn.datasets as ds

class svm:
    
    def __init__(self, kernel = 'rbf'):
        self.model = 'svm'          # 'svm', 'lssvm', 'twsvm'
        self.method = 'ovo'         # Multi-class method. 'ovo': One vs One, 'ova': One vs All
        # Kernel settings
        self.kernel_type = 'rbf'   # 'linear', 'poly', 'rbf', 'erbf', 'tanh', 'lspline'
        self.b = 1                  # constant multiplier for hyperbolic tangent
        self.c = 1                  # constant sum for polynomial, tangent and linear splines
        self.d = 2                  # polynomial power
        self.sigma = 3              # RBF and ERBF sigma
        # Twin SVM settings
        self.c1 = 1                 # TW-SVM soft margin 1
        self.c2 = 1                 # TW-SVM soft margin 2
        # SVM settings
        self.C = 10                 # soft margin
        self.solver = 'optimize'   # Quadratic problem solver. 'cvxopt' or 'optimize'
        #
        self.alpha = None           # Lagrange multiplier (SVM and LSSVM)
        self.SV = None              # Support vectors
        self.bias = 1               # bias
        # Kernel types
        # linear : linear
        # poly   : polynomial
        # rbf    : radial base gaussian function (RFB)
        # erbf   : radial base exponencial
        # tanh   : hyperbolic tangent
        # lspline: linear splines
        return


    # ==================================
    # 0.1 Train any model (SVM, LSSVM, TWSVM)
    # ==================================
    def train(self, X_train, y_train, model):
        """
        Train SVM, LSSVM, TWSVM models
        
        Input
            X_train:        train set. Numeric normalized values or categorical values
                            encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
            model:          'svm', 'lssvm', 'twsvm'
        Output
           
            Object with parameters, weights and biases.
        """
        # Multiclass
        if y_train.shape[1] > 2:
            print('Only binary models, for multi-class use svm-mc')
            return
            
        self.model = model
        if model == 'svm':
            self.svm_fit(X_train, y_train)
        elif model == 'lssvm':
            self.lssvm_fit(X_train, y_train)
        elif model == 'twsvm':
            self.twsvm_fit(X_train, y_train)

        return


    # ==================================
    # 0.2 Predict any model (SVM, LSSVM, TWSVM)
    # ==================================
    def predict(self, X_test, X_train = None, y_train = None):
        """
        Predict new values.
        
        Input
            X_test:         test set. Numeric normalized values or categorical values
                            encoded as numeric.
            X_train:        train set. Numeric normalized values or categorical values
                            encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        
        Output
            Predicted values.
        """

        if self.model == 'svm':
            y_hat= self.svm_predict(X_test, X_train, y_train)
        elif self.model == 'lssvm':
            y_hat= self.lssvm_predict(X_test, X_train, y_train)
        elif self.model == 'twsvm':
            y_hat= self.twsvm_predict(X_test)

        return y_hat


    # ==================================
    # 0.3 Kernel
    # ==================================
    # Compute kernel matrix k(x1, x2)
    def kernel(self, x1, x2):
        # Input parameters
        #   t = kernel type
        #       linear:     linear
        #       poly:       polynomial
        #       rbf:        radial base gaussian function (RFB)
        #       erbf:       radial base exponencial
        #       tanh:       hyperbolic tangent
        #       lspline:    linear splines
        #   b:      constant multiplier for hyperbolic tangent
        #   c:      constant sum for polynomial, tangent and linear splines
        #   d:      polynomial and linear splines power indicator
        #   sigma:  free parameter for RBF and exponential base
        # One problem with the polynomial kernel is that it may suffer
        # from numerical instability:
        # when x1Tx2 + c < 1, k(x1, x2) = (x1Tx2 + c)^d tends to zero with increasing d,
        # whereas when x1Tx2 + c > 1, k(x1, x2) tends to infinity
        #
        if x1.ndim == 1:
            x1 = np.array([x1]).T

        if x2.ndim == 1:
            x2 = np.array([x2]).T

        m1 = x1.shape[0]
        m2 = x2.shape[0]
        k = np.zeros((m1, m2))

        t = self.kernel_type
        b = self.b
        c = self.c
        d = self.d
        sigma = self.sigma
        for i in range(m1):
            for j in range(m2):

                # Linear kernel
                if t == 'linear':
                    k[i, j] = x1[i] @ x2[j]
                # Polynomial kernel
                elif t == 'poly':
                    k[i, j] = (x1[i] @ x2[j] + c) ** d
                # Radial base gaussian function (RBF)
                elif t == 'rbf':
                    k[i, j] = np.exp(-(x1[i] - x2[j]) @ (x1[i] - x2[j]) / (2 * sigma ** 2))
                # Radial base exponential function
                elif t == 'erbf':
                    k[i, j] = np.exp(-np.abs(x1[i] - x2[j]) / (2 * sigma ** 2))
                # Hyperbolic tangent
                elif t == 'tanh':
                    k[i, j] = np.tanh(b * (x1[i] @ x2[j]) + c)
                # Linear splines
                elif t == 'lspline':
                    k[i, j] = c + x1[i] * x2[j] + x1[i] * x2[j] * min(x1[i], x2[j]) + 1 / 2 * (x1[i] + x2[j]) *  min(x1[i], x2[j]) ** d

        return k

    # ==================================
    # 1. Traditional SVM
    # ==================================
    # Reference:
    # Committee machines: a unified approach using support vector machines
    # Clodoaldo Aparecido de Moraes Lima
    # ==================================
    # 1.1 SVM: Predict
    # ==================================
    def svm_predict(self, X_test, X_train, y_train):
        """
        Predict SVM.
        
        Input values
            X_test:         test set. Numeric normalized values or categorical values
                            encoded as numeric.
            X_train:        train set. Numeric normalized values or categorical values
                            encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        Output
            y_hat:          predicted values
        """

        if y_train.ndim == 1:
            y_train = np.array([y_train]).T

        SV = self.SV
        y_train = np.where(y_train == 0, -1, y_train)
        y_train = y_train[SV[:, 0]]
        X_train = X_train[SV[:, 0]]
        alpha = self.alpha[SV[:, 0]]

        # Compute the bias
        k = self.kernel(X_train, X_test)
        SV_neg = y_train < 0
        SV_pos = y_train > 0
        self.bias = (-1 / 2) * (np.max(k[SV_neg[:, 0], :].T @ alpha[SV_neg]) + np.min(k[SV_pos[:, 0], :].T @ alpha[SV_pos]))
        self.bias = y_train - np.sum(alpha * y_train * k, axis=1, keepdims=True)
        self.bias = np.mean(self.bias)

        # Predict new values
        y_hat = np.sign(np.sum(alpha * y_train * k, axis=0, keepdims=True) + self.bias).T
        y_hat = np.where(y_hat == -1, 0, y_hat)

        return y_hat


    # ==================================
    # 1.2 SVM: Train
    # ==================================
    def svm_fit(self, X_train, y_train):
        """
        Train SVM model. Parameters defined in the svm object. Trained values 
        stored in the svm object.
        
        Input
            X_train:        train set. Numeric normalized values or categorical
                            values encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        SVM parameter
            C:              SVM soft margin
        Kernel parameters
            t:              kernel type: 'linear', 'poly', 'rbf', 'erbf','tanh',
                            'lspline'
            b:              constant multiplier for hyperbolic tangent
            c:              constant sum for polynomial, hyperbolic tangent and
                            linear splines
            d:              polynomial and linear splines power indicator
            sigma:          free parameter for RBF and exponential base

        Output
            Trained model stored in svm object.
        """

        y_train = np.where(y_train == 0, -1, y_train)
        N, m = X_train.shape
        C = self.C
        # Compute the Gram matrix
        k = self.kernel(X_train, X_train)
        # Construct P, q, A, b, G, h matrices for CVXOPT
        P = cvxopt.matrix(np.outer(y_train, y_train) * k, tc='d')
        q = cvxopt.matrix(np.ones(N) * -1, tc='d')
        A = cvxopt.matrix(y_train, (1, N), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        # hard-margin SVM
        if C is None or C == 0:
            G = cvxopt.matrix(np.diag(np.ones(N) * -1), tc='d')
            h = cvxopt.matrix(np.zeros(N), tc='d')
        # soft-margin SVM
        else:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(N) * -1), np.eye(N))), tc='d')
            h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)), tc='d')
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        self.alpha = np.asarray(solution['x'])
        # Support vectors have non zero lagrange multipliers
        self.SV = self.alpha > 1e-5  # some small threshold

        return


    # ==================================
    # 2 LS-SVM (Least Squares SVM)
    # ==================================
    # Reference:
    # Committee machines: a unified approach using support vector machines
    # Clodoaldo Aparecido de Moraes Lima
    # ==================================
    # 2.1 LSSVM: Predict
    # ==================================
    def lssvm_predict(self, X_test, X_train, y_train):
        """
        Predict LS-SVM values.
        
        Input
            X_test:         test set. Numeric normalized values or categorical values
                            encoded as numeric.
            X_train:        train set. Numeric normalized values or categorical values
                            encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        Output
            y_hat:          predicted values
        """
        
        y_train = np.where(y_train == 0, -1, y_train)

        y_hat = np.sign(np.sum(self.alpha * y_train * self.kernel(X_train, X_test), axis=0,
                               keepdims=True) + self.bias).T
        y_hat = np.where(y_hat == -1, 0, y_hat)

        return y_hat


    # ==================================
    # 2.2 LSSVM: Train
    # ==================================
    def lssvm_fit(self, X_train, y_train):
        """
        Train LS-SVM model. Parameters defined in the svm object. Trained values 
        stored in the svm object.
        
        Input
            X_train:        train set. Numeric normalized values or categorical
                            values encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        SVM parameter
            C:              SVM soft margin
        Kernel parameters
            t:              kernel type: 'linear', 'poly', 'rbf', 'erbf','tanh',
                            'lspline'
            b:              constant multiplier for hyperbolic tangent
            c:              constant sum for polynomial, hyperbolic tangent and
                            linear splines
            d:              polynomial and linear splines power indicator
            sigma:          free parameter for RBF and exponential base

        Output
            Trained model stored in svm object.
        """
        
        y_train = np.where(y_train == 0, -1, y_train)

        N = X_train.shape[0]
        nc = y_train.shape[1]
        K = self.kernel(X_train, X_train)

        # 3. Compute omega
        omega = np.zeros((N, N), int)
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):
                omega[k, l] = y_train[k] * y_train[l] * K[k, l]

        # 4. Build Matrix A and vector b
        I = np.eye(omega.shape[0])
        ZZCI = omega + self.C ** -1 * I

        # 4.1 Build matrix A
        A11 = np.zeros((1, 1))  # Element A(1,1)
        A1 = np.hstack((A11, -y_train.T))  # Row 1
        A2 = np.hstack((y_train, ZZCI))  # Row 2

        # Build matrix A
        A = np.vstack((A1, A2))

        # 4.2 Output vector b
        b = np.vstack((np.zeros((1, 1)), np.ones((N, 1))))

        # 5. Solve the linear equation Ax = b
        x = np.linalg.solve(A, b)

        self.bias = x[0]
        self.alpha = x[1:len(x)]

        return


    # ==================================
    # 3. TW-SVM (Twin SVM)
    # ==================================
    # Ref:
    # 1. Twin Support Vector Machines for Pattern Classification
    # Jayadeva - Reshma Khemchandani - Suresh Chandra
    # https://ieeexplore.ieee.org/document/4135685
    #
    # 2. Twin Support Vector Machines - Models, Extensions and Applications
    # Jayadeva - Reshma Khemchandani - Suresh Chandra
    # https://www.springer.com/gp/book/9783319461847
    # ==================================
    # 3.1 TWSVM: Predict
    # ==================================
    # Paper documentation
    def twsvm_predict(self, X_test):
        """
        Predict TW-SVM values.
        Input
            X_test:         test set. Numeric normalized values or categorical values
                            encoded as numeric.
        Output
            y_hat:          predicte values.
        """
        # Input parameters:
        # z1, z2
        # Ct

        # eq 3.33
        u1 = self.z1[:-1]   # surface 1 weights
        b1 = self.z1[-1:]   # surface 1 bias
        u2 = self.z2[:-1]   # surface 2 weights
        b2 = self.z2[-1:]   # surface 2 bias

        K_XCt = self.kernel(X_test, self.Ct)

        # Define the two surfaces (eq 3.21)
        surface1 = K_XCt @ u1 + b1
        surface2 = K_XCt @ u2 + b2

        # Calculate the distance from X to each surface
        dist1 = abs(surface1)  # class 1
        dist2 = abs(surface2)  # class -1

        # A new data point x ∈ Rn is assigned to class r (r = 1, 2),
        # depending on which of the two planes given by (3.19) it lies closer to.

        y_hat = np.argmax(np.hstack((dist1, dist2)), axis = 1)  # eq 3.20

        if y_hat.ndim == 1:
            y_hat = np.array([y_hat]).T

        return y_hat


    # ==================================
    # 3.2 TWSVM: Predict
    # ==================================
    # Prof. Clodoaldo explanation
    def twsvm_predict_clodoaldo(self, X_test):
        """
        Predict TW-SVM values
        Input
            X_test:         test set. Numeric normalized values or categorical values
                            encoded as numeric.
        Output
            y_hat:          predicted values.
        """

        u1 = self.z1[:-1]   #
        b1 = self.z1[-1:]   # bias
        u2 = self.z2[:-1]   #
        b2 = self.z2[-1:]   # bias

        K_XCt = self.kernel(X_test, self.Ct)

        # Define the two surfaces: equation 33
        surface1 = K_XCt @ u1 + b1
        surface2 = K_XCt @ u2 + b2

        # Calculate the distance from X to each surface
        dist1 = abs(surface1) / np.linalg.norm(u1)
        dist2 = abs(surface2) / np.linalg.norm(u2)

        # Initialize y_hat = 2
        y_hat = np.ones((X_test.shape[0], 1)) * 2
        pos = (surface1 >= 0) & (surface2 >= 0)  # class  1
        neg = (surface1 <= 0) & (surface2 <= 0)  # class -1
        mid = (surface1 <= 0) & (surface2 >= 0)  # X between both surfaces
        y_hat[pos] = 1  # class  1
        y_hat[neg] = 0  # class -1
        y_hat[mid] = np.where(dist1[mid] < dist2[mid], 1, 0)

        return y_hat

    # ==================================
    # 3.3 TWSVM: Solve quadratic problem
    # ==================================
    def solveQP(p, q, x0, C=None):
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
        p = cvxopt.matrix(p, tc='d')
        q = cvxopt.matrix(q, tc='d')
        if C is None or C == 0:  # hard-margin SVM
            g = cvxopt.matrix(np.diag(np.ones(N) * -1), tc='d')
            h = cvxopt.matrix(np.zeros((N, 1)), tc='d')
        else:  # soft-margin SVM
            g = cvxopt.matrix(np.vstack((np.diag(np.ones(N) * -1), np.eye(N))), tc='d')
            h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)), tc='d')
        # solve QP problem
        x = cvxopt.solvers.qp(p, q, g, h)

        return x


    # TWSVM: Solve quadratic problem using SciPy optimize
    # Optimize equations 45 and 47
    def minimize(self, alpha, M):

        g = -np.sum(alpha) + 0.5 * alpha.T @ M @ alpha
        return g


    # ==================================
    # 3.4 TWSVM: Train
    # ==================================
    # A, B: matrices of points belonging to classes 1 and -1
    # e1, e2: vectors of ones of appropriate dimensions
    # I: identity matrix of size (n+1) x (n+1)
    # epsilon x I: regularization term
    def twsvm_fit(self, X_train, y_train):
        """
        Train TWSVM model. Parameters defined in the svm object. Trained values 
        stored in the svm object.
        Input
            X_train:        train set. Numeric normalized values or categorical
                            values encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        TWSVM parameter
            c1, c2:         TWSVM soft margin
            solver:         'cvxopt' = CVXOPT, 'optimizer' = SciPy
        Kernel parameters
            t:              kernel type: 'linear', 'poly', 'rbf', 'erbf','tanh',
                            'lspline'
            b:              constant multiplier for hyperbolic tangent
            c:              constant sum for polynomial, hyperbolic tangent and
                            linear splines
            d:              polynomial and linear splines power indicator
            sigma:          free parameter for RBF and exponential base

        Output
            Trained model stored in svm object.
        """
        y_train = np.where(y_train == 0, -1, y_train)
        if y_train.ndim == 1:
            y_train = np.array([y_train]).T

        A = X_train[y_train[:, 0] == 1]
        B = X_train[y_train[:, 0] == -1]

        self.Ct = np.vstack((A, B))

        m1 = A.shape[0]
        m2 = B.shape[0]

        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        K_ACt = self.kernel(A, self.Ct)
        K_BCt = self.kernel(B, self.Ct)

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
        if self.solver == 'cvxopt':
            # CVXOPT algorithm:
            alpha = self.solveQP(P1, -e2, C=self.c1)  # eq 3.35
            gamma = self.solveQP(P2, -e1, C=self.c2)  # eq 3.36

            alpha = np.ravel(alpha['x'])
            gamma = np.ravel(gamma['x'])
        elif self.solver == 'optimize':
            # Scipy optimize
            b1 = optimize.Bounds(0, self.c1)
            b2 = optimize.Bounds(0, self.c2)
            alpha = optimize.minimize(self.minimize, x0=alpha0, args=P1, method='L-BFGS-B', bounds=b1).x
            gamma = optimize.minimize(self.minimize, x0=gamma0, args=P2, method='L-BFGS-B', bounds=b2).x

        if alpha.ndim == 1:
            alpha = np.array([alpha]).T

        if gamma.ndim == 1:
            gamma = np.array([gamma]).T

        # Equation 3.34
        epsilon = 1e-16
        I = np.eye(len(S.T @ S))
        self.z1 = -np.linalg.pinv(S.T @ S + epsilon * I) @ R.T @ alpha

        I = np.eye(len(J.T @ J))
        self.z2 = np.linalg.pinv(J.T @ J + epsilon * I) @ L.T @ gamma

        return


    # +++++++++++++++++++++++++++++++++
    # 4. Multiclass SVM, TWSVM, LSSVM
    # +++++++++++++++++++++++++++++++++
    # The best methods are one vs one and DDAG
    # Reference:
    # A Comparison of Methods for Multi-class Support Vector Machines
    # Chih-Wei Hsu and Chih-Jen Lin
    def svm_mc(self, X_train, y_train, X_test, y_test):
        """
        Input
            X_train:        train set. Numeric normalized values or categorical values
                            encoded as numeric.
            y_train:        train set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
            X_test:         test set. Numeric normalized values or categorical values
                            encoded as numeric.
            y_test:         test set, y={-1, 1} or y={0, 1}. Vector if binary,
                            one-hot enconded if multiclass.
        Input parameters
            model:          'svm', 'twsvm', 'lssvm'
            method:         'ono' = one vs one
                            'ova' = one vs all
            solver:         'cvxopt' = cvxopt library
                            'optimize' = SciPy optimize
        SVM parameters
            C:              SVM soft margin
        TWSVM parameters
            c1:             soft margin 1
            c2:             soft margin 2
        Kernel parameters
            kernel_type:    'linear', 'poly', 'rbf', 'erbf', 'tanh', 'lspline'
            b:              constant multiplier
            c:              constant adder
            d:              Polynomial power
            sigma:          RBF kernel sigma
        Output
            Trained model stored in svm object.
        """
        if type(y_train) is not np.ndarray:
            y_train = np.array(y_train)

        # ---------------
        # One vs One
        # ---------------
        if self.method == 'ovo':
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

                    # Train and predict
                    self.train(X1, y1, self.model)            # Train
                    y = self.predict(X2, X1, y1)  # Predict

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
            y_hat = (np.max(y_hat, axis = 1, keepdims = True) == y_hat) * 1

        # ---------------
        # One vs All
        # ---------------
        elif self.method == 'ova':

            # Number of instances and classes
            N, nc = y_test.shape

            X1 = X_train
            X2 = X_test
            acc = np.ones((nc))
            for i in range(nc):
                # Create the train and test sets
                # X1, y1: train
                # X2, y2: test
                y1 = y_train[:, i:i+1]

                # Train and predict
                self.train(X1, y1, self.model)  # Train
                y = self.predict(X2, X1, y1)  # Predict

                # Save predictions in a matrix
                if i == 0:
                    y_hat = y
                else:
                    y_hat = np.hstack((y_hat, y))

        y_hat = np.where(y_hat == -1, 0, y_hat)
        return y_hat


    # ==================================
    # 5 Tune RBF Kernel for SVM, TWSVM, LSSVM
    # ==================================
    # Tune RBF kernel using grid search with cross validation.
    # Find best values of soft margin and RBF sigma
    # Ref: A Practical Guide to Support Vector Classification.
    # C.-W.. Hsu, C.-C. Chang, C.-J. Lin (2016).
    # https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    def tune_rbf(self, X_train, y_train, X_test, y_test, K=5, plot=False, pdir = '', DS_name=''):
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
        nc = y_train.shape[1]   # Number of classes
        I = np.eye(d)           # Identity matrix
        E = np.array([])        # Error array
        C_max = None            # Keep max C  (SVM soft margin)
        c1_max = None           # Keep max c1 (TWSVM soft margin)
        c2_max = None           # Keep max c2 (TWSVM soft margin)
        s_max = None            # Keep max sigma (rbf kernel parameter)
        acc_max = 0             # Keep max accuracy

        # Values to be tuned
        C_vals = [2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15]
        gamma_vals = [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1, 2 ** 3]
        # ----------------------------
        # 1.1 TWSVM - not in use
        # ----------------------------
        if self.model == 'twsvm-X':
            # plotData: data to create plot
            plotData = pd.DataFrame(columns=['C1', 'C2', 'Sigma', 'Accuracy'])
            # Grid search over several values of C and sigma
            for c1 in C_vals:  # soft margin 1
                self.c1 = c1
                for c2 in C_vals:   # soft margin 2
                    self.c2 = c2
                    for gamma in gamma_vals:    # rbf kernel parmeter
                        self.sigma = np.sqrt(0.5 / gamma)
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
                                self.twsvm_fit(X1, y1)
                                # Predict
                                y_hat = self.twsvm_predict(X2)

                            # Multi-class
                            else:
                                y_hat = self.svm_mc(X1, y1, X2, y2)

                            # Compute the accuracy
                            acc = (100 * np.mean(y_hat == y2)).round(2)
                            # Update the accuracy vector
                            acc_vec = np.insert(arr=acc_vec, obj=acc_vec.shape, values=acc)
                        # Cross-validation accuracy is the
                        # mean value of all k-folds
                        cv_acc = np.mean(acc_vec)
                        plotData.loc[len(plotData) + 1] = [self.c1, self.c2, self.sigma, cv_acc]
                        # Keep the best values after running
                        # for each K folds,
                        if cv_acc > acc_max:
                            acc_max = np.mean(acc_vec)
                            c1_max = self.c1
                            c2_max = self.c2
                            s_max = self.sigma
        # ----------------------------
        # 1.2 SVM and LSSVM, binary and multi-class
        # ----------------------------
        else:
            # plotData: data to create plot
            plotData = pd.DataFrame(columns=['C', 'Sigma', 'Accuracy'])
            # Grid search over several values of C and sigma
            for C in C_vals:
                self.C = C
                self.c1 = C
                self.c2 = C
                for gamma in gamma_vals:
                    self.sigma = np.sqrt(1 / (2 * gamma))
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
                            if self.model == 'svm':
                                self.svm_fit(X1, y1)              # Train the model
                                y_hat = self.svm_predict(X2, X1, y1)   # Predict

                            # LS-SVM
                            elif self.model == 'lssvm':
                                self.lssvm_fit(X1, y1)            # Train
                                y_hat = self.lssvm_predict(X2, X1, y1) # Predict

                            # TW-SVM
                            elif self.model == 'twsvm':
                                self.twsvm_fit(X1, y1)          # Train
                                y_hat = self.twsvm_predict(X2) # Predict

                        # Multi class classification
                        else:
                            y_hat = self.svm_mc(X1, y1, X2, y2)

                        # Compute the accuracy
                        acc = (100 * np.mean(y_hat == y2)).round(2)
                        # Update the accuracy vector
                        acc_vec = np.insert(arr=acc_vec, obj=acc_vec.shape, values=acc)
                    # Cross-validation accuracy is the
                    # mean value of all k-folds
                    cv_acc = np.mean(acc_vec)
                    plotData.loc[len(plotData) + 1] = [C, self.sigma, cv_acc]

                    # Keep the best values after running
                    # for each K folds
                    if cv_acc > acc_max:
                        acc_max = cv_acc
                        C_max = C
                        s_max = self.sigma
        # ----------------------------
        # 2. Use the best parameter C and sigma to train
        #   the whole training set
        # ----------------------------
        self.C = C_max
        self.c1 = C_max
        self.c2 = C_max
        self.sigma = s_max
        if nc == 1:
            # Traditional SVM
            if self.model == 'svm':
                self.svm_fit(X_train, y_train)                         # Train
                y_hat = self.svm_predict(X_test, X_train, y_train)     # Predict
            # LS-SVM
            elif self.model == 'lssvm':
                self.lssvm_fit(X_train, y_train)                       # Train
                y_hat = self.lssvm_predict(X_test, X_train, y_train)   # Predict
            elif self.model == 'twsvm':
                self.twsvm_fit(X_train, y_train)        # Train
                y_hat = self.twsvm_predict(X_test)      # Predict
        # Multi-class
        else:
            y_hat = self.svm_mc(X_train, y_train, X_test, y_test)
        # ----------------------------
        # 3. Plot CV Accuracy, C and Sigma
        # ----------------------------
        # Reference:
        # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_with_legend.html
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        if plot:
            fig, ax = plt.subplots()
            scatter = ax.scatter('C', 'Sigma', c = 'Accuracy', data=plotData)
            plt.title(self.model.upper() + ' - ' + DS_name.title())
            plt.xlabel('Soft margin C')
            plt.ylabel('Sigma')
            plt.yscale('log')
            plt.xscale('log')
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(*scatter.legend_elements(), bbox_to_anchor = (1.04, 1),
                                 loc = 'upper left', title = "Accuracy")

            # Save plot and results
            if pdir != '':
                plt.savefig(pdir + DS_name + '_' + self.model + '_tuning.png',
                            dpi=300, bbox_inches='tight')
                plotData.to_csv(pdir + DS_name + "_" + self.model + "_tuning.csv", index=False, decimal=',', sep='\t')
            plt.show()

        return y_hat


    # ==================================
    # 6. Train all models
    # ==================================
    # Train SVM, LSSVM, TWSVM with linear, polynomial and RBF kernels
    # Tune RBF kernel using grid search and cross validation
    # ==================================
    # 6.1 Binary classification
    # ==================================
    def multi_svm(self, X_train, y_train, X_test, y_test, DS_name, results, tune=True, plot = False, pdir = ''):
        # Input
        # DS_name: dataset name, e.g. 'Diabetes'
        # results: dataframe with test results
        # ----------------------------
        # 1. Train SVM with linear, polynomial and rbf kernels
        # ----------------------------
        for model in ['svm', 'lssvm', 'twsvm']:
            self.model = model
            for kernel_type in ['linear', 'poly', 'rbf']:
                self.kernel_type = kernel_type
                # Train
                self.train(X_train, y_train, model)
                # Predict
                y_hat = self.predict(X_test, X_train, y_train)
                # Accuracy
                acc = (100 * np.mean(y_hat == y_test)).round(2)
                results = results.append({'Dataset': DS_name.title(),
                                          'Method': model.upper(),
                                          'kernel': kernel_type,
                                          'Accuracy': acc},
                                          ignore_index=True)
            # ----------------------------
            # 2. Tune RBF kernel using grid search and cross validation
            # ----------------------------
            if tune:
                y_hat = self.tune_rbf(X_train, y_train, X_test, y_test, K=5,
                                 plot = plot, pdir = pdir, DS_name=DS_name)
                acc = (100 * np.mean(y_hat == y_test)).round(2)

                results = results.append({'Dataset': DS_name.title(),
                                          'Method': model.upper(),
                                          'kernel': 'Tuned rbf',
                                          'Accuracy': acc},
                                         ignore_index=True)
        return results


    # ==================================
    # 6.2 Multiclass
    # ==================================
    # Multi-class SVM, LSSVM and TWSVM training
    def multi_mc(self, X_train, y_train, X_test, y_test, results, tune=True,
                 DS_name = 'Iris', plot=True, pdir=''):
        # DS_name: dataset name, e.g. 'Diabetes'
        # results: dataframe with test results
        # ----------------------------
        # 1. Train LSSVM with linear, polynomial and rbf kernels
        # ----------------------------
        for model in ['svm', 'twsvm', 'lssvm']:
            self.model = model
            for kernel_type in ['linear', 'poly', 'rbf']:
                self.kernel_type = kernel_type
                for method in ['ovo', 'ova']:
                    self.method = method
                    # Train the model
                    y_hat = self.svm_mc(X_train, y_train, X_test, y_test)
                    acc = (100 * np.mean(y_hat == y_test)).round(2)

                    results = results.append({'Dataset': DS_name,
                                              'class': 'overall',
                                              'Method': model.upper(),
                                              'kernel': kernel_type,
                                              'ovo-ova': method,
                                              'Accuracy': acc},
                                              ignore_index=True)

                    # Accuracy for each class
                    acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)

                    for k in range(2):
                        results = results.append({'Dataset': DS_name,
                                                  'class': 'Class ' + str(k),
                                                  'Method': model.upper(),
                                                  'kernel': kernel_type,
                                                  'ovo-ova': method,
                                                  'Accuracy': acc[k]},
                                                  ignore_index=True)

        # ----------------------------
        # 2. Tune RBF kernel using grid search and cross validation
        # ----------------------------
        if tune:
            for model in ['svm', 'twsvm', 'lssvm']:
                self.model = model
                y_hat = self.tune_rbf(X_train, y_train, X_test, y_test, K=5,
                                 plot=plot, pdir=pdir, DS_name=DS_name)
                acc = (100 * np.mean(y_hat == y_test)).round(2)
                results = results.append({'Dataset': DS_name,
                                          'class': 'overall',
                                          'Method': model.upper(),
                                          'kernel': 'Tuned rbf',
                                          'Accuracy': acc},
                                          ignore_index=True)
                # Accuracy for each class
                acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)

                for k in range(2):
                    results = results.append({'Dataset': DS_name,
                                              'class': 'Class ' + str(k),
                                              'Method': model.upper(),
                                              'kernel': 'Tuned rbf',
                                              'Accuracy': acc[k]},
                                              ignore_index=True)

        return results


#===============================
# Demo
#===============================
if __name__ == '__main__':
    # ----------------------------
    # Min-max normalization
    # ----------------------------
    def normalize(X, xmin = 0, xmax = 0):
        if xmin == 0 or xmax == 0:
            xmin = np.min(X)
            xmax = np.max(X)
        return (X - xmin) / (xmax - xmin), xmin, xmax

    # ----------------------------
    # Prepare dataset
    # ----------------------------
    # Import dataset
    data = ds.load_breast_cancer()  # binary classification
    data = ds.load_iris()           # multiclass
    data = ds.load_wine()           # multiclass

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

    # ----------------------------
    # Settings
    # ----------------------------
    fit = svm()
    fit.model = 'svm'           # 'svm', 'lssvm', 'twsvm'
    fit.method = 'ovo'          # Multi-class method. 'ovo': One vs One, 'ova': One vs All
    # Kernel settings
    fit.kernel_type = 'rbf'     # 'linear', 'poly', 'rbf', 'erbf', 'tanh', 'lspline'
    fit.b = 1                   # constant multiplier for hyperbolic tangent
    fit.c = 1                   # constant sum for polynomial, tangent and linear splines
    fit.d = 2                   # polynomial power
    fit.sigma = 3               # RBF and ERBF sigma
    # Twin SVM settings
    fit.c1 = 1                  # TW-SVM soft margin 1
    fit.c2 = 1                  # TW-SVM soft margin 2
    # SVM settings
    fit.C = 10                  # soft margin
    fit.solver = 'optimize'     # Quadratic problem solver. 'cvxopt' or 'optimize'
    # Predicted values
    fit.alpha = None            # Lagrange multiplier (SVM and LSSVM)
    fit.SV = None               # Support vectors
    fit.bias = 1                # bias

    # ----------------------------
    # 1. Train and predict
    # ----------------------------
    model = 'svm'       # 'svm', 'lssvm', 'twsvm'
    if binary:
        fit.train(X_train, y_train, model = model)
        y_hat = fit.predict(X_test, X_train, y_train)
    else:
        fit.model = model
        y_hat = fit.svm_mc(X_train, y_train, X_test, y_test)
    acc = np.mean(y_hat == y_test)
    print('Accuracy:', acc)
    
    # ----------------------------
    # 2. Tune RBF kernel: grid search and cross validation
    # ----------------------------
    fit.model = 'svm'   # possible values: 'svm', 'lssvm', 'twsvm'
    y_hat = fit.tune_rbf(X_train, y_train, X_test, y_test, K = 5, plot = True,
                         pdir = '', DS_name = 'Test dataset')
    acc = np.mean(y_hat == y_test)
    print('Tuned RBF kernel Accuracy:', acc)

    # ----------------------------
    # 3. Multiple tests - train all models
    # ----------------------------
    # Train SVM, LSSVM, TWSVM with linear, polynomial and RBF kernels.
    # Tune RBF kernel using grid search and cross validation.
    # This process takes some minutes to complete.
    results = pd.DataFrame()
    if binary:
        results = fit.multi_svm(X_train, y_train, X_test, y_test, DS_name = 'Test dataset',
                      results = results, tune = True, plot = True, pdir = '')
    else:
        results = fit.multi_mc(X_train, y_train, X_test, y_test, DS_name = 'Test dataset',
                        results = results, tune = True, plot = True, pdir = '')

    print(results)
    
    print(1)
