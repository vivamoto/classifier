# -*- coding: utf-8 -*-
"""
Consider this problem:

    Min f(x1,x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2

    Min f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2

Solve this problem using unconstrained optimization algorithms.

The methods gradient descent, Newton, Newton modified,
    Levenberg-Marquardt and One Step Secant are deployed in a single function
    for each method.

The methods Davidon-Fletcher-Powell and Broyden-Fletcher-Goldfarb-Shanno
    are deployed in function "quasi_newton".

The methods Hestenes-Stiefel, Polak-Ribiere and Fletcher-Reeves are
    deployed in function "conjugate_gradient"

Examples of usage are available at the end of the code, with random values
    for x1 and x2. There're also examples from the book "Nonlinear Programming".

#============================================================
Created on Wed Apr 22 16:07:16 2020

@author: Victor Ivamoto

Reference:
"Nonlinear Programming - Theory and Algorithms"
M. S. Bazaraa, H. D. Sherali, C. M. Shetty.-3rd edition

"Aula 03 - Revisao sobre metodos de otimizacao"
Clodoaldo A. M. Lima
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


#==================================
# Plot function
#==================================
# Create a plot of the function f(x1, x2)
# with contour lines.
def plot_function():
    """
    Create a plot of the function f(x1, x2) with contour lines.
    """
    # Compute contour lines
    x_1 = np.arange(-10, 10, 0.5)
    x_2 = np.arange(-10, 10, 0.5)
    x_1, x_2 = np.meshgrid(x_1, x_2)

    for equation in [1, 3]:
        fig, ax = plt.subplots()

        # f(x) = x_1^2 + 2.x_2^2 - 2.x_1.x_2 - 2.x_2"
        if equation == 1:
            title = "F(x1, x2) Contour Map"
            f = x_1**2 + 2*x_2**2 - 2*x_1*x_2 - 2*x_2
            locator = ticker.AutoLocator()

        # Example 8.8.7 from "Nonlinear Programming"
        # f(x) = (x_1 - 2)^4 + (x_1 - 2.x_2)^2"
        else:
            title = "Example 8.8.7 Contour Map"
            f = (x_1 - 2)**4 + (x_1 - 2*x_2)**2
            locator = ticker.SymmetricalLogLocator(linthresh = 0.1, base = 2)

        # Plot contour lines
        CS1 = ax.contourf(x_1, x_2, f, locator = locator, cmap=plt.get_cmap("plasma"))
        CS2 = ax.contour(x_1, x_2, f, locator = locator, colors = 'white', linewidths=(0.5,))

        # Color bar
        fig.colorbar(CS1, shrink=0.9)

        # Lables and title
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        plt.show()

        return
    
    
#==================================
# Plot results
#==================================
# Plot x1 vs x2 with contour lines
#
def plot(df, title, equation = 1):
    """
    Plot x1 vs x2 with contour lines.
    
    Input
        df:         dataframe with values to be plotted
        title:      chart title
        equation:       1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
                        2: f(x1, x2) = r1(x)**2 + r2(x)**2 (method Levenberg-Marquardt)
                        3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2

    """

    fig, ax = plt.subplots()

    # Compute contour lines
    delta_x1 = (1.5*max(df.x1) - min(df.x1)) / 1000
    delta_x2 = (1.5*max(df.x2) - min(df.x2)) / 1000

    d1 = (max(df.x1) - min(df.x1)) * 0.1
    d2 = (max(df.x2) - min(df.x2)) * 0.1
    x_1 = np.arange(min(df.x1) - d1, max(df.x1) + d1, delta_x1)
    x_2 = np.arange(min(df.x2) - d2, max(df.x2) + d2, delta_x2)
    x_1, x_2 = np.meshgrid(x_1, x_2)

    if equation == 1:
        f = x_1**2 + 2*x_2**2 - 2*x_1*x_2 - 2*x_2
        CS = ax.contour(x_1, x_2, f, locator = ticker.SymmetricalLogLocator(linthresh=1, base=10))
        locator = ticker.AutoLocator()

    elif equation == 3:
        f = (x_1 - 2)**4 + (x_1 - 2*x_2)**2
        locator = ticker.SymmetricalLogLocator(linthresh=0.1, base=2)

    else:
        print ("Only equations 1 and 3 are supported.")
        #return


    # Plot contour lines and labels
    CS = ax.contour(x_1, x_2, f, locator = locator)
    ax.clabel(CS, inline=1, fontsize=10)

    # Points
    x1 = df.x1
    x2 = df.x2

    ax.plot(x1, x2)
    ax.plot(x1, x2, 'ro')

    # Lables and title
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    plt.show()

    return


#==================================
# Function, Gradient and Hessian
#==================================
# Return the function, gradient and hessian.
# Equation 3 is example 8.8.7 from "Nonlinear Programming"
#
# 1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
# 2: f(x1, x2) = r1**2 + r2**2 (method Levenberg-Marquardt)
# 3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
def gradient(x, equation = 1):
    """
    Compute the gradient and hessian of each equation.
    
    Input:
        x:              vector with initial values x1 and x2
        equation:       1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
                        2: f(x1, x2) = r1(x)**2 + r2(x)**2 (method Levenberg-Marquardt)
                        3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
    Output
        f, g, h:        function, gradient and hessian
    """

    x1 = x[0][0]
    x2 = x[1][0]

    if equation == 1:
        # Function f(x1, x2)
        f = x1**2 + 2*x2**2 - 2*x1*x2 - 2*x2

        # Gradient  (first derivative)
        dFdx1 = 2*x1 - 2*x2
        dFdx2 = 4*x2 - 2*x1 -2
        g = np.array([[dFdx1, dFdx2]]).T

        # Hessian (second derivative)
        d2Fdx1dx1 =  2
        d2Fdx1dx2 = -2
        d2Fdx2dx1 = -2
        d2Fdx2dx2 =  4
        h = np.array([[d2Fdx1dx1, d2Fdx1dx2], [d2Fdx2dx1, d2Fdx2dx2]])

    elif equation == 2:
        # Functions used in Levenberg-Marquardt method
        # f(x) = sum[r(x)^2]
        r1 = (x1 - x2)
        r2 = (x2 -  1)
        r = np.array([[r1], [r2]])
        f = r1**2 + r2**2
        f = r

        # Gradient
        dr1x1 = 1
        dr1x2 = -1
        dr2x1 = 0
        dr2x2 = 1
        g = np.array([[dr1x1, dr1x2], [dr2x1, dr2x2]])

        # Hessian is not used, return 0
        h = 0

    elif equation == 3:
        # Example 8.8.7 from the book "Nonlinear programming"
        # Function f(x1, x2)
        f = (x1 - 2)**4 + (x1 - 2*x2)**2

        # Gradient  (first derivative)
        dFdx1 = 4 * (x1 - 2) ** 3 + 2 * (x1 - 2*x2)
        dFdx2 = -4 * (x1 - 2*x2)
        g = np.array([[dFdx1], [dFdx2]])

        # Hessian (second derivative)
        d2Fdx1dx1 =  12 * (x1 -2) ** 2 + 1
        d2Fdx1dx2 = -4
        d2Fdx2dx1 = -4
        d2Fdx2dx2 =  8
        h = np.array([[d2Fdx1dx1, d2Fdx1dx2], [d2Fdx2dx1, d2Fdx2dx2]])

    # Return the function, gradient and hessian
    return f, g, h


#==================================
# 1. Gradient descent (slide 28)
#==================================
def gradient_descent(x, plot_result = False, verbose = False, save = False):
    """
    Minimze equation using gradient descent method.
    
    Input:
        x:              vector. Initial value for x
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
    """

    epsilon = 1e-3
    alpha = 0.1

    f, g, h = gradient(x)

    # Step 0
    it = 0
    itmax = 100
    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(g)})

    while np.linalg.norm(g) > epsilon and it < itmax:

        # Step 1
        f, g, h = gradient(x)

        # step 2
        alpha = calc_alpha(x, -g)

        # Step 3
        x = x - alpha * g

        # Print result
        if verbose:
            print(it, " Norm: ", np.linalg.norm(g))

        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f,
                        'Gradient': np.linalg.norm(g)},
                       ignore_index=True)

        it = it + 1

    # Plot
    if plot_result:
        plot(df, 'Gradient Descent')

    # Save result
    if save:
        df.to_csv("gradient_descent.csv")

    return x

#==================================
# Calculate alpha
# This function is called by the variable gradient function
# defined below
#==================================
def calc_alpha(x, d, equation = 1):
    """
    Bisection method used to optimize the learning rate (alpha).
    
    Input
        d = direction
        equation:       1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
                        2: f(x1, x2) = r1(x)**2 + r2(x)**2 (method Levenberg-Marquardt)
                        3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
    """

    epsilon = 1e-3
    hlmin = 1e-3
    alpha_l = 0     # Lower alpha
    alpha_u = np.random.rand() # Upper alpha

    # New x position
    xn = x + alpha_u * d

    # Calculate the gradient of new position
    # (f and h aren't used)
    f, g, h = gradient(xn, equation)
    hl = g.T @ d

    while hl.item() < 0 :
        #
        alpha_u = 2 * alpha_u

        # Calculate the new position
        xn = x + alpha_u * d

        # Calculate the gradient of new position
        # f and h aren't used
        f, g, h = gradient(xn, equation)

        hl = g.T @ d

    # Alpha medium is the average of alphas
    alpha_m = (alpha_l + alpha_u) / 2

    # Estimate the maximum number of iterations
    itmax = np.ceil(np.log ((alpha_u - alpha_l) / epsilon))

    # Iteration counter
    it = 0

    while abs(hl.item()) > hlmin and it < itmax :

       # Calculate new position
        xn = x + alpha_m * d

        # Calculate the gradient of the new position
        # Note: f and h aren't used
        f, g, h = gradient(xn, equation)

        hl = g.T @ d
        if hl.item() > 0 :
            # Decrease upper alpha
            alpha_u = alpha_m
        elif hl.item() < 0 :
            # Increase lower alpha
            alpha_l = alpha_m
        else:
            break

        # Alpha medium is the alpha average
        alpha_m = (alpha_l + alpha_u) / 2

        # Increase number of iterations
        it = it + 1

    return alpha_m


#==================================
# 2. Method Newton (slide 41)
#==================================
def newton(x, plot_result = False, verbose = False, save = False):
    """
    Minimize equation using Newton method.
    
    Input:
        x:              vector. Initial value for x
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
    
    """
    epsilon = 1e-3

    # Step 0
    k = 0

    f, g, h = gradient(x)
    d = -np.linalg.inv(h) @ g

    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(d)})

    itmax = 100
    while np.linalg.norm(g) > epsilon and k < itmax:

        if verbose:
            print(k, " Norm: ", np.linalg.norm(d))

        # Step 1
        f, g, h = gradient(x)
        d = -np.linalg.inv(h) @ g

        # Step 2
        alpha = 1

        # Step 3
        x = x + alpha * d
        k = k + 1

        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f,
                        'Gradient': np.linalg.norm(d)},
                        ignore_index=True)

    # Plot
    if plot_result:
        plot(df, 'Newton Method')

    # Save result
    if save:
        df.to_csv("newton.csv")

    return x


#==================================
# 3. Newton Modified (slide 46)
#==================================
def newton_modified(x, plot_result = False, verbose = False, save = False):
    """
    Minimize equation using Newton Modified method.
    
    Input:
        x:              vector. Initial value for x
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
    """
    epsilon = 1e-3

    f, g, h = gradient(x)
    I = np.eye(len(x))

    # Check singular matrix
    if np.linalg.det(h) == 0:
        print("Singular matrix")
        exit

    # Eigenvalue of hessian (slide 46)
    l, v = np.linalg.eig(h)

    # Minimum eigenvalue (slide 46)
    lmin = np.min(l)

    # (slide 46)
    if lmin > 0:
        m = h
    else:
        e = lmin * 1.1
        m = h + (e - lmin) * I

    d = - np.linalg.inv(m) @ g

    # Step 0
    k = 0
    itmax = 100

    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(d)})

    while np.linalg.norm(g) > epsilon and k < itmax:

        if verbose:
            print(k, " Norm: ", np.linalg.norm(d))

        f, g, h = gradient(x)

        # Steps used to compute matrix M:
        # Eigenvalue of hessian (slide 45-46)
        l, v = np.linalg.eig(h)

        # Minimum eigenvalue (slide 45)
        lmin = np.min(l)

        # (slide 46)
        if lmin > 0:
            m = h
        else:
            e = lmin * 1.1
            m = h + (e - lmin) * I

        d = - np.linalg.inv(m) @ g

        # Step 2
        alpha = 1

        # Step 3
        x = x + alpha * d
        k = k + 1

        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f,
                        'Gradient': np.linalg.norm(d)},
                        ignore_index=True)

    # Plot
    if plot_result:
        plot(df, 'Newton Modified Method')

    # Save result
    if save:
        df.to_csv("newton_modified.csv")

    return x

#==================================
# 4. Levenberg-Marquardt (slide 52)
#==================================
def lm(x, plot_result = False, verbose = False, save = False):
    """
    Minimize equation using Levenberg-Marquardt method.
    
    Input:
        x:              vector. Initial value for x
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
    """

    # Step 0
    epsilon = 1e-3
    N = x.shape[0]
    I = np.eye(N)
    mu = 1e-6

    # Step 4
    r, g, h = gradient(x, equation = 2)

    # Step 5
    delta = - np.linalg.inv(g.T @ g + mu * I) @ g.T @ r

    f, g, h = gradient(x, equation = 1)
    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(delta)})
    # Step 3
    it = 0
    itmax = 100
    while it < itmax and np.linalg.norm(g) > epsilon:

        # Step 4
        r, g, h = gradient(x, equation = 2)

        # Step 5
        delta = -np.linalg.inv(g.T @ g + mu * I) @ g.T @ r

        # Step 6
        alpha = calc_alpha(x, delta, equation = 1)

        # Step 7
        x = x + alpha * delta

        it = it + 1

        if verbose:
            print(it, ", alpha: ", alpha, "mu: ", mu, "delta: ", np.linalg.norm(delta))

        f, g, h = gradient(x, equation = 1)
        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f,
                        'Gradient': np.linalg.norm(delta)},
                        ignore_index=True)

    # Plot
    if plot_result:
        plot(df, 'Levenberg-Marquardt Method')

    # Save result
    if save:
        df.to_csv("lm.csv")

    return x


#==================================
# 5. Quasi-Newton methods
# Davidon-Fletcher-Powell - DFP
# Broyden-Fletcher-Goldfarb-Shanno - BFGS
# slide 54-57
# "Nonlinear Programming", pg 408
#==================================
def quasi_newton(x, method = 'dfp', equation = 1, plot_result = False,
                 verbose = False, save = False):
    """
    Input:
        x:              vector. Initial value for x
        method:         'dfp' =  Davidon-Fletcher-Powell method
                        'bfgs' = Broyden-Fletcher-Goldfarb-Shanno method
        equation:       1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
                        2: f(x1, x2) = r1(x)**2 + r2(x)**2 (method Levenberg-Marquardt)
                        3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
    
    """


    # Initialization Step
    epsilon = 1e-3
    n = x.shape[0]
    M = np.eye(n)
    y = x
    k = 1
    j = 1


    # Main Step
    f, g, h = gradient(x, equation = equation)
    d = g
    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(d)})

    it = 0
    itmax = 100
    while np.linalg.norm(g) > epsilon and it < itmax:

        # Step 1
        f, g, h = gradient(y, equation = equation)
        d = -M @ g
        alpha = calc_alpha(y, d, equation)
        y = y + alpha * d

        if j < n:
            # Step 2
            f1, g1, h1 = gradient(y, equation = equation)

            # Equation 8.31
            p = alpha * d

            # Equation 8.32
            q = g1 - g

            # Davidon-Fletcher-Powell (DFP)
            if method.lower() == 'dfp':
                # slide 54 and equation 8.30
                M = M + (p @ p.T) / (p.T @ q) - (M @ q @ q.T @ M)/(q.T @ M @ q)

            # Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            elif method.lower() == 'bfgs':
                # slide 56 and equation 8.48
                M = M + (p @ p.T) / (p.T @ q) * (1 + ((q.T @ M @ q)/(p.T @ q))[0][0]) - (M @ q @ p.T + p @ q.T @ M)/(p.T @ q)

            g = g1
            j = j + 1

        elif j == n:
            x = y
            k = k + 1
            j = 1
            M = np.eye(n)

        if verbose:
            print(it, "|d|: ", np.linalg.norm(d))

        it = it + 1

        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f,
                        'Gradient': np.linalg.norm(d)},
                        ignore_index=True)

    # Plot
    if method.lower() == 'dfp':
        title = "Davidon-Fletcher-Powell - DFP"
    elif method.lower() == 'bfgs':
        title = "Broyden-Fletcher-Goldfarb-Shanno - BFGS"

    if plot_result:
        plot(df, title, equation)

    # Save result
    if save and method.lower() == 'dfp':
        df.to_csv("dfp.csv")
    elif save and method.lower() == 'bfgs':
        df.to_csv("bfgs.csv")

    return x


#==================================
# 6. One Step Secante - OSS (Slide 59-60)
#==================================
def OSS(x, plot_result = False, verbose = False, save = False):
    """
    Optimize equation using one step secant method.
    
    Input
        x:              vector. Initial value for x
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
        
    Output
        x:              optimized values
    """
    P = x.shape[0]

    # Step 1
    epsilon = 1e-3

    # Step 2
    f, g, h = gradient(x)
    g = -g
    d = g
    i = 0

    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(d)})

    # Step 3
    itmax = 100
    while i < itmax and np.linalg.norm(g) > epsilon:

        # Step 4
        if i != 0:
            # slide 59
            A = -(1 + (q.T @ q) / (s.T @ q)) * (s.T @ g) / (s.T @ q) + (q.T @ g) / (s.T @ q)
            B = (s.T @ g) / (s.T @ q)

            d = -g + A * s + B * q

        # Step 5
        if np.mod(i, P) == 0:
            d = g

        # Step 6
        alpha = calc_alpha(x, d)

        # Step 7
        x = x + alpha * d

        # Step 8
        s = alpha * d
        f1, g1, h1 = gradient(x)

        # Step 9
        q = g1 - g

        g = g1

        if verbose:
            print(i,
                  'x1: ', round(x[0][0], 4),
                  'x2: ', round(x[1][0], 4),
                  'f(x): ', round(f1, 4),
                  '|g|: ', round(np.linalg.norm(d), 4))

        # Step 10
        i = i + 1

        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f1,
                        'Gradient': np.linalg.norm(d)},
                        ignore_index=True)

    # Plot results
    if plot_result:
        plot(df, 'One Step Secant')

    # Save result
    if save:
        df.to_csv("oss.csv")

    return x

#==================================
# 7. Conjugate Gradient Method
# Description from "Nonlinear Programming" p. 423
# Hestenes-Stiefel HS  (equation 8.57)
# Polak-Ribiere - PR   (equation 8.58)
# Fletcher-Reeves - FR (equation 8.60)
#==================================
def conjugate_gradient(x, method = "pr", equation = 1, plot_result = False,
                       verbose = False, save = False):
    """
    Optimizes equation using one of conjugate gradient methods.
    
    Input
        x:              vector with initial values
        method:         'hs' = Hestenes-Stiefel
                        'pr' = Polak-Ribiere
                        'fr' = Fletcher-Reeves
        equation:       1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
                        2: f(x1, x2) = r1(x)**2 + r2(x)**2 (method Levenberg-Marquardt)
                        3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
        plot_result:    if True creates plot
        verbose:        if True displays messages
        save:           if True saves results as csv
    
    Output
        x:              minimum value of equation
    """
    # Initialization Step
    epsilon = 1e-3
    y = x
    f, g, h = gradient(x, equation = equation)
    d = -g
    k = 1
    j = 1

    df = pd.DataFrame(data = {'x1' : [x[0][0]],
                              'x2' : [x[1][0]],
                              'f(x)' : f,
                              'Gradient': np.linalg.norm(d)})

    n = len(x)
    i = 0
    itmax = 100
    # Main step
    while np.linalg.norm(g) > epsilon and i < itmax:

        # Step 1
        l = calc_alpha(y, d, equation = equation)
        y = y + l * d

        if verbose:
            print(i, "k: ", k, "j: ", j, "g: ", g.T, "|g|: ", np.linalg.norm(g), "d: ", d.T)


        # Step 2
        if j < n:
            f1, g1, h1 = gradient(y, equation = equation)
            q = g1 - g

            # Hestenes-Stiefel - HS (8.57)
            if method.lower() == 'hs':
                alpha = (g1.T @ q)/(d.T @ q)

            # Polak-Ribiere - PR (8.58)
            elif method.lower() == 'pr':
                alpha = (g1.T @ q) / (np.linalg.norm(g)**2)

            # Fletcher-Reeves - FR (8.60)
            elif method.lower() == 'fr':
                alpha = (np.linalg.norm(g1)**2) / (np.linalg.norm(g) **2)

            d = -g1 + alpha * d
            g = g1
            j = j + 1


        # Step 3
        else:
            x = y
            d = -g
            f, g, h = gradient(y, equation = equation)
            j = 1
            k = k + 1

        i = i + 1

        df = df.append({'x1' : x[0][0],
                        'x2' : x[1][0],
                        'f(x)' : f,
                        'Gradient': np.linalg.norm(d)},
                        ignore_index=True)

    # Hestenes-Stiefel - HS
    if method.lower() == 'hs':
        title = "Hestenes-Stiefel - HS"

    # Polak-Ribiere - PR
    elif method.lower() == 'pr':
        title = "Polak-Ribiere - PR"

    # Fletcher-Reeves - FR
    elif method.lower() == 'fr':
        title = "Fletcher-Reeves - FR"

    # Plot results
    if plot_result:
        plot(df, title, equation)

    # Save result
    if save:
      df.to_csv(method.lower() + '.csv')

    return x

#==================================
# Test all methods defined above
#==================================

# Create initial random values for x1 and x2
np.random.seed(1234)
x1 = np.random.uniform(-10, 10)
x2 = np.random.uniform(-10, 10)
x = np.array([[x1], [x2]])

print("x: \n", x)

# Or use fixed initial values
# Uncomment for fixed value
x = np.array([[1], [4]])

# Plot f(x1, x2) with contour lines
# It's useful to see how the function behaves
plot_function()

plot_result = True
verbose = False
save = False

# Gradient Descent
gradient_descent(x = x, plot_result = plot_result, verbose = verbose, save = save)

# Newton
newton(x = x, plot_result = plot_result, verbose = verbose, save = save)

# Newton Modified
newton_modified(x = x, plot_result = plot_result, verbose = verbose, save = save)

# Levenberg-Marquardt
lm(x = x, plot_result = plot_result, verbose = verbose, save = save)

# Davidon-Fletcher-Powell - DFP
quasi_newton(x = x, method = 'dfp', equation = 1, plot_result = plot_result , verbose = verbose, save = save)

# Broyden-Fletcher-Goldfarb-Shanno (BFGS)
quasi_newton(x = x, method = 'bfgs', equation = 1, plot_result = plot_result , verbose = verbose, save = save)


# One Step Secant
OSS(x = x, plot_result = plot_result , verbose = verbose, save = save)

# Hestenes-Stiefel (HS)
conjugate_gradient(x = x, method = 'hs', equation = 1, plot_result = plot_result , verbose = verbose, save = save)

# Polak-Ribiere - PR
conjugate_gradient(x = x, method = 'pr', equation = 1, plot_result = plot_result , verbose = verbose, save = save)

# Fletcher-Reeves - FR
conjugate_gradient(x = x, method = 'fr', equation = 1, plot_result = plot_result , verbose = verbose, save = save)


#----------------------------------
# Example from the book "Nonlinear Programming"
#----------------------------------

# Initial values for x
# x1 = 0, x2 = 3
# f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
#
x3 = np.array([[0],[3]])
x3

# Davidon-Fletcher-Powell - DFP
quasi_newton(x = x3, method = 'dfp', equation = 3, plot_result = True)

# Broyden-Fletcher-Goldfarb-Shanno (BFGS)
quasi_newton(x = x3, method = 'bfgs', equation = 3, plot_result = True)


# Hestenes-Stiefel (HS)
conjugate_gradient(x = x3, method = 'hs', equation = 3, plot_result = True)

# Polak-Ribiere - PR
conjugate_gradient(x = x3, method = 'pr', equation = 3, plot_result = True)

# Fletcher-Reeves - FR
conjugate_gradient(x = x3, method = 'fr', equation = 3, plot_result = True)




