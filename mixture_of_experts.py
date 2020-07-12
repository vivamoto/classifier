# m - numero de especialistas
# X - matriz de entrada
# Y - matriz de saida
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
pdir = 'D:/Documentos/Profissão e Carreira/Mestrado/Aprendizado de Máquina/Aulas/'
os.chdir(pdir)

ts = np.loadtxt("mistura_especialista_conj_treinamento.txt")
best_lag(ts)
lag = 20
m = cv_experts(ts = ts, lag = lag, gating='linear')
final_ME(ts = ts, m = 5, lag = lag, gating = 'linear')

def best_lag(ts):
    # Input
    # ts: time series dataset

    # 1
    end = 50
    X = ts[:end]
    plt.figure()
    for lag in range(12):
        X1 = ts[lag:end+lag]
        plt.subplot(4, 3, 1 + lag)
        plt.plot(np.arange(len(X)), X, np.arange(len(X1)), X1)
        plt.title("Lag: " + str(lag))
    plt.show()

    #=======================
    # Ref: Bruno Kemmer
    df = pd.DataFrame(ts)
    df.columns = ['y']

    lags = 25
    for i in range(1, lags + 1):
        df['y_' + str(i)] = df['y'].shift(i)
    df = df.dropna()

    # 3 Correlation heatmap
    plt.figure()
    plt.title('Correlação entre todas as variáveis e atrasos')
    sns.heatmap(df.corr())
    # plt.savefig('imgs/me_corrplot.png')
    # plt.show()

    # 4 Correlation line chart
    plt.figure()
    plt.plot(df.corr()['y'])
    plt.axhline(y=0.1, color='y', linestyle='--', lw=.5)
    plt.axhline(y=-0.1, color='y', linestyle='--', lw=.5)
    plt.axhline(y=0.05, color='r', linestyle='--', lw=.5)
    plt.axhline(y=-0.05, color='r', linestyle='--', lw=.5)
    plt.axvline(x=12, color='b', linestyle='--', lw=0.5)
    plt.axvline(x=18, color='b', linestyle='--', lw=0.5)
    plt.title('Correlação entre variável resposta e atrasos')
    plt.xlabel('Atrasos')
    plt.ylabel('Correlação com y')
    # plt.savefig('imgs/.png')
    plt.show()

    return

# Ref: Committee machines: a unified approach using support vector machines
# C. A. M. Lima
def cv_experts(ts, lag = 20, gating = 'linear'):
    # Input:
    # ts: time series dataset
    # lag: time lag used to build the (x, y) dataset
    # gating: gating type 'linear' or 'gaussian'

    kfolds  = 5         # number of cross-validation folds
    bestm   = 1         # best number of experts
    lik_max = 0         # best likelihood

    #------------------------------------------
    # 1. Convert time series to  (X, Y) dataset
    #------------------------------------------
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
    mmax = int(np.ceil((N - lag) / (2 * (lag + 1))))
    for m in range(1, 3):

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
                p_xv = prob(X_val, gamma, sigma)
                ygat = gating_output(X_val, p_xv, alpha)    # gating output
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
        lik_max = cv_lik if m == 1 else lik_max
        if cv_lik > lik_max:
            lik_max = cv_lik    # highest likelihood
            bestm = m           # best number of experts

        # Keep data for plot
        plotData.loc[len(plotData) + 1] = [m, lik]

        if m % 10 == 0:
            print("CV Likelihood: ", cv_lik)

    #------------------------------------------
    # 3 Plot
    #------------------------------------------
    title = gating.title() + ' Gating - ' + lag + ' Lags'
    plt.scatter('Number of Experts', 'Likelihood', data=plotData)
    plt.xlabel('Number of Experts')
    plt.ylabel('Likelihood')
    plt.title(title)
    plt.savefig(pdir + title + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return bestm

# Train and predict the model, then creates a plot
def final_ME(ts, m, lag = 20, gating = 'linear'):
    # Input:
    # ts: time series dataset
    # m: number of experts
    # lag: time series lag
    # gating: gating type 'linear' or 'gaussian'

    #------------------------------------------
    # 1. Create (X, Y) dataset
    #------------------------------------------
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

    y_test = np.array([y_test]).T

    # Insert column of 1s for bias
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test  = np.insert(X_test, 0, 1, axis=1)

    #------------------------------------------
    # 3 Train and predict the model
    #------------------------------------------
    wexp, wgat, var, alpha, gamma, sigma = ME(X_train, y_train, m = m, gating = gating, add_bias = False)

    # Predict ME output
    if gating == 'linear':
        ygat = softmax(X_test @ wgat.T)     # gating output
    elif gating == 'gaussian':
        p_xv = prob(X_test, gamma, sigma)
        ygat = gating_output(X_test, p_xv, alpha)
    yexp = calc_saida_esp(X_test, wexp)     # expert output
    y_hat = np.sum(ygat * yexp, 1)          # ME output

    # Compute likelihood
    p = calc_prob_exp(yexp, y_test, var)
    lik  = likelihood(ygat, p)

    #------------------------------------------
    # 4 Plot predicted and real values
    #------------------------------------------
    xrange = np.arange(N-mid)
    plt.plot(xrange, y_test, xrange, y_hat)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.suptitle('Real vs Predicted Time Series')
    plt.title(gating.title() + ' Gating, ' + lag + ' Lags, ' + m + ' Experts')
    plt.savefig(pdir + gating + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    return


def ME(X, Yd, m = 4, gating = 'linear', add_bias = False):
    # Input:
    # X:  input matrix
    # Yd: desired ME output
    # m:  number of experts
    # add_bias: if True, add column of ones in X matrix

    # Dimensions:
    # wexp (ns x ne x m)
    # wgat (m x ne)
    # yexp (N, ns, m)
    # ygat (N x m)
    # g = ygat = (N x m)
    # p (N x m)
    # h (N x m) ?

    # Add column with 1s for the bias
    if add_bias:
        X = np.insert(X, 0, 1, axis=1)
    if Yd.ndim == 1:
        Yd = np.array([Yd]).T
    N, ne = X.shape

    ns    = Yd.shape[1]                 # number of ME outputs
    var   = np.ones(m)                  # variance
    wexp  = np.random.rand(ns, ne, m)   # expert weight
    wgat  = np.random.rand(m, ne)       # linear gating weight
    ygat  = softmax(X @ wgat.T)         # gating output (a priori probability)
    yexp  = calc_saida_esp(X, wexp)     # expert output

    alpha = np.random.rand(m)           # gaussian gating parameter
    gamma = np.random.rand(m, ne)       # gaussian gating parameter
    sigma = np.random.rand(m)           # gaussian gating parameter

    p = calc_prob_exp(yexp, Yd, var)    # P(y|X,tetha)
    p_xv = np.zeros((N, m))             # P(X|v)
    lik_old = 0
    lik_new = likelihood(ygat, p)
    it = 0
    itmax = 1000
    while abs(lik_new - lik_old) > 1e-3 and it < itmax:
        #print('lik = %g', lik_new, "diff: ", abs(lik_new - lik_old))
        # Linear gating
        if gating == 'linear':
            # Step E - Estimar H
            p = calc_prob_exp(yexp, Yd, var)    # P(y|X,tetha)
            h = prob_h(ygat, p)
            # Step M: optimize
            for i in range(m):
                wexp[:,:,i], var[i] = atualiza_exp(X, Yd, h[:, i], var[i], wexp[:,:,i], ns)
            wgat = atualiza_gat(X, h, wgat) # gating weights
            ygat = softmax(X @ wgat.T)      # gating output
            yexp = calc_saida_esp(X, wexp)  # expert output
        # Gaussian gating
        else:
            # Step E:
            p_xv = prob(X, gamma, sigma)
            p = calc_prob_exp(yexp, Yd, var)
            h = alpha * p_xv * p / np.sum(alpha * p_xv * p, axis = 1, keepdims=True)
            # Step M:
            for i in range(m):
                wexp[:,:,i], var[i] = atualiza_exp(X, Yd, h[:, i], var[i], wexp[:,:,i], ns)
            alpha = np.sum(h, axis=0) / np.sum(h)
            gamma = h.T @ X / np.sum(h, axis=0, keepdims=True).T
            for i in range(m):
                s = 0
                for t in range(N):
                    s += h[t, i] * (X[t, :] - gamma[i, :]) @ (X[t, :] - gamma[i, :]).T
                sigma[i] = s / np.sum(h[i], axis=0, keepdims=True)

            ygat = gating_output(X, p_xv, alpha)    # gating output
            yexp = calc_saida_esp(X, wexp)          # expert output

        it += 1
        lik_old = lik_new               # likelihood
        lik_new = likelihood(ygat, p)

        #print('lik = %g', lik_new, "diff: ", abs(lik_new - lik_old))

    return wexp, wgat, var, alpha, gamma, sigma 


def atualiza_gat(X, h, w):
    # Use gradient ascent to update the gating weights
    # until the gradient is near zero, i.e. reach a local maximum.
    Y = X @ w.T         # gating is linear combination of input
    g = softmax(Y)      # gating output (a priori probability)
    dQdE = (h - g)      # slide 129
    dQdw = dQdE.T @ X
    lr = 0.1            # learning rate
    it = 0              # iteration counter
    itmax = 1500        # maximum iterations
    while np.linalg.norm(dQdw) > 1e-3 and it < itmax:
        it = it + 1
        w = w + lr * dQdw       # update weights array
        Y = X @ w.T             # update gate
        g = softmax(Y)          # gating output (a priori probability)
        dQdE = (h - g)          # Q derivative w.r.t. gating output
        dQdw = dQdE.T @ X       # Q derivative w.r.t. weights
    return w


def atualiza_exp(X, Yd, h, var, w, ns):
    # Use gradient ascent to update the expert weights
    # until the gradient is near zero, i.e. reach a local maximum.
    # Input:
    # X, Yd: dataset
    # h: a posteriori probability
    # var: variance
    # w: expert weight
    # ns:>
    h = np.array([h]).T
    y = X @ w.T
    dQdu = (h / var) * (Yd - y)     # slide 129
    dQdw = dQdu.T @ X               # slide 129
    lr = 0.05                       # learning rate
    it = 0                          # iteration counter
    itmax = 1000                    # maximum iterations
    while np.linalg.norm(dQdw) > 1e-3 and it < itmax:
        it = it + 1
        lr = calc_lr(X, w, d = dQdw, h = h, Yd = Yd, var = var)
        w = w + lr * dQdw
        y = X @ w.T
        dQdu = (h / var) * (Yd - y)   # slide 129
        dQdw = dQdu.T @ X

    var = (1 / ns) * np.sum(h * (np.linalg.norm(Yd - y, axis=1, keepdims=True) ** 2), axis=0) / np.sum(h, axis = 0) # slide 129
    # prevent overflow, nan and infinity when computing the expert probability
    var = np.maximum(var, 1e-6)

    return w, var


def likelihood(g, p):
    return np.sum(np.log(np.sum(g * p, 1)), axis = 0)


def prob_h(g, p):
    # g = ygat = (N x m)
    # p (N x m)
    # h (N x m) ?
    num = g * p
    den = np.sum(num, axis = 1, keepdims = True)
    h = num / den
    return h


def calc_prob_exp(yexp, Yd, sig):
    N, m = yexp.shape
    ns = Yd.shape[1]
    p = np.zeros((N, m))
    for i in range(m):
        Y = yexp[:,i:i+1]
        for n in range(N):
            p[n, i] = (1 / ((2 * np.pi * sig[i]) ** (ns / 2))) * \
                      np.exp(-np.linalg.norm(Yd[n,:] - Y[n,:])/(2*sig[i]))  # Victor (slide 129)

    return p


def softmax(s, axis = 1):
    # stable version of softmax prevents overflow for large values of s
    # ygat (N x m)
    max_s = np.max(s, axis=axis, keepdims=True)
    e = np.exp(s - max_s)
    ygat = e / np.sum(e, axis=axis, keepdims=True)

    return ygat

def calc_saida_esp(X, wexp):
    # yexp (N, ns, m)
    N = X.shape[0]
    ns, ne, m = wexp.shape
    yexp = np.zeros((N, m))
    for i in range(m):
        yexp[:,i:i+1] = X @ wexp[:,:,i].T
    return yexp


def gating_output(X, p_xv, alpha):
    return alpha * p_xv / np.sum(alpha * p_xv, axis=1, keepdims=True)


def prob(X, gamma, sigma):
    N, ne = X.shape
    m = len(sigma)
    p_xv = np.zeros((N, m))
    for t in range(N):
        for i in range(m):
            p_xv[t, i] = 1 / ((2 * np.pi) ** (ne / 2) * abs(sigma[i] ** 0.5)) * np.exp(
                1 / 2 * (X[t, :] - gamma[i:i + 1, :]) @ (sigma[i] ** (-1) * (X[t, :] - gamma[i:i + 1, :])).T)
    return p_xv

def calc_lr(X, w, d, h, Yd, var):
    # d = direction
    np.random.seed(1234)
    epsilon = 1e-3
    hlmin = 1e-3
    lr_l = 0  # Lower lr
    lr_u = np.random.rand() * 1e-6  # Upper lr

    # New w position
    wn = w + lr_u * d
    y = X @ wn.T
    # Calculate the gradient of new position
    g = exp_derivative(X, Yd, y, h, var)

    hl = g @ d.T
    while hl < 0:
        lr_u = 2 * lr_u
        # Calculate the new position
        wn = w - lr_u * d
        y = X @ wn.T
        # Calculate the gradient of new position
        # f and h aren't used
        g = exp_derivative(X, Yd, y, h, var)
        hl = g @ d.T
        print(hl)

    # lr medium is the average of lrs
    lr_m = (lr_l + lr_u) / 2
    # Estimate the maximum number of iterations
    itmax = np.ceil(np.log((lr_u - lr_l) / epsilon))
    it = 0
    while abs(hl) > hlmin and it < itmax:
        # Calculate new position
        wn = w - lr_m * d
        y = X @ wn.T

        # Calculate the gradient of the new position
        # Note: f and h aren't used
        g = exp_derivative(X, Yd, y, h, var)

        hl = g @ d.T
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

# Expert derivative
def exp_derivative(X, Yd, y, h, var):
    dQdu = (h / var) * (Yd - y)  # slide 129
    dQdw = dQdu.T @ X
    return dQdw

# Gating derivative
def gat_derivative(X, y, h):
    #w = w + lr * dQdw  # update weights array
    #Y = X @ w.T  # update gate
    g = softmax(y)  # gating output (a priori probability)
    dQdE = (h - g)  # Q derivative w.r.t. gating output
    dQdw = dQdE.T @ X  # Q derivative w.r.t. weights
    return dQdw

