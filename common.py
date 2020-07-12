import numpy as np
import pandas as pd

# ======================
# 1.5 Create the training and test sets
# ======================
# The train and test dasasets for diabetes were provided
# We use 2/3 of the data set for training and 1/3 for test.
# For hepatitis, the initial part is used for training.

# Iris
# Use every 3 rows for test set and the other 2 for training.
# Example:
# Train rows = 0, 1, 3, 4, 6, 7, ...
# Test rows  = 2, 5, 8, 11, ...
def create_train_test(DS_name, DS=None):
    # Input
    # DS: complete dataset
    # DS_name: dataset name
    if DS_name == 'iris':
        row = range(DS.shape[0])  # Range of row numbers, start with 0
        row = np.array(row) + 1  # Convert to vector and add 1
        train = DS.iloc[np.mod(row, 3) != 0]  # Skip every 3 rows
        test = DS.iloc[np.mod(row, 3) == 0]  #

    elif DS_name == 'hepatitis':
        # The initial part is used for training.
        mid = int(DS.shape[0] * 2 / 3)
        end = DS.shape[0]
        train = DS[0:mid]  # Training
        test = DS[mid:end]  # Test
    elif DS_name == 'diabetes':
        train = pd.read_csv("diabetes/dataset_train.txt", sep='\t')
        test = pd.read_csv("diabetes/dataset_test.txt", sep='\t')

    return train, test


# ======================
# 1.6 Data normalization
# ======================
# Each dependent variable is normalizaded using min-max or z-score.
# min-max transforms the values to fit in the range [0, 1].
# The formula is:
# x = (x - xmin) / (xmax - xmin),
# where 'xmax' is the maximum value and 'xmin' is the minimum value
# z-score uses normal distribution with mean = 0 and standard deviation = 1
def normalize(X, method, mu=0, sd=0, xmin=0, xmax=0):
    # Min - max
    if method == 'min-max':
        if min == 0 or max == 0:
            xmin = X.min()
            xmax = X.max()
        return (X - xmin) / (xmax - xmin), xmin, xmax

    # Norm Z
    elif method == 'z-score':
        if mu == 0 or sd == 0:
            mu = X.mean()
            sd = np.std(X)
        return (X - mu) / sd, mu, sd

    return None


def norm_X(train, test, fields, method):
    # Input
    # train, test: X sets to normalize
    # fields: name or range of fields
    # method: min-max or z-score
    # Output
    # train, test: normalized sets

    for i in fields:
        if method == 'min-max':
            train.loc[:, i], xmin, xmax = normalize(X=train.loc[:, i], method=method)
            test.loc[:, i],  xmin, xmax = normalize(X=test.loc[:, i], xmin=xmin, xmax=xmax, method=method)
        elif method == 'z-score':
            train.loc[:,i], mu, sd = normalize(X=train.loc[:, i].copy(), method=method)
            test.loc[:, i], mu, sd = normalize(X=test.loc[:, i], mu=mu, sd=sd, method=method)

    return train, test

# ==================================
# Feature selection
# ==================================
def feature_selection(X, y):
    # This function is not in use
    # Input:
    # X: matrix of features
    # y: matrix of outcomes

    nc = X.shape(1)

    # Build the list of best individual predictors
    for i in range(nc):
        w = train(X)
        y_hat = predict(X, w)
        acc_temp = np.mean(y == y_hat)
        if i == 0:
            k = i
            acc = acc_temp
        else:
            acc = np.hstack(acc, acc_temp)


def create_xy(train, test, dataset):
    if dataset == 'diabetes':
        y_class = 'class'
        x_class = ['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7', 'atr8']
    elif dataset == 'hepatitis':
        y_class = ['Class']
        x_class = ['AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM',
                   'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN',
                   'HISTOLOGY']
    # Iris dataset contains 3 classes and is used for multiclass classification
    elif dataset == 'iris':
        y_class = ['setosa', 'versicolor', 'virginica']
        x_class = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Independent variable (Y):
    y_train = train.loc[:, y_class]
    y_test = test.loc[:, y_class]

    # Dependent variables (X):
    X_train = train.loc[:, x_class]
    X_test = test.loc[:, x_class]

    # Convert to numpy arrays
    y_train = np.array(y_train) if y_train.ndim == 0 else np.array(y_train, ndmin=2).T
    y_test  = np.array(y_test)  if y_test.ndim  == 0 else np.array(y_test, ndmin=2).T
    X_train = np.array(X_train)
    X_test  = np.array(X_test)

    return X_train,y_train,X_test,y_test

