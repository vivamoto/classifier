## Regression
[![DOI](https://zenodo.org/badge/279084447.svg)](https://zenodo.org/badge/latestdoi/279084447)

Python and NumPy implementation from scratch of regression methods for classification. The code can be adapted for regression.

- Linear regression for classification
- Linear regression with regularization
- Logistic regression

The training set with ***N*** elements is defined as ***D={(X1, y1), . . ., (XN, yN)}***, where ***X*** is a vector and ***y={0, 1}*** is a column vector for binary classification and one-hot encoded for multi-class.

Sample code at the end of the file.

## Usage
```
regression_train(X, y):
    Train linear regression for classification
    
    Input
        X:      train set without bias
        y:      y in {0, 1}, vector for binary classification or one-hot encoded for multiclass
    Output
        w:      regression weights
        
        
regression_predict(X, w):
    Predict linear regression for classification
    
    Input
        X:          test set without bias
        w:          weights matrix
    Output
        y_hat:      predicted values        
        
        
logistic_train(X, y, v=True, binary=True, maxiter=1000, lr=0.1):
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


logistic_predict(X, w, binary):
    Predict logistic regression
    Input
        X:          test set
        w:          weights matrix
        binary:     True for binary classification
    Output        
        y_hat:      predicted values


regularization(X, y):
    Linear regression for classification with regularizaton.
    Perform grid search with cross validation to find the best regularization factor.

    Input
        X:      train set
        y:      train set, one-hot encoded
    Output
        Return the best values: weight, lambda, accuracy and data frame        
    
```


## Reference

An Introduction to Statistical Learning  
G. James, D. Witten, T. Hastie, R. Tibshirani

Lecture notes CS480/680–Fall 2018 - University of Waterloo  
Yaoliang Yu  
https://cs.uwaterloo.ca/~y328yu/mycourses/480/note06.pdf  
        
MATLAB code from Clodoaldo Lima  

Lima, C.A.M., 2020, *Aula 02 - Modelos Lineares*, lecture notes, Universidade de Sao Paulo, delivered April, 2020.

Eli Bendersky website  
https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
