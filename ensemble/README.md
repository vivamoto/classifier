## Ensemble
Python and NumPy implementation of three ensemble methods of multi layer perceptron (MLP) networks for classification. The code can be adapted for regression.

The training set with ***N*** elements is defined as ***D={(X1, y1), . . ., (XN, yN)}***, where ***X*** is a vector and ***y={0, 1}*** is one-hot encoded.

Sample code at the end of the file.

**Warning**  
DECORATE requires *neuralnets.py* available at https://github.com/vivamoto/Classifier/tree/master/neuralnets

## Ensemble Learning Using Decorrelated Neural Networks  
The idea behind negative correlation learning is to encourage different individual networks in an ensemble to learn different parts or aspects of a training data so that the ensemble can learn the whole training data better. In negative correlation learning, all the individual networks in the ensemble are trained simultaneously through the correlation penalty terms in their error functions. Negative correlation learning attempts to train and combine individual networks in the same learning process. That is, the goal of each individual training is to generate the best result for the whole ensemble. 

### Usage
```
negcor_train(X, yd, ens, lamb = 0.5, plot = True, pdir = '', DS_name = '')
    Ensemble learning via negative correlation.
    Trains an ensemble of multi layer perceptron networks, with negative
    correlated errors.

    Input
       X, yd:       training set
       ens:         ensemble object with parameters
       lamb:        lambda parameter. Provides a way to balance the
                    bias-variance-covariance trade-off
       plot:        True to make plot
       DS_name:     dataset name used in plot title
       pdir:        directory to save the plot. Uncomment to save.

     Output
       ens:        ensembe parameters with weights and biases

negcor_predict(X_test, nc, ens)
    Predict ensemble via negative correlation.
     
    Input
       X_test:       test set
       nc:           number of classes
       ens:          ensemble object
    
    Output
       y_hat:        predicted values
```
## Ensemble Learning Using Decorrelated Neural Networks  
Create ensemble networks that are linear combinations of individually trained networks. The approach taken here is to train the individual networks not only to reduce their approximation errors and but also to reduce the correlations of individual network’ s errors.
### Usage
```
decorrelated_train(X, yd, ens, lamb, alternate = False, plot = True, pdir = '', DS_name = '')
    Ensemble Learning Using Decorrelated Neural Networks.
    Trains an ensemble of multi layer perceptron networks, using decorrelated
    neural networks.
    
    Input
       X, yd:       training set. yd is one-hot encoded.
       ens:         ensemble object
       lamb:        The scaling function lambda(t) is either constant or is time
                    dependent. Typically determined by cross-validation.
       alternate:   If False, individual networks are decorrelated with the
                    previously trained network. if True, alternate networks are
                    trained independently of one another yet decorrelate pairs of networks.
       plot:        True to make plot
       pdir:        directory to save the plot. Uncomment to save.
       DS_name:     dataset name used in plot title

    Output
       ens:     ensemble object with weights and biases
```

## Diverse Ensemble Creation by Oppositional Relabeling of Artiflcial Training Examples (DECORATE)
DECORATE uses an existing "strong" learner (one that provides high accuracy on the training data) to build an effective diverse committee in a simple, straightforward manner. This is accomplished by adding different randomly constructed examples to the training set when building new committee members. These artiflcially constructed examples are given category labels that disagree with the current decision of the committee, thereby easily and directly increasing diversity when a new classifler is trained on the augmented data and added to the committee.

In Decorate, an ensemble is generated iteratively, flrst learning a classifler and then adding it to the current ensemble. 

**Warning**  
DECORATE requires *neuralnets.py* available at https://github.com/vivamoto/Classifier/tree/master/neuralnets

### Usage
```
decorate_train(X_train, y_train, Csize =15, Imax = 50, Rsize = 0.5):
    Creating Diversity In Ensembles Using Artificial Data
    Trains an ensemble of multi layer perceptron networks, adding artificially
    created data.

    Inputs
        X_train:        normalized X values
        y_train:        train set, one-hot encoded
        Csize:          desired ensemble size
        Imax:           maximum number of iterations to build an ensemble
        Rsize:          factor that determines number of artificial examples to generate

    Output
        C:              dictionary with ensemble weights and biases

decorate_predict(X_test, C):
    Ensemble Learning Using Decorrelated Neural Networks.
    Predict from trained ensemble.
    
    Input
        X_test:         test set
        C:              dictionary with ensemble weights and biases
    
    Output
        y_hat:          predicted values, one-hot encoding        
```

## Backpropagation
The derivatives calculations are available at:  
https://github.com/vivamoto/Classifier/blob/master/Neural_Network_Derivatives.pdf
## Reference
Ensemble learning via negative correlation  
Y. Liu, X. Yao  
https://doi.org/10.1016/S0893-6080(99)00073-8

Ensemble Learning Using Decorrelated Neural Networks  
BRUCE E ROSEN  
https://doi.org/10.1080/095400996116820

Creating Diversity In Ensembles Using Artificial Data  
Prem Melville and Raymond J. Mooney  
http://www.cs.utexas.edu/~ml/papers/decorate-jif-04.pdf
