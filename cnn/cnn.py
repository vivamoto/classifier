#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network

Flexible architecture: setup the number of layers
- Convolution layer
- Pooling layer
- Full connected layer

Activation functions: relu and sigmoid
Output layer: softmax

Features:
- dropout
- regularization
- minibatch
- momentum

Sample code in the end of this file.

Original MATLAB code:
https://github.com/ClodoaldoLima/Convolutional-Neural-Networks---Matlab

Author: Clodoaldo A. M. Lima
Translated to Python by Victor Ivamoto, Wesley Ramos
Translated to English: Victor Ivamoto
July, 2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, cuda

class obj:
    pass


# *************************************************************
# Feedforward
# ************************************************************
def ff_conv(images, W, b, strider, fativ):
    """
    Convolution layer
    
    Args:
        images:         array (dimX x dimY x numCanais x numImagens), where:
                        dimX, dimY:     image size
                        numCanais:      number of channels
                        numImagens:     number of images

        W:              (dimFiltroX x dimFiltroY x numCanais x numFilters)
        b:              bias
        strider:        strider size
        fativ:          activtion function, 'relu' or 'sig'

    Returns:

    """

    filterDim = W.shape[0]
    numFilters1 = W.shape[2]    # number of channels
    numFilters2 = W.shape[3]
    numImages = images.shape[3] # number of images
    imageDimX = images.shape[0] # image X dimension
    imageDimY = images.shape[1] # image Y dimension
    convDimX = int(np.floor((imageDimX - filterDim) / strider) + 1)
    convDimY = int(np.floor((imageDimY - filterDim) / strider) + 1)

    dtype = np.float64
    Features = np.zeros((convDimX, convDimY, numFilters2, numImages))
    # Activation function derivative
    dfativ   = np.zeros((convDimX, convDimY, numFilters2, numImages), dtype = dtype)
    # Convolution result, position
    result_conv = np.zeros((convDimX, convDimY, numFilters1 + 1))

    for i in range(numImages):
        for fil2  in range(numFilters2):
            convolvedImage = np.zeros((convDimX, convDimY))
            dconvolvedImage = np.zeros((convDimX, convDimY))
            for fil1 in range(numFilters1):
                filter = W[:, :, fil1, fil2]
                im = images[:, :, fil1, i]
                # Perform convolution - doesn't rotate the filter
                result_conv[:,:,fil1] = conv_mod(im, filter, strider)
            
            if fativ == 'relu':
                # Apply bias before or afther the maximum has no effect on index
                # max(x1+b,x2+b)
                # Add bias in all channels
                result_conv[:,:, :-1] = result_conv[:,:, :-1] + b[fil2]
                # Relu activation function
                convolvedImage  = np.max(result_conv, axis = 2)
                # Index of activated neurons
                dfativ[:, :, fil2, i] = np.argmax(result_conv, axis = 2)
            elif fativ == 'sig':
                convolvedImage = np.sum(result_conv, axis = 2)  # Add convolution
                convolvedImage = convolvedImage + b[fil2]       # Add bias
                convolvedImage = 1 / (1 + np.exp(-convolvedImage)) # sigmoid
                dfativ[:, :, fil2, i] = (1 - convolvedImage) * convolvedImage # sigmoid derivative

            Features[:, :, fil2, i] = convolvedImage

    return Features, dfativ


def ff_full(activation, W, b, fativ):
    """
    Full connected layer output.
    
    Args:
        activation:
        W:          weights array
        b:          bias
        fativ:      activation function

    Returns:

    """
    m, N, numFilter, numImagens = activation.shape

    Z = np.zeros((W.shape), order = 'F')
    dZ = np.zeros((W.shape), order = 'F')
    Zin = np.zeros((W.shape), order = 'F')
    if numImagens > 1:   # Images already concatenated
        # Join previous output to next layers
        activation = np.reshape(activation,(-1, numImagens), order = 'F')
        N = numImagens

    if fativ == 'sig':
        Zin = W @ activation + b #h x N
        activation = 1/(1 + np.exp(-Zin, order = 'F'))    #n x N
        dativacao = (1 - activation) * activation

    elif fativ == 'relu':
        h, ne = W.shape
        for i in range(h):
            for j in range(ne):
                Zin[j, :] = W[i, j] @ activation[j, :]
            val, pos = np.maximum(0, Zin, order = 'F')
            Z[i, : ] = val
            dZ[i, :] = pos - 1   #remap 1 to 0
        activation = Z
        dativacao = dZ

    return activation, dativacao

# Pooling output
def ff_pool(poolDim, convolvedFeatures,strider,criterio):
    """
    Pooling layer output.
    
    Args:
        poolDim:            pooling dimension
        convolvedFeatures:  array of images after convolution
        strider:            strider size
        criterio:           pooling type: 'max' or 'mean'

    Returns:

    """

    numImages = convolvedFeatures.shape[3]
    numFilters = convolvedFeatures.shape[2]
    convolvedDimx = convolvedFeatures.shape[0]
    convolvedDimy = convolvedFeatures.shape[1]
    # image size after pooling
    dimx = int(np.floor((convolvedDimx - poolDim) / strider) + 1)   
    dimy = int(np.floor((convolvedDimy - poolDim) / strider) + 1)   

    pooledFeatures = np.zeros((dimx, dimy, numFilters, numImages), order = 'F')
    dpooledFeatures = np.zeros((dimx * dimy, 2, numFilters, numImages), order = 'F')

    for imageNum in range(numImages):
        for featureNum in range(numFilters):
            featuremap = np.squeeze(convolvedFeatures[:,:,featureNum,imageNum])
            if criterio == 'mean':
                pooledFeaturemap, dpooledFeaturemap = conv_mod(featuremap, 1 / (poolDim ** 2), strider)
            elif criterio == 'max':
                pooledFeaturemap, dpooledFeaturemap = conv_max(featuremap, np.ones((poolDim, poolDim), order = 'F'), strider)

            pooledFeatures[:,:,featureNum,imageNum] = pooledFeaturemap
            dpooledFeatures[:,:,featureNum,imageNum] = dpooledFeaturemap

    return pooledFeatures, dpooledFeatures

@jit
def conv_comp(img, mask, strider):
    m1, n1 = img.shape
    m2, n2 = mask.shape
    if strider > 1:
        tamx = np.floor(m1 / m2)  # size of filter x
        tamy = np.floor(n1 / n2)  # size of filter y
        dimx = np.int(strider * m2 + ((m1 - tamx) % strider) - 1)
        dimy = np.int(strider * n2 + ((n1 - tamy) % strider) - 1)
        maskaux = mask
        mask = np.zeros((dimx,dimy))
        ix = 0
        for i in range(0, strider * m2, strider):
            jx=0
            for j in range(0, strider*n2, strider):
                mask[i,j] = maskaux[ix, jx]
                jx = jx + 1
            ix = ix + 1
        m2 = dimx
        n2 = dimy

    mConv = (m1 - m2 + 1)
    nConv = (n1 - n2 + 1)
    Conv = np.zeros((mConv, nConv))
    x = 0
    y = 0
    for i in range(mConv,):
        for j in range(nConv):
            img1 = img[x:x + m2,y : y + n2]
            Conv[i, j] = np.sum(img1 * mask)
            y = y + 1
        x = x + 1
        y = 0

    return Conv

@jit
def conv_full(img, mask, dimX, dimY, strider):

    mask[0][0],mask[1][1] = mask[1][1],mask[0][0]
    mask = mask.T
    # mask = np.rot90(mask, k=2)  # rotate the kernel

    m1, n1 = img.shape      # e.g.: 27x27
    m2, n2 = mask.shape     # e.g.: 2x2

    # addd zeros before and in the middle e.g.: (29x29) for strider = 1
    tamx = m1 + (2 * m2) - 2 + (strider - 1) * (m1 - 1)
    tamy = n1 + (2 * n2) - 2 + (strider - 1) * (n1 - 1)

    img1 = np.zeros((tamx, tamy))

    kx = 0
    for i in range(m2-1, m2 + strider * (m1 - 1), strider):
        ky = 0
        for j in range(n2-1, n2 + strider * (n1 - 1), strider):
            img1[i, j] = img[kx, ky]
            ky = ky + 1
        kx = kx + 1

    strider = 1    # convolution is strider 1
    m1, n1 = img1.shape
    mConv = np.int(np.floor((m1 - m2) / strider) + 1)
    nConv = np.int(np.floor((n1 - n2) / strider) + 1)

    Conv = np.zeros((dimX, dimY))
    x = 0
    y = 0
    for i in range(mConv):
        for j in range(nConv):
            img2 = img1[x:x + m2, y:y + n2]
            Conv[i, j] = np.sum(img2 * mask)
            y = y + strider
        x = x + strider
        y = 0

    return Conv
#***********************************************************************
# Performs convolution, no kernel rotation
#*******************************************************************
@jit
def conv_max(img, mask, strider):
    """
    Convolution with no kernel rotation.
    
    Args:
        img:
        mask:
        strider:

    Returns:

    """
    m1, n1 = img.shape
    m2, n2 = mask.shape

    mConv = np.int(np.floor((m1 - m2) / strider) + 1)
    nConv = np.int(np.floor((n1 - n2) / strider) + 1)
    Conv = np.zeros((mConv, nConv))
    dConv = np.zeros((mConv * nConv, 2))    # stores the positions
    x = 0
    y = 0
    cont = 0
    for i in range(mConv):
        for j in range(nConv):
            img1 = img[x:x + m2, y:y + n2]
            val = np.max(img1)
            px, py = np.where(img1==val)
            Conv[i, j] = val
            dConv[cont, 0] = px[0]  # stores the indexes of maximum values
            dConv[cont, 1] = py[0]  # of image in the kernel
            y = y + strider
            cont = cont + 1

        x = x + strider
        y = 0

    return Conv, dConv


#***********************************************************************
# Performs convolution, no kernel rotation
#*******************************************************************
@jit
def conv_mod(img,mask,strider):
    """
    Performs convolution, no kernel rotation
    
    Args:
        img:
        mask:
        strider:

    Returns:

    """
    m1, n1 = img.shape
    m2, n2 = mask.shape

    mConv = np.int(np.floor((m1 - m2) / strider) + 1)
    nConv = np.int(np.floor((n1 - n2) / strider) + 1)
    Conv = np.zeros((mConv, nConv))
    x = 0
    y = 0
    for i in range(mConv):
        for j in range(nConv):
            img1 = img[x:x + m2 , y:y + n2 ]
            Conv[i, j] = np.sum(img1 * mask)
            y = y + strider

        x = x + strider
        y = 0

    return Conv

#*************************************************************
# Neuron derivatives
#*************************************************************
def delta_pool(delta,dConv,dimPool,strider,dimx,dimy,criterio):
    """
    Pooling neurons derivative.
    
    Args:
        delta:      neuron derivative
        dConv:      derivative of convolution
        dimPool:    pooling dimension
        strider:    strider
        dimx:       
        dimy:       
        type:       pooling type: 'max' or 'mean'

    Returns:
        Convolution neuron derivative.
    """

    m, n = delta.shape
    deltaConv = np.zeros((dimx, dimy), order='F') # pooling size

    if type == 'max':
        p = 0
        for i in range(m):
            for j in range(n):
                px = dConv[p, 0]
                py = dConv[p, 1]
                Ix = int(px + strider * i)
                Iy = int(py + strider * j)
                deltaConv[Ix, Iy] = deltaConv[Ix, Iy] + delta[i, j]
                p = p + 1

    elif type == 'mean':
        for i in range(m):
            for j in range(n):
                for kx in range(dimPool):
                    for ky in range(dimPool):
                        Ix = kx + strider * i
                        Iy = ky + strider * j
                        deltaConv[Ix, Iy] = deltaConv[Ix, Iy] + delta[i, j]

    return deltaConv


def delta_full(delta, W, dfativ, fativ):
    """
    Compute neuron derivative.

    Args:
        delta:      neuron derivative 
        W:          weights array
        dfativ:     activation function derivative
        fativ:      activation funciton

    Returns:
        Neuron derivative
    """
    
    m, N = dfativ.shape
    h, ne = W.shape
    
    if fativ == 'sig':  # sigmoid activation function
        delta_full = W.T @ (delta * dfativ)
    
    elif fativ == 'relu':  # relu activation function
        delta_full = np.zeros((ne, N), order = 'F')
        for i in range(ne):
            dfativ_aux = np.where(dfativ == i, 1, 0)
            for j in range(h):
                delta_full[j, :] = delta_full[j, :] + W[j, i] @ (
                            delta * dfativ_aux[j, :])
    
    return delta_full


#*************************************************************
# dropout: freeze neurons
#*************************************************************
def dropout(y, dy, idx, tx):
    """
    Freeze neurons.
    
    Args:
        y:      activation or array with images
        dy:     activation derivative
        idx:    frozen individuals
        tx:     frozen rate

    Returns:

    """

    dimX, dimY, numFilters , numImagens = y.shape
    num = len(idx) - 1
    cont = 0
    for i in range(numFilters):
        if cont <= num:
            if idx[cont] == i: # neuron is frozen
                # activation and derivative are zeros
                y[:,:,i,:] = np.zeros((dimX, dimY, numImagens), order = 'F')
                dy[:,:,i,:] = np.zeros((dimX, dimY, numImagens), order = 'F')
                cont = cont + 1
            else:
                y[:,:,i,:] = y[:,:,i,:]  / (1 - tx) # increase the output

        else:
            y[:,:,i,:] = y[:,:,i,:]  / (1 - tx)  # increase the output

    return y, dy

def init_params(cnn,opts):
    """
    Setup CNN based on network parameters.
    
    Args:
        cnn:    object with cnn parameters
        opts:   options object

    Returns:
        cnn object with initial setup
    """
    numFiltros1 = opts.imageCanal  #number of channels - number of first layer input
    dimInputX = opts.imageDimX     #original image X dimension
    dimInputY = opts.imageDimY     #original image Y dimension
    
    
    for l  in range(len(cnn.layers)):
        layer = cnn.layers[l]   # convolucional layer
        # convolucional layer
        if layer.type == 'c':
            numFiltros2 = layer.numFilters   # number of filters
            dimFiltros = layer.dimFiltros    # filter dimension
            layer.W = 1e-1 * np.ones((dimFiltros,dimFiltros,numFiltros1,numFiltros2)) # initialize w
            layer.b = np.zeros((numFiltros2, 1), order = 'F')  # initialize bias
            layer.W_velocidade = np.zeros((layer.W.shape), order = 'F')
            layer.b_velocidade = np.zeros((layer.b.shape), order = 'F')
            
            # image size after convolution
            dimConvX = int(np.floor((dimInputX - layer.dimFiltros) / layer.strider + 1))
            dimConvY = int(np.floor((dimInputY - layer.dimFiltros) / layer.strider + 1))
            layer.delta = np.zeros((dimConvX,dimConvY,numFiltros2,opts.batchsize), order = 'F')    #Guarda as derivadas para todo conjunto e filtros

            numFiltros1 = numFiltros2  #number of next layer filters
            dimInputX = dimConvX       #next layer image dimension
            dimInputY = dimConvY       #next layer image dimension

        #Pooling layer
        elif layer.type == 'p':
               dimPooledX = int(np.floor((dimInputX - layer.dimPool) / layer.strider) + 1)  #image dimension after pooling
               dimPooledY = int(np.floor((dimInputY - layer.dimPool) / layer.strider) + 1)  #image dimension after pooling
               layer.delta = np.zeros((dimPooledX,dimPooledY,numFiltros1,opts.batchsize), order = 'F')  # stores the derivatives
               dimInputX =  dimPooledX    #next layer image dimension
               dimInputY =  dimPooledY    #next layer image dimension

        #full connected layer
        elif layer.type == 'f':  
            numFiltros2 = layer.numhidden   # number of neurons
            if l > 1:
                if not (cnn.layers[l-1].type == 'f'):
                    numAtrib = numFiltros1 * dimInputX * dimInputY   #number of features
                else:
                    numAtrib = numFiltros1

            else:
                numAtrib = numFiltros1 * dimInputX * dimInputY

            layer.W = 1e-1 * np.random.random((numFiltros2, numAtrib))   # h x ne
            layer.b = np.zeros((numFiltros2,1), order = 'F')  # initialize bias
            layer.W_velocidade = np.zeros((layer.W.shape), order = 'F')
            layer.b_velocidade = np.zeros((layer.b.shape), order = 'F')
            layer.delta = np.zeros((numFiltros2, opts.batchsize), order = 'F')
            numFiltros1 = numFiltros2

        cnn.layers[l] = layer

    # last layer
    if layer.type == 'f': 
        cnn.quantNeuron =  numFiltros1  # number of inputs in output layer
    else:
       cnn.quantNeuron = numFiltros1 * dimInputX * dimInputY

    cnn.cost = 0
    cnn.probs = np.zeros((opts.numClasses, opts.batchsize), order = 'F')
    
    r  = np.sqrt(6) / np.sqrt(opts.numClasses + cnn.quantNeuron+1)
    cnn.Wd = np.random.random((opts.numClasses, cnn.quantNeuron)) * 2 * r - r  # output layer weights
    cnn.bd = np.zeros((opts.numClasses,1), order = 'F')        # output layer bias
    cnn.Wd_velocidade = np.zeros((cnn.Wd.shape), order = 'F')
    cnn.bd_velocidade = np.zeros((cnn.bd.shape), order = 'F')
    cnn.delta = np.zeros((cnn.probs.shape), order = 'F')
    
    cnn.imageDimX = opts.imageDimX      # image X axis dimension
    cnn.imageDimY = opts.imageDimY      # image Y axis dimension
    cnn.imageCanal = opts.imageCanal    # number of channels in image
    cnn.numClasses = opts.numClasses    # number of classes
    cnn.alpha = opts.alpha              # learning rate
    cnn.minibatch = opts.batchsize      # batch size
    cnn.numepochs = opts.numepochs      # number of epochs
    cnn.lambda_ = opts.lambda_          # regularization factor
    cnn.momentum = opts.momentum        # final momentum factor
    cnn.mom = opts.mom                  # initial momentum factor
    cnn.momIncrease = opts.momIncrease  # number of epochs to change momentum
    cnn.ratio = opts.ratio

    return cnn

def train_cnn(cnn,images,labels, pdir = ''):
    """
    Train convolutional neural network.
    
    Args:
        cnn:        cnn object with initial setup and parameters
        images:     array with images to process
        labels:     vector with image labels
        pdir:       directory to save plot if pdir is not empty

    Returns:
        create 'Total Cost vs Epoch' plot and return cnn object with weights
    """
    it = 0   # number of iterations
    plotData = pd.DataFrame(columns=['Epoch', 'Iteration', 'Total Cost', 'Cost'])

    # Load some parameters
    epocasMax = cnn.numepochs           # number of epochs
    minibatch = cnn.minibatch           # minibatch size
    momIncrease = cnn.momIncrease       # number of epochs to increase momentum
    mom = cnn.mom                       # initial momentum factor
    momentum = cnn.momentum             # momentum term
    alpha = cnn.alpha                   # learning rate
    lambda_ = cnn.lambda_               # regularization coefficient
    ratio = cnn.ratio                   # neuron frozen rate
    numlayers = len(cnn.layers)         # number of layers


    N = max(labels.shape)               # number of inputs
    cont = 0
    for nep in range(epocasMax):
        #*****************************************************************
        # 1) For each epoch, freeze some neurons
        #****************************************************************
        for l in range(numlayers):
            layer = cnn.layers[l]
            Idx = []
            # convolution layer
            if layer.type == 'c':
                numFilters = layer.numFilters
                layer.indcongFiltros = np.where(np.random.rand(numFilters) <= ratio)[0] # index of frozen filters
                Idx = layer.indcongFiltros
                layer.txcongFiltros = max(np.shape(Idx)) / numFilters   # rate of frozen filters
                print('In convolution layer %i, %d filters are frozen' % (l,max(np.shape(Idx))))
                cnn.layers[l] = layer
            # pooling layer uses same freezen as convolution layer
            elif layer.type == 'p':
                print('In pooling layer %i, %d filters are frozen' % (l, max(np.shape(Idx))))
            # Freeze neurons in full connected layer
            elif layer.type == 'f':
                numFilters = layer.numhidden
                layer.indcongFiltros = np.where(np.random.rand(numFilters) <= ratio)[0] # index of frozen filters
                Idx = layer.indcongFiltros
                layer.txcongFiltros = max(np.shape(Idx)) / numFilters # rate of frozen filters
                print('In full connected layer %i, %d filters are frozen' % (l,max(np.shape(Idx))))
                cnn.layers[l]=layer

        #******************************************************************
        # Initialize input index
        p = np.random.permutation(N)

        #********************************************************************
        # Split data in minibatch,
        #********************************************************************
        for s in range(0, N - minibatch + 1, minibatch):
            it = it + 1   # count iterations
            # increase momentum
            if it == momIncrease:
                mom = momentum

            # Create training set
            X = images[:,:,:,p[s:s+minibatch]]  # select training input
            Yd = labels[p[s:s+minibatch]]       # select training labels
            Yd = Yd.reshape(len(Yd), 1, order = 'F')
            numImagens = X.shape[3]             # [dimX, dimY, channel, number of images]
            #*************************************************************
            # Feedforward
            #************************************************************
            # activation(dimX x dimY x channels x minibatch)
            # dimX, dimY: tamanho da imagem
            # channels: number of channels
            # minibatch: minibatch size, i.e. number of images
            activation = X # input layer activation
            cont = 0 # image data
            for l in range(numlayers):
                layer = cnn.layers[l]
                # layer.W (dimFiltro x dimFiltro x channels x numFilter)
                # dimFiltro:   filter size (kernel)
                # channels:    number of channels in image
                # numFilter:   number of filters
                # convolution layer
                if layer.type == 'c':
                    strider = layer.strider
                    fativ = layer.fativ
                    activation, dfativ = ff_conv(activation, layer.W, layer.b, strider, fativ)
                    indcong = layer.indcongFiltros  # index of frozen filters
                    txcong = layer.txcongFiltros    # rate of frozen filters
                    activation, dfativ = dropout(activation, dfativ, indcong, txcong)   #next layer activation (freeze neurons)
                # Pooling layer
                elif layer.type == 'p':
                    strider = layer.strider
                    criterio = layer.criterio
                    activation, dfativ = ff_pool(layer.dimPool,activation,strider,criterio)
                    if l>0:
                        layer.numFilters = cnn.layers[l-1].numFilters
                    else:
                        layer.numFilters = cnn.imageCanal

                # full connected layer
                elif layer.type == 'f':
                    fativ = layer.fativ
                    activation, dfativ = ff_full(activation, layer.W, layer.b, fativ)
                    cont = 1 # data has been concatenated

                layer.activation = activation
                layer.dfativ = dfativ
                cnn.layers[l] = layer

            if cont == 0:
                # concatenate previous output for next layers
                activation = np.reshape(activation, (-1, numImagens), order='F')

            # output layer: Softmax
            probs = np.exp(cnn.Wd @ activation + cnn.bd, order = 'F')
            sumProbs = np.sum(probs, 0, keepdims=True)
            probs = probs / sumProbs

            ## --------- Cost Computation ----------
            # Cross-entropy
            logp = np.log(probs)  # number of classes x number of samples
            index = np.arange(probs.shape[1]) * logp.shape[0] + Yd.T
            Custo = -np.sum(logp.ravel(order = 'F')[index]) # same as -Yd.*log(Probs)
            ############################################
            # Sum of weights squared
            wCusto = 0
            for l in range(numlayers):
                layer = cnn.layers[l]
                if not (layer.type == 'p'):
                    wCusto = wCusto + np.sum(layer.W ** 2)

            # Add the sum of weights squared
            wCusto = lambda_ / 2 * (wCusto + np.sum(cnn.Wd ** 2))
            CustoTotal = Custo + wCusto
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #---Backpropagation
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #softmax layer
            output = np.zeros(probs.shape, order = 'F')
            output.ravel(order = 'F')[index] = 1 # Include value ' to desired output
            DeltaSoftmax = (probs - output) # compute the error


            dimSaidaX = cnn.layers[numlayers - 1].activation.shape[0]  # Number of output
            dimSaidaY = cnn.layers[numlayers - 1].activation.shape[1]  #
            layer = cnn.layers[numlayers - 1]     # last layer
            if layer.type == 'f':
                # Use matrix if there's intermediate layer
                delta_ant = cnn.Wd.T @ DeltaSoftmax
            else:
                # no intermediate layer, transform image data for backpropagation
                numFiltros2 = cnn.layers[numlayers - 1].numFilters
                delta_ant = np.reshape(cnn.Wd.T @ DeltaSoftmax,(dimSaidaX,dimSaidaY,numFiltros2,numImagens), order = 'F')


            # load data from output layer, save delta in last layer
            # other layers
            for l in range(numlayers - 1, -1, -1):
                layer = cnn.layers[l]
                if layer.type == 'f':
                    activation = cnn.layers[l].activation
                    delta_full = delta_ant
                    fativ = cnn.layers[l].fativ
                    dfativ = cnn.layers[l].dfativ
                    delta = delta_full(delta_full,layer.W,dfativ,fativ)
                    cnn.layers[l].delta = delta_ant * dfativ  # activation function
                    if l > 0:
                        if cnn.layers[l-1].type == 'c' or cnn.layers[l-1].type == 'p':
                            # previous layer is convolutional
                            # transform image in matrix format
                            dimSaidaX = (cnn.layers[l-1].activation).shape[0] # number of outputs
                            dimSaidaY = (cnn.layers[l-1].activation).shape[1] # number of outpus
                            numFilters = cnn.layers[l-1].numFilters           # number of filters
                            delta_ant = np.reshape(delta,(dimSaidaX,dimSaidaY,numFilters,numImagens), order = 'F')   # previous layer

                        else:
                            delta_ant = delta

                    else: # previous layer is the input
                        dimSaidaX = X.shape[0]   # number of output
                        dimSaidaY = X.shape[1]   #
                        numFilters = cnn.imageCanal # number of filters
                        delta_ant = np.reshape(delta,(dimSaidaX,dimSaidaY,numFilters,numImagens), order = 'F')
                elif layer.type == 'p': # pooling layer
                    if l > 0:
                        numFilters = cnn.layers[l-1].numFilters          # number of filters in previous layer
                        dimOut1 = (cnn.layers[l-1].activation).shape[0]  # output dimension
                        dimOut2 = (cnn.layers[l-1].activation).shape[1]  # output dimension
                    else:
                        numFilters = cnn.imageCanal
                        dimOut1 = X.shape[0]   # output dimension
                        dimOut2 = X.shape[1]   # output dimension

                    strider = cnn.layers[l].strider
                    dimPool = cnn.layers[l].dimPool   # pooling kernel dimension
                    convDim1 = dimOut1    # dimOut1 * strider + dimPool - 1
                    convDim2 = dimOut2    # dimOut2 * strider + dimPool - 1
                    criterio = cnn.layers[l].criterio
                    deltaPool = delta_ant
                    dfativ = cnn.layers[l].dfativ
                    #Unpool da ultima layer
                    delta = np.zeros((convDim1,convDim2,numFilters,numImagens), order = 'F')
                    for imNum in range(numImagens):
                        for FilterNum in range(numFilters):
                            unpool = deltaPool[:,:,FilterNum,imNum]
                            dfativaux = dfativ[:,:,FilterNum,imNum]
                            delta[:,:,FilterNum,imNum]= delta_pool(unpool, dfativaux, dimPool, strider, convDim1, convDim2, criterio)

                    cnn.layers[l].delta = delta
                    delta_ant=delta
                elif layer.type == 'c':
                    if l > 0:
                        numFiltros1 = cnn.layers[l-1].numFilters        # number of filters in previous layer
                        dimOut1 = (cnn.layers[l-1].activation).shape[0] # number of outputs in current layer
                        dimOut2 = (cnn.layers[l-1].activation).shape[1] # number of outputs in current layer
                    else:
                        numFiltros1 = cnn.imageCanal
                        dimOut1 = X.shape[0]   # number of outputs in previous layer
                        dimOut2 = X.shape[1]   # number of outputs in previous layer

                    numFiltros2 = cnn.layers[l].numFilters  # number of filters in next layer
                    delta = np.zeros((dimOut1,dimOut2,numFiltros1,numImagens), order = 'F') # derivative array with zeros
                    deltaConv = delta_ant   # copy next layer derivative
                    Wc = cnn.layers[l].W    # next layer weights
                    strider = cnn.layers[l].strider
                    fativ  = cnn.layers[l].fativ
                    dfativ = cnn.layers[l].dfativ
                    delta_aux = deltaConv
                    for i  in range(numImagens):
                        for f1  in range(numFiltros1):
                            for f2  in range(numFiltros2):
                                # convolution with rotated kernel
                                if fativ == 'sig':  # sigmoid
                                    # sigmoid derivative computed on feedforward, 
                                    # on method ff_conv
                                    df = dfativ
                                elif fativ == 'relu':
                                    df = np.where(dfativ[:, :, f2, i] == f1, 1, 0)

                                delta_aux[:, :, f2, i] = deltaConv[:, :, f2, i] * df   # multiply by activation function derivative
                                delta[:, :, f1, i] = delta[:, :, f1, i] + conv_full(delta_aux[:, :, f2, i], Wc[:, :, f1, f2], dimOut1, dimOut2, strider)

                    cnn.layers[l].delta = delta_aux   #armazena na layer
                    delta_ant = delta

            # gradients
            activation = cnn.layers[numlayers - 1].activation   # last layer activation
            if cnn.layers[numlayers - 1].type == 'c' or cnn.layers[numlayers - 1].type == 'p':
                activation = np.reshape(activation,(-1, numImagens), order = 'F') # tranform in vector

            Wd_grad = DeltaSoftmax @ activation.T   #dJdw
            bd_grad = np.sum(DeltaSoftmax, 1, keepdims = True)     #dIdb

            cnn.Wd_velocidade = mom * cnn.Wd_velocidade + alpha * (Wd_grad / minibatch + lambda_ * cnn.Wd)
            cnn.bd_velocidade = mom * cnn.bd_velocidade + alpha * (bd_grad / minibatch)
            cnn.Wd = cnn.Wd - cnn.Wd_velocidade   # update output layer weights
            cnn.bd = cnn.bd - cnn.bd_velocidade   # update bias

            # update variables
            for l in range(numlayers -1, -1, -1):
                layer = cnn.layers[l]
                if layer.type == 'f':
                    numhidden = layer.numhidden
                    if l == 0:
                        activation = np.reshape(X, (-1, numImagens), order='F')
                    else:
                        activation = cnn.layers[l - 1].activation
                        if not (cnn.layers[l-1].type == 'f') :
                            activation = np.reshape(activation, (-1, numImagens), order='F')
                    Wc_grad = np.zeros((layer.W.shape), order = 'F')
                    bc_grad = np.zeros((layer.b.shape), order = 'F')
                    delta = layer.delta
                    bc_grad = np.sum(delta, axis = 1, keepdims=True)

                    Wc_grad = delta @ activation.T
                    layer.W_velocidade = mom * layer.W_velocidade + alpha * (Wc_grad / numImagens + lambda_ * layer.W)
                    layer.b_velocidade = mom * layer.b_velocidade + alpha * (bc_grad / numImagens)
                    layer.W = layer.W - layer.W_velocidade
                    layer.b = layer.b - layer.b_velocidade

                # convolution layer
                elif(layer.type == 'c'):
                    numFiltros2 = layer.numFilters
                    if l == 0:
                        numFiltros1 = cnn.imageCanal
                        activation = X
                    else:
                        numFiltros1 = cnn.layers[l-1].numFilters
                        activation = cnn.layers[l-1].activation

                    Wc_grad = np.zeros(layer.W.shape, order='F')
                    DeltaConv = layer.delta
                    strider = layer.strider

                    for fil2 in range(numFiltros2):
                        for fil1  in range(numFiltros1):
                            for im  in range(numImagens):
                                Wc_grad[:,:,fil1,fil2] = Wc_grad[:,:,fil1,fil2] + \
                                    conv_comp(activation[:, :, fil1, im], DeltaConv[:, :, fil2, im], strider)

                    bc_grad = np.sum(DeltaConv, axis=(0, 1, 3), keepdims=True).reshape(layer.numFilters, 1)
                    layer.W_velocidade = mom * layer.W_velocidade + alpha * (Wc_grad / numImagens + lambda_ * layer.W)
                    layer.b_velocidade = mom * layer.b_velocidade + alpha * (bc_grad / numImagens)
                    layer.W = layer.W - layer.W_velocidade
                    layer.b = layer.b - layer.b_velocidade
                cnn.layers[l] = layer
            print('Epoch %d: Total cost and cost in iteration  %d is %f %f\n' % (nep, it, CustoTotal, Custo))
            plotData.loc[len(plotData) + 1] = [nep, it, CustoTotal, Custo]
            #break

        #cnnTest(cnn,testimages,testlabels)
        #alpha = alpha/2.0
    #['Epoca', 'Iteration', 'Total Cost', 'Cost'])
    plt.figure()
    plt.plot('Epoch', 'Total Cost', data = plotData)
    plt.xlabel('Epoch')
    plt.ylabel('Total Cost')
    plt.title('Convoluntional Neural Network - CNN' )
    if pdir != '':
        plt.savefig('cnn.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cnn

#=========================================
# Load MNIST dataset
#=========================================
# Return  28x28x[number of MNIST images] matrix containing
# the raw MNIST images
def loadMNISTImages(filename):
    
    fp = open(filename, 'rb')

    assert fp != -1, 'Could not open ' + filename

    magic = np.fromfile(fp, dtype='>u4', count=1)
    assert magic == 2051, 'Bad magic number in ' + filename

    numImages = int(np.fromfile(fp, dtype='>u4', count = 1))
    numRows = int(np.fromfile(fp, dtype='>u4', count = 1))
    numCols = int(np.fromfile(fp, dtype='>u4', count = 1))

    images = np.fromfile(fp, dtype='>u1')
    images = np.reshape(images, (numCols, numRows, numImages), order = 'F')
    images = images.transpose(1, 0, 2)

    fp.close()

    # Reshape to #pixels x #examples
    images = np.reshape(images, (images.shape[0] * images.shape[1], images.shape[2]), order = 'F')
    # Convert to double and rescale to [0,1]
    images = images / 255


    return images
    
def loadMNISTLabels(filename):
    #loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    #the labels for the MNIST images

    fp = open(filename, 'rb')
    assert fp != -1, 'Could not open ' + filename

    magic = np.fromfile(fp, dtype='>u4', count = 1)
    assert magic == 2049, 'Bad magic number in ' + filename

    numLabels = np.fromfile(fp, dtype='>u4', count = 1)
    labels = np.fromfile(fp, dtype='>u1')
    labels = labels.reshape(len(labels),1, order = 'F')

    assert labels.shape[0] == numLabels, 'Mismatch in label count'

    fp.close()

    return labels

#=====================================
# Demo with MNIST dataset
#=====================================
if __name__ == '__main__':

    opts = obj()
    opts.alpha = 1e-1     # learning rate
    opts.batchsize = 50   # training set size = 150
    opts.numepochs = 200  # number of epochs
    opts.imageDimX = 28   # image X axis dimension
    opts.imageDimY = 28   # image Y axis dimension
    opts.imageCanal = 1   # number of channels in input image
    opts.numClasses = 10  # number of classes
    opts.lambda_ = 0.0001 # weight decay factor
    opts.ratio = 0.0      # weight freeze factor
    opts.momentum = 0.95  # final momentum factor
    opts.mom = 0.5        # initial momentum factor
    opts.momIncrease = 20 # number of epochs to change momentum

    # load MNIST - Train
    path = './images/'
    images = loadMNISTImages(path + 'train-images-idx3-ubyte')
    images = np.reshape(images, (opts.imageDimX, opts.imageDimY, 1, -1), order = 'F')
    labels = loadMNISTLabels(path + 'train-labels-idx1-ubyte')
    images = images[:, :, :, 0:500]
    labels = labels[0:500]

    # Load MINST - Test
    testImages = loadMNISTImages(path + 't10k-images-idx3-ubyte')
    testImages = np.reshape(testImages, (opts.imageDimX, opts.imageDimY, 1, -1), order = 'F')
    testLabels = loadMNISTLabels(path + 't10k-labels-idx1-ubyte')

    # layers
    # c ---- convolucional
        #numFilters, strider, dimFiltros, activation function (sig, relu,)
    # p ---- pooling
        # strider, dimPool, Max ou Media
    # f ---- full conect 
        # fativ (sig, relu), numhidden 
        
    cnn = obj()
    cnn.layers = {}
    # convolution layer
    cnn.layers[0] = obj()
    cnn.layers[0].type = 'c'            # c = convolution
    cnn.layers[0].numFilters = 4        # number of filters
    cnn.layers[0].strider = 1           # strider
    cnn.layers[0].dimFiltros = 2        # filter size
    cnn.layers[0].fativ = 'relu'       # sig/relu: activation function
    
    # pooling layer
    cnn.layers[1] = obj()
    cnn.layers[1].type = 'p'            # p = pooling
    cnn.layers[1].strider = 2           # strider
    cnn.layers[1].dimPool = 2           # filter size
    cnn.layers[1].criterio = 'max'      # max/mean

    # convolution layer 
    # cnn.layers[2] = obj()
    # cnn.layers[2].type = 'c'          # c = convolution
    # cnn.layers[2].numFilters = 4      #
    # cnn.layers[2].strider = 1         # strider
    # cnn.layers[2].dimFiltros = 2      # filter size
    # cnn.layers[2].fativ = 'relu'      # sig/relu: activation function
    #
    # pooling layer
    # cnn.layers[3] = obj()
    # cnn.layers[3].type = 'p'          # p = pooling
    # cnn.layers[3].strider = 2         # strider
    # cnn.layers[3].dimPool = 2         # filter size
    # cnn.layers[3].criterio = 'max'    # max/mean

    # full connected layer: last layer setup is optional
    # cnn.layers[2] = obj()              #
    # cnn.layers[2].type = 'f'           # f = full connected
    # cnn.layers[2].fativ = 'sig'        # sig/relu: activation funciton
    # cnn.layers[2].numhidden = 100      # number of neurons in hidden layer
    #

    # initialize CNN parameters
    cnn = init_params(cnn, opts)

    # train
    cnn = train_cnn(cnn, images, labels)

