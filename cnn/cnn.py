#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network


Author: Victor Ivamoto
July, 2020
"""
import numpy as np
from numba import vectorize
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, cuda
from scipy.ndimage import rotate

class obj:
    pass


def sub2ind(array_shape, rows, cols):
    ind = cols * array_shape[0] + rows
    return ind


def calc_delta(delta, W, dfativ, fativ):
    m, N = dfativ.shape
    h, ne = W.shape

    if fativ == 'sig': #função ativação sigmoid
        delta_full = W.T @ (delta * dfativ)

    elif fativ == 'relu':  #funçao ativação relu
        delta_full = np.zeros((ne, N), order = 'F')
        for i in range(ne):
            #dfativ_aux = np.zeros((dfativ.shape), order = 'F')
            #px, py = np.where(dfativ==i)   # [px,py]=find(dfativ==i)
            dfativ_aux = np.where(dfativ == i, 1, 0)
            for j in range(h):
                delta_full[j, :] = delta_full[j, :] + W[j, i] @ (delta * dfativ_aux[j, :])

    return delta_full


def calc_dfativ(fativ, dfativ, numFiltro):
    # Retorna a derivada da função de ativação.
    #
    if fativ == 'sig': #função de ativação sigmoid
        # a derivada da sigmoid foi calculdada no feedforward, na função cnnConv
        df = dfativ
    elif fativ == 'relu':
        #dfativ_aux = np.zeros((dfativ.shape), order = 'F')
        #px, py = np.where(dfativ == numFiltro) # [px,py]=find(dfativ==numFiltro)
        #dfativ_aux[px, py] = 1
        #df = dfativ_aux
        df = np.where(dfativ == numFiltro, 1, 0)
        # np.take_along_axis(result_conv, np.expand_dims(dfativ[:, :, fil2, i], axis=2), axis=2).squeeze(axis=2)
    return df

# @vectorizevoid(float64[:],float64[:])

def cnnConv(images, W, b, strider, fativ):
    # Input:
    # images (dimX x dimY x numCanais x numImagens).
    #         dimX, dimY: tamanho da imagem
    #         numCanais: quantidade de canais
    #         numImagens: quantidade de imagens
    # W (dimFiltroX x dimFiltroY x numCanais x numFiltros)
    filterDim = W.shape[0]
    numFilters1 = W.shape[2]  # numero de canais
    numFilters2 = W.shape[3]
    numImages = images.shape[3]
    imageDimX = images.shape[0]
    imageDimY = images.shape[1]
    convDimX = int(np.floor((imageDimX - filterDim) / strider) + 1)
    convDimY = int(np.floor((imageDimY - filterDim) / strider) + 1)

    dtype = np.float64
    Features = np.zeros((convDimX, convDimY, numFilters2, numImages))  # Armazena as features
    dfativ   = np.zeros((convDimX, convDimY, numFilters2, numImages), dtype = dtype)  #Armazena derivada ativação
    result_conv = np.zeros((convDimX, convDimY, numFilters1 + 1)) 	    #Armazena o resultado da convolução, posição

    for i in range(numImages):
        for fil2  in range(numFilters2):
            convolvedImage = np.zeros((convDimX, convDimY))
            dconvolvedImage = np.zeros((convDimX, convDimY))
            for fil1 in range(numFilters1):
                filter = W[:, :, fil1, fil2]#np.squeeze(W[:, :, fil1, fil2])
                im = images[:, :, fil1, i]#np.squeeze(images[:, :, fil1, i])
                result_conv[:,:,fil1] = conv_mod(im, filter, strider)  #Realiza convolução - não rotaciona a o filtro
            
            if fativ == 'relu':
                #Aplicar bias antes do maximo ou depois não altera o indice
                # max(x1+b,x2+b)
                #[convolvedImage,dconvolvedImage]=max(result_conv,[],3); %Calcula o máximo

#                convolvedImage  = np.max(result_conv[0,:,:,:], axis = 2)  #Calcula o máximo
#                dconvolvedImage = np.max(result_conv[1,:,:,:], axis = 2)
#                dfativ[:, :, fil2, i] = dconvolvedImage - 1 #remap 1 para 0, guarda o indice do valor do maximo
#                convolvedImage = convolvedImage + b[fil2]   #soma o bias

                result_conv[:,:, :-1] = result_conv[:,:, :-1] + b[fil2] # Adiciona bias em todos os canais
                convolvedImage  = np.max(result_conv, axis = 2)  # função de ativação relu
                dfativ[:, :, fil2, i] = np.argmax(result_conv, axis = 2)  # indice dos neuronios ativados
                # np.take_along_axis(result_conv, np.expand_dims(dfativ[:,:,fil2,i], axis=-1), axis=-1).squeeze(axis=-1)
                # np.take_along_axis(result_conv, np.expand_dims(dfativ[:, :, fil2, i], axis=2), axis=2).squeeze(axis=2)
            elif fativ == 'sig':
                convolvedImage = np.sum(result_conv, axis = 2)  #Soma o resultado convolução
                convolvedImage = convolvedImage + b[fil2]       #soma o bias
                convolvedImage = 1 / (1 + np.exp(-convolvedImage))     #função ativação
                dfativ[:, :, fil2, i] = (1 - convolvedImage) * convolvedImage       #derivada sigmoid

            Features[:, :, fil2, i] = convolvedImage

    return Features, dfativ


def cnnfull(ativacao, W, b, fativ):
    m, N, numFiltro, numImagens = ativacao.shape

    Z = np.zeros((W.shape), order = 'F')
    dZ = np.zeros((W.shape), order = 'F')
    Zin = np.zeros((W.shape), order = 'F')
    if numImagens > 1:   #As imagens ainda já foram concatenadas
        #Concatena a saida anterior para as proximas camadas
        ativacao = np.reshape(ativacao,(-1, numImagens), order = 'F')
        N = numImagens

    if fativ == 'sig':
        Zin = W @ ativacao + b #h x N
        ativacao = 1/(1 + np.exp(-Zin, order = 'F'))    #n x N
        dativacao = (1 - ativacao) * ativacao

    elif fativ == 'relu':
        h, ne = W.shape
        for i in range(h):
            for j in range(ne):
                Zin[j, :] = W[i, j] @ ativacao[j, :]
            val, pos = np.maximum(0, Zin, order = 'F')
            Z[i, : ] = val
            dZ[i, :] = pos - 1   #remap 1 para 0
        ativacao = Z
        dativacao = dZ

    return ativacao, dativacao

# Saída do Pooling
def cnnPool(poolDim, convolvedFeatures,strider,criterio):
    # Inputs
    # convolvedFeatures: matriz com imagens após convolução
    # strider
    numImages = convolvedFeatures.shape[3]
    numFilters = convolvedFeatures.shape[2]
    convolvedDimx = convolvedFeatures.shape[0]
    convolvedDimy = convolvedFeatures.shape[1]

    dimx = int(np.floor((convolvedDimx - poolDim) / strider) + 1)   # tamanho da imagem após o pooling
    dimy = int(np.floor((convolvedDimy - poolDim) / strider) + 1)   # tamanho da imagem após o pooling

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
        tamx = np.floor(m1 / m2)  #tamanho do filtro x
        tamy = np.floor(n1 / n2)  #tamanho do filtro y
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
    # mask = np.rot90(mask, k=2)  #Rotaciona o kernel

    m1, n1 = img.shape      # ex: 27x27
    m2, n2 = mask.shape     # ex: 2x2

    tamx = m1 + (2 * m2) - 2 + (strider - 1) * (m1 - 1)  #adciona zeros antes e no meio ex: (29x29) p/ strider = 1
    tamy = n1 + (2 * n2) - 2 + (strider - 1) * (n1 - 1)  #adciona zeros antes e no meio

    img1 = np.zeros((tamx, tamy))

    kx = 0
    for i in range(m2-1, m2 + strider * (m1 - 1), strider):
        ky = 0
        for j in range(n2-1, n2 + strider * (n1 - 1), strider):
            img1[i, j] = img[kx, ky]
            ky = ky + 1
        kx = kx + 1

    strider = 1    #independente do strider de entrada, a convolução sera strider 1
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
# Realiza Convolução, mas não rotaciona os kernels
#*******************************************************************
@jit
def conv_max(img, mask, strider):
    m1, n1 = img.shape
    m2, n2 = mask.shape

    mConv = np.int(np.floor((m1 - m2) / strider) + 1)
    nConv = np.int(np.floor((n1 - n2) / strider) + 1)
    Conv = np.zeros((mConv, nConv))
    dConv = np.zeros((mConv * nConv, 2))    #guarda as posições
    x = 0
    y = 0
    cont = 0
    for i in range(mConv):
        for j in range(nConv):
            img1 = img[x:x + m2, y:y + n2]
            val = np.max(img1)
            px, py = np.where(img1==val)
            Conv[i, j] = val
            dConv[cont, 0] = px[0]  # Guarda os indices do maximo da imagem
            dConv[cont, 1] = py[0]  # no kernel
            y = y + strider
            cont = cont + 1

        x = x + strider
        y = 0

    return Conv, dConv


#***********************************************************************
# Realiza Convolução, mas não rotaciona os kernels
#*******************************************************************
@jit
def conv_mod(img,mask,strider):
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

#    dConv = np.zeros((mConv, nConv), order = 'F')   #não precisa armazenar a derivada de convolução

    return Conv #, dConv

def delta_pool(delta,dConv,dimPool,strider,dimx,dimy,criterio):
    # Derivada dos neuronios do pooling
    m, n = delta.shape
    deltaConv = np.zeros((dimx, dimy), order='F') #tamanho do pooling

    if criterio == 'max':
        p = 0
        for i in range(m):
            for j in range(n):
                #p = n * (i - 1) + j
                px = dConv[p, 0]
                py = dConv[p, 1]
                Ix = int(px + strider * i)
                Iy = int(py + strider * j)
                deltaConv[Ix, Iy] = deltaConv[Ix, Iy] + delta[i, j]
                p = p + 1

    elif criterio == 'mean':
        for i in range(m):
            for j in range(n):
                for kx in range(dimPool):
                    for ky in range(dimPool):
                        Ix = kx + strider * i
                        Iy = ky + strider * j
                        deltaConv[Ix, Iy] = deltaConv[Ix, Iy] + delta[i, j]

    return deltaConv

#*************************************************************
#  y - ativação do neuronio
# dy - derivada da função de ativação do neuronio
# idx - indices dos neuronios congelados
#*************************************************************

def dropout(y, dy, idx, tx):
    # y: ativação ou matriz com imagens
    # dy: derivada da ativação
    # idx: individuos congelados
    # tx: taxa de congelamento
    dimX, dimY, numFilters, numImagens = y.shape
    num = len(idx) - 1
    cont = 0
    for i in range(numFilters):
        if cont <= num:
            if idx[cont] == i: # neuronio foi congelado
#               y[:,:,i,j] = np.zeros((dimX, dimY), order = 'F')  #ativação vai para zero
#               dy[:,:,i,j] = np.zeros((dimX, dimY), order = 'F') #derivada vai para zero
                y[:,:,i,:] = np.zeros((dimX, dimY, numImagens), order = 'F')  #ativação vai para zero
                dy[:,:,i,:] = np.zeros((dimX, dimY, numImagens), order = 'F') #derivada vai para zero
                cont = cont + 1
            else:
#                y[:, :, i, j] = y[:, :, i, j] / (1 - tx)  # aumenta a saida
                y[:,:,i,:] = y[:,:,i,:]  / (1 - tx) # aumenta a saida

        else:
#           y[:,:,i,j] = y[:,:,i,j]  / (1 - tx)  # aumenta a saida
            y[:,:,i,:] = y[:,:,i,:]  / (1 - tx)  # aumenta a saida

    return y, dy

def inicializa_parametros(cnn,opts):
    numFiltros1 = opts.imageCanal  #Numero de canal - numero entrada para primeira camada
    dimInputX = opts.imageDimX     #Dimensão da imagem original X
    dimInputY = opts.imageDimY     #Dimensão da imagem original Y
    
    
    for l  in range(len(cnn.camadas)):
        camada = cnn.camadas[l]   # Camada convolucional
        # Camada convolucional
        if camada.tipo == 'c':
           numFiltros2 = camada.numFiltros   # Numero de filtros
           dimFiltros = camada.dimFiltros    # Dimensão do filtro
           camada.W = 1e-1 * np.ones((dimFiltros,dimFiltros,numFiltros1,numFiltros2)) #Inicializa o w
           camada.b = np.zeros((numFiltros2, 1), order = 'F')  # Inicializa o valor do bias
           camada.W_velocidade = np.zeros((camada.W.shape), order = 'F')
           camada.b_velocidade = np.zeros((camada.b.shape), order = 'F')
                      
           dimConvX = int(np.floor((dimInputX - camada.dimFiltros) / camada.strider + 1))  #Dimensão da imagem gerada apos convolução
           dimConvY = int(np.floor((dimInputY - camada.dimFiltros) / camada.strider + 1))  #Dimensão da imagem gerada apos convolução
           camada.delta = np.zeros((dimConvX,dimConvY,numFiltros2,opts.batchsize), order = 'F')    #Guarda as derivadas para todo conjunto e filtros

           numFiltros1 = numFiltros2  #Numero de filtros da proxima camda (Numrero de entradas)
           dimInputX = dimConvX       #Dimensão da imagem para proxima camada
           dimInputY = dimConvY       #Dimensão da imagem para proxima camada

        #Camada Pooling
        elif camada.tipo == 'p':
           dimPooledX = int(np.floor((dimInputX - camada.dimPool) / camada.strider) + 1)  #Dimensão da Imagem gerada pelo Pooling
           dimPooledY = int(np.floor((dimInputY - camada.dimPool) / camada.strider) + 1)  #Dimensão da Imagem gerada pelo Pooling
           camada.delta = np.zeros((dimPooledX,dimPooledY,numFiltros1,opts.batchsize), order = 'F')  # Guarda as derivadas
           dimInputX =  dimPooledX    #Dimensão da Imagem para proxima camada
           dimInputY =  dimPooledY    #Dimensão da Imagem para proxima camada           

        #Camada full
        elif camada.tipo == 'f':  
            numFiltros2 = camada.numhidden   # Numero de neuronios da camada
            if l > 1:
                if not (cnn.camadas[l-1].tipo == 'f'):
                    numAtrib = numFiltros1 * dimInputX * dimInputY   #Numero de atributos
                else:
                    numAtrib = numFiltros1

            else:
                numAtrib = numFiltros1 * dimInputX * dimInputY

            camada.W = 1e-1 * np.random.random((numFiltros2, numAtrib))   # hxne
            camada.b = np.zeros((numFiltros2,1), order = 'F')  # Inicializa o valor do bias
            camada.W_velocidade = np.zeros((camada.W.shape), order = 'F')
            camada.b_velocidade = np.zeros((camada.b.shape), order = 'F')
            camada.delta = np.zeros((numFiltros2, opts.batchsize), order = 'F')
            numFiltros1 = numFiltros2

        cnn.camadas[l] = camada

    #ultima camada
    if camada.tipo == 'f': 
        cnn.quantNeuron =  numFiltros1  #Numero de entradas camada saida
    else:
       cnn.quantNeuron = numFiltros1 * dimInputX * dimInputY

    cnn.cost = 0
    cnn.probs = np.zeros((opts.numClasses, opts.batchsize), order = 'F')
    
    r  = np.sqrt(6) / np.sqrt(opts.numClasses + cnn.quantNeuron+1)
    cnn.Wd = np.random.random((opts.numClasses, cnn.quantNeuron)) * 2 * r - r  # pesos da camada de saida
    cnn.bd = np.zeros((opts.numClasses,1), order = 'F')        # bias da camada de saida)
    cnn.Wd_velocidade = np.zeros((cnn.Wd.shape), order = 'F')
    cnn.bd_velocidade = np.zeros((cnn.bd.shape), order = 'F')
    cnn.delta = np.zeros((cnn.probs.shape), order = 'F')
    
    cnn.imageDimX = opts.imageDimX
    cnn.imageDimY = opts.imageDimY
    cnn.imageCanal = opts.imageCanal    # Quantidade de canais na imagem
    cnn.numClasses = opts.numClasses    # Numero de classes
    cnn.alpha = opts.alpha              # taxa de aprendizado
    cnn.minibatch = opts.batchsize      # tamanho do batch
    cnn.numepocas = opts.numepocas      # numero de epocas
    cnn.lambda_ = opts.lambda_          # Fator de regularização
    cnn.momentum = opts.momentum
    cnn.mom = opts.mom
    cnn.momIncrease = opts.momIncrease
    cnn.ratio = opts.ratio

    return cnn

def treinamento_cnn(cnn,imagens,labels):
    it = 0   #numero de iterações
    plotData = pd.DataFrame(columns=['Epoch', 'Iteration', 'Total Cost', 'Cost'])

    # Carrega alguns parametros
    epocasMax = cnn.numepocas           # Numero de epocas
    minibatch = cnn.minibatch           # tamanho do minibatch
    momIncrease = cnn.momIncrease       # Incremento do momento
    mom = cnn.mom                       #
    momentum = cnn.momentum             #termo de momento
    alpha = cnn.alpha                   #taxa de aprendizado
    lambda_ = cnn.lambda_               #coeficente de regularização
    ratio = cnn.ratio                   # Taxa de congelamento dos neuronios
    numCamadas = len(cnn.camadas)       #Numero de camadas


    N = max(labels.shape)              #Numero de entradas
    cont = 0
    for nep in range(epocasMax):
        #*****************************************************************
        # 1) Para cada epoca, realiza um congelamento de alguns neuronios
        #****************************************************************
        for l in range(numCamadas):
            camada = cnn.camadas[l]
            Idx = []
            # Camada de convolução
            if camada.tipo == 'c':
                numFiltros = camada.numFiltros
                camada.indcongFiltros = np.where(np.random.rand(numFiltros) <= ratio)[0] #Indice dos filtros congelados
                Idx = camada.indcongFiltros
                camada.txcongFiltros = max(np.shape(Idx)) / numFiltros   #Taxa de filtros congelados
                print('Na camada Convolucao %i, foram congelados efetivamente %d filtros' % (l,max(np.shape(Idx))))
                cnn.camadas[l] = camada
            # Camada pooling segue o mesmo congelamento da camada convolução
            elif camada.tipo == 'p':
                print('Na camada %i, foram congelados efetivamente %d filtros' % (l, max(np.shape(Idx))))
            # Congela neuronios da camada totalmente conectada
            elif camada.tipo == 'f':
                numFiltros = camada.numhidden
                camada.indcongFiltros = np.where(np.random.rand(numFiltros) <= ratio)[0] #Indice dos filtros congelados
                Idx = camada.indcongFiltros
                camada.txcongFiltros = max(np.shape(Idx)) / numFiltros #Taxa de filtros congelados
                print('Na camada Totalmente Conectada %i, foram congelados efetivamente %d filtros' % (l,max(np.shape(Idx))))
                cnn.camadas[l]=camada

        #******************************************************************
        # Gera randomicamente os indices de entrada
        p = np.random.permutation(N)

        #********************************************************************
        # Separa os dados em minibatch,
        #********************************************************************
        for s in range(0, N - minibatch + 1, minibatch):
            it = it + 1   # Conta o numero de atualização
            #incrementa o momento
            if it == momIncrease:
                mom = momentum

            # Gera o subconjunto de treinamen
            X = imagens[:,:,:,p[s:s+minibatch]]  #Seleciona as entrada para treinamento
            Yd = labels[p[s:s+minibatch]]        # seleciona os rotulos para treinamento
            Yd = Yd.reshape(len(Yd), 1, order = 'F')
            numImagens = X.shape[3]                # [dimX, dimY, canal, numero de imagens]
            #*************************************************************
            #Realiza feedforward
            #************************************************************
            # ativacao(dimX x dimY x canais x minibatch)
            # dimX, dimY: tamanho da imagem
            # canais: numero de canais da imagem
            # minibatch: tamanho do minibatch, ou seja quantidade de imagens
            ativacao = X #Ativação para camada entrada
            cont = 0 #sinaliza que os dados são imagens
            for l in range(numCamadas):
                camada = cnn.camadas[l]
                # camada.W (dimFiltro x dimFiltro x canais x numFiltro)
                # dimFiltro: tamanho do filtro (kernel)
                # canais: quantidade de canais da imagem
                # numFiltro: quantidade de filtros
                #Camada convolução
                if camada.tipo == 'c':
                    strider = camada.strider
                    fativ = camada.fativ
                    ativacao, dfativ = cnnConv(ativacao, camada.W, camada.b, strider, fativ)
                    indcong = camada.indcongFiltros  # indice dos filtros  congelados
                    txcong = camada.txcongFiltros    # taxa de filtros congelados
                    ativacao, dfativ = dropout(ativacao, dfativ, indcong, txcong)   #ativação para proxima camada (Congela neurônios)
                # Camada de Pooling
                elif camada.tipo == 'p':
                    strider = camada.strider
                    criterio = camada.criterio
                    ativacao, dfativ = cnnPool(camada.dimPool,ativacao,strider,criterio)
                    if l>0:
                        camada.numFiltros = cnn.camadas[l-1].numFiltros #adicionei este campo para facilita
                    else:
                        camada.numFiltros = cnn.imageCanal #adicionei este campo para facilita

                #Camada totalmente conectadaelif
                elif camada.tipo == 'f':
                    fativ = camada.fativ
                    #save all
                    ativacao, dfativ = cnnfull(ativacao, camada.W, camada.b, fativ)
                    cont = 1 #Sinaliza que os dados foram concatenados

                camada.ativacao = ativacao
                camada.dfativ = dfativ
                cnn.camadas[l] = camada

            if cont == 0:
                #Concatena a saida anterior para as proximas camadas
                ativacao = np.reshape(ativacao, (-1, numImagens), order='F')

            #camada saida: Softmax
            probs = np.exp(cnn.Wd @ ativacao + cnn.bd, order = 'F') #np.exp(W*ativ + b, order = 'F')
            sumProbs = np.sum(probs, 0, keepdims=True)   # calcula a soma das exponenciais
            probs = probs / sumProbs  # np.exp(W*ativ + b)/soma exponencial

            ## --------- Calculo do Custo ----------
            # Calcula a entropia cruzada
            logp = np.log(probs)  #Numero de classes x Numero de amostras
            #index = sub2ind(size(logp), Yd.T, 1:size(probs,2)); #busca o indice que correponde a classe
            index = sub2ind(logp.shape, Yd.T, np.arange(probs.shape[1])) #busca o indice que correponde a classe
            Custo = -np.sum(logp.ravel(order = 'F')[index]) #Equivalente -Yd.*log(Probs)
            #Custo = -Yd.T*np.log(probs)
            ############################################
            # Calcula a soma dos pesos ao quadrado
            wCusto = 0
            for l in range(numCamadas):
                camada = cnn.camadas[l]
                if not (camada.tipo == 'p'):
                    wCusto = wCusto + np.sum(camada.W ** 2)

            # Adiciona a soma dos pesos ao quadrado
            wCusto = lambda_ / 2 * (wCusto + np.sum(cnn.Wd ** 2))
            CustoTotal = Custo + wCusto    # adiciona os pesos na função custo
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #---Realiza o Algoritmo Backpropagation
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #softmax layer
            output = np.zeros(probs.shape, order = 'F')
            output.ravel(order = 'F')[index] = 1 #Aciona o valor de 1 a saida desejada
            DeltaSoftmax = (probs - output) #Calcula o erro


            #Transforma o erro em matriz
            #if numCamadas>1
            #    if not strcmp(cnn.camadas[numCamadas],'f')
            #        numFiltros2 = cnn.camadas[numCamadas].numFiltros; #Numero de filtros da ultima camada
            #    end
            #else
            #    numFiltros2 = size(X,2); #Numero de filtros da ultima camada
            #end

            dimSaidaX = cnn.camadas[numCamadas - 1].ativacao.shape[0]  # Numero de saida
            dimSaidaY = cnn.camadas[numCamadas - 1].ativacao.shape[1]  # Numero de saida
            camada = cnn.camadas[numCamadas - 1]     #ultima camada
            if camada.tipo == 'f':
                # Se tiver camada intermediaria, trabalha com matriz
                delta_ant = cnn.Wd.T @ DeltaSoftmax
            else:
                #Não tem camada intermediaria, transforma os dados em imagem
                #para retropropagar
                numFiltros2 = cnn.camadas[numCamadas - 1].numFiltros
                delta_ant = np.reshape(cnn.Wd.T @ DeltaSoftmax,(dimSaidaX,dimSaidaY,numFiltros2,numImagens), order = 'F')


            # Carrega dados da camada de saida salva delta na ultima camada
            # Carrega dados da camada full salva delta na camada anterior
            # Outras camadas
            for l in range(numCamadas - 1, -1, -1):
                camada = cnn.camadas[l]
                if camada.tipo == 'f':
                    ativacao = cnn.camadas[l].ativacao
                    delta_full = delta_ant
                    fativ = cnn.camadas[l].fativ
                    dfativ = cnn.camadas[l].dfativ
                    delta = calc_delta(delta_full,camada.W,dfativ,fativ)
                    cnn.camadas[l].delta = delta_ant * dfativ  #passa pela função de ativação
                    if l > 0:
                        if cnn.camadas[l-1].tipo == 'c' or cnn.camadas[l-1].tipo == 'p':
                            #camada anterior é convolucional
                            #Precisa transformar forma matricial para imagem
                            dimSaidaX = (cnn.camadas[l-1].ativacao).shape[0]     # Numero de saida
                            dimSaidaY = (cnn.camadas[l-1].ativacao).shape[1]     # Numero de saida
                            numFiltros = cnn.camadas[l-1].numFiltros             #Numero de filtros
                            delta_ant = np.reshape(delta,(dimSaidaX,dimSaidaY,numFiltros,numImagens), order = 'F')   #camada anterior

                        else:
                            delta_ant = delta

                    else: #camada anterior é entrada
                        dimSaidaX = X.shape[0]   # Numero de saida
                        dimSaidaY = X.shape[1]   # Numero de saida
                        numFiltros = cnn.imageCanal #Numero de filtros
                        delta_ant = np.reshape(delta,(dimSaidaX,dimSaidaY,numFiltros,numImagens), order = 'F')
                elif camada.tipo == 'p': # camada pooling
                    if l > 0:
                        numFiltros = cnn.camadas[l-1].numFiltros            #Numero de filtros da camada anterior
                        dimSaida1 = (cnn.camadas[l-1].ativacao).shape[0]      #dimensao da saida
                        dimSaida2 = (cnn.camadas[l-1].ativacao).shape[1]      #dimensao da saida
                    else:
                        numFiltros = cnn.imageCanal
                        dimSaida1 = X.shape[0]   #dimensao da saida
                        dimSaida2 = X.shape[1]   #dimensao da saida

                    strider = cnn.camadas[l].strider
                    dimPool = cnn.camadas[l].dimPool   #dimensão do kernel de pooling
                    convDim1 = dimSaida1    # dimSaida1 * strider + dimPool - 1
                    convDim2 = dimSaida2    # dimSaida2 * strider + dimPool - 1
                    criterio = cnn.camadas[l].criterio
                    deltaPool = delta_ant
                    dfativ = cnn.camadas[l].dfativ
                    #Unpool da ultima camada
                    delta = np.zeros((convDim1,convDim2,numFiltros,numImagens), order = 'F')    #Cria
                    for imNum in range(numImagens):
                        for FilterNum in range(numFiltros):
                            unpool = deltaPool[:,:,FilterNum,imNum]
                            dfativaux = dfativ[:,:,FilterNum,imNum]
                            delta[:,:,FilterNum,imNum]= delta_pool(unpool, dfativaux, dimPool, strider, convDim1, convDim2, criterio)

                    cnn.camadas[l].delta = delta
                    delta_ant=delta
                elif camada.tipo == 'c':
                    if l > 0:
                        numFiltros1 = cnn.camadas[l-1].numFiltros       #Numero de filtros da camada anterior
                        dimSaida1 = (cnn.camadas[l-1].ativacao).shape[0]   #Numero de saida da camada atual
                        dimSaida2 = (cnn.camadas[l-1].ativacao).shape[1]   #Numero de saida da camada atual
                    else:
                        numFiltros1 = cnn.imageCanal
                        dimSaida1 = X.shape[0]   #Numero de saida da camada anterior
                        dimSaida2 = X.shape[1]   #Numero de saida da camada anterior

                    numFiltros2 = cnn.camadas[l].numFiltros     #Numero de filtrso da camada posterior
                    delta = np.zeros((dimSaida1,dimSaida2,numFiltros1,numImagens), order = 'F') # Matriz de derivada com zero
                    deltaConv = delta_ant    #copia a derivada da camada da frente
                    Wc = cnn.camadas[l].W    #pesos da camada posterior
                    strider = cnn.camadas[l].strider
                    fativ  = cnn.camadas[l].fativ
                    dfativ = cnn.camadas[l].dfativ
                    delta_aux = deltaConv
                    for i  in range(numImagens):
                        for f1  in range(numFiltros1):
                            for f2  in range(numFiltros2):
                                #Precisa fazer convolução full com kernel
                                #rotacionado
                                df = calc_dfativ(fativ, dfativ[:, :, f2, i], f1)
                                delta_aux[:, :, f2, i] = deltaConv[:, :, f2, i] * df   #Multiplica pela derivada da função ativação
                                delta[:, :, f1, i] = delta[:, :, f1, i] + conv_full(delta_aux[:, :, f2, i], Wc[:, :, f1, f2], dimSaida1, dimSaida2, strider)

                    cnn.camadas[l].delta = delta_aux   #armazena na camada
                    delta_ant = delta

            # gradients
            ativacao = cnn.camadas[numCamadas - 1].ativacao   #ativacao da ultima camada
            #ativacao = reshape(ativacao,[],numImagens, order = 'F')   # Transforma em vetor
            if cnn.camadas[numCamadas - 1].tipo == 'c' or cnn.camadas[numCamadas - 1].tipo == 'p':
                ativacao = np.reshape(ativacao,(-1, numImagens), order = 'F') # Transforma em vetor

            Wd_grad = DeltaSoftmax @ ativacao.T   #dJdw
            bd_grad = np.sum(DeltaSoftmax, 1, keepdims = True)     #dIdb

            cnn.Wd_velocidade = mom * cnn.Wd_velocidade + alpha * (Wd_grad / minibatch + lambda_ * cnn.Wd)
            cnn.bd_velocidade = mom * cnn.bd_velocidade + alpha * (bd_grad / minibatch)
            cnn.Wd = cnn.Wd - cnn.Wd_velocidade   #atualiza os pesos da camada de saida
            cnn.bd = cnn.bd - cnn.bd_velocidade   #atualiza o bias

            #Realiza a atualização
            for l in range(numCamadas -1, -1, -1):
                camada = cnn.camadas[l]
                if camada.tipo == 'f':
                    numhidden = camada.numhidden
                    if l == 0:
                        ativacao = np.reshape(X, (-1, numImagens), order='F')
                    else:
                        ativacao = cnn.camadas[l - 1].ativacao
                        if not (cnn.camadas[l-1].tipo == 'f') :
                            ativacao = np.reshape(ativacao, (-1, numImagens), order='F')
                    Wc_grad = np.zeros((camada.W.shape), order = 'F')
                    bc_grad = np.zeros((camada.b.shape), order = 'F')
                    delta = camada.delta
                    bc_grad = np.sum(delta, axis = 1, keepdims=True)

                    Wc_grad = delta @ ativacao.T
                    camada.W_velocidade = mom * camada.W_velocidade + alpha * (Wc_grad / numImagens + lambda_ * camada.W)
                    camada.b_velocidade = mom * camada.b_velocidade + alpha * (bc_grad / numImagens)
                    camada.W = camada.W - camada.W_velocidade
                    camada.b = camada.b - camada.b_velocidade

                # Camada de convolução
                elif(camada.tipo == 'c'):
                    numFiltros2 = camada.numFiltros
                    if l == 0:
                        numFiltros1 = cnn.imageCanal
                        ativacao = X
                    else:
                        numFiltros1 = cnn.camadas[l-1].numFiltros
                        ativacao = cnn.camadas[l-1].ativacao

                    Wc_grad = np.zeros(camada.W.shape, order='F')
                    DeltaConv = camada.delta
                    strider = camada.strider

                    for fil2 in range(numFiltros2):
                        for fil1  in range(numFiltros1):
                            for im  in range(numImagens):
                                Wc_grad[:,:,fil1,fil2] = Wc_grad[:,:,fil1,fil2] + conv_comp(ativacao[:, :, fil1, im], DeltaConv[:, :, fil2, im], strider)

                    bc_grad = np.sum(DeltaConv, axis=(0, 1, 3), keepdims=True).reshape(camada.numFiltros, 1)
                    camada.W_velocidade = mom * camada.W_velocidade + alpha * (Wc_grad / numImagens + lambda_ * camada.W)
                    camada.b_velocidade = mom * camada.b_velocidade + alpha * (bc_grad / numImagens)
                    camada.W = camada.W - camada.W_velocidade
                    camada.b = camada.b - camada.b_velocidade
                cnn.camadas[l] = camada
            print('Epoca %d: Custo Total e Custo na iteracao %d is %f %f\n' % (nep, it, CustoTotal, Custo))
            plotData.loc[len(plotData) + 1] = [nep, it, CustoTotal, Custo]
            #break

        #cnnTest(cnn,testimages,testlabels)
       # alpha = alpha/2.0
    #['Epoca', 'Iteration', 'Total Cost', 'Cost'])
    plt.figure()
    plt.plot('Epoch', 'Total Cost', data = plotData)
    plt.xlabel('Epoch')
    plt.ylabel('Total Cost')
    plt.title('Convoluntional Neural Network - CNN' )
    plt.savefig('cnn.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cnn


# Return  28x28x[number of MNIST images] matrix containing
# the raw MNIST images

def loadMNISTImages(filename):
    
    #fp = fopen(filename, 'rb')
    fp = open(filename, 'rb')
#    np.fromfile(fp, dtype='>u4', count = 1)

    assert fp != -1, 'Could not open ' + filename


    magic = np.fromfile(fp, dtype='>u4', count=1)
    #magic = fread(fp, 1, 'int32', 0, 'ieee-be')
    assert magic == 2051, 'Bad magic number in ' + filename

    #numImages = fread(fp, 1, 'int32', 0, 'ieee-be')
    #numRows = fread(fp, 1, 'int32', 0, 'ieee-be')
    #numCols = fread(fp, 1, 'int32', 0, 'ieee-be')

    numImages = int(np.fromfile(fp, dtype='>u4', count = 1))
    numRows = int(np.fromfile(fp, dtype='>u4', count = 1))
    numCols = int(np.fromfile(fp, dtype='>u4', count = 1))

#    images = fread(fp, inf, 'unsigned char')
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

#    magic = fread(fp, 1, 'int32', 0, 'ieee-be')
    magic = np.fromfile(fp, dtype='>u4', count = 1)
    assert magic == 2049, 'Bad magic number in ' + filename

#    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be')
    numLabels = np.fromfile(fp, dtype='>u4', count = 1)

    #    labels = fread(fp, inf, 'unsigned char')
    labels = np.fromfile(fp, dtype='>u1')

    labels = labels.reshape(len(labels),1, order = 'F')

    assert labels.shape[0] == numLabels, 'Mismatch in label count'

    fp.close()

    return labels
if __name__ == '__main__':
    opts = obj()
    opts.alpha = 1e-1     # taxa  de aprendizado
    opts.batchsize = 50   # tamanho do conjunto de treinamento 150
    opts.numepocas = 200  # Numero de epocas
    opts.imageDimX = 28   # Dimensão do eixo X da imagem
    opts.imageDimY = 28   # Dimensão do eixo X da imagem
    opts.imageCanal = 1   # Quantidade de canais da imagem de entrada
    opts.numClasses = 10  # Numero de classes
    opts.lambda_ = 0.0001 # fator de decaimento dos pesos
    opts.ratio = 0.0      # fator de congelamento dos pesos
    opts.momentum = 0.95  # fator do momento
    opts.mom = 0.5        # Altera momento
    opts.momIncrease = 20 # Numero de epocas para incrementar momento

    # Carrega base de dados MINST Treinamento 
    #addpath .\imagens\
    path = '../../../Convolutional-Neural-Networks---Matlab-master/'
    images = loadMNISTImages(path + 'imagens/train-images-idx3-ubyte')
    images = np.reshape(images, (opts.imageDimX, opts.imageDimY, 1, -1), order = 'F')
    labels = loadMNISTLabels(path + 'imagens/train-labels-idx1-ubyte')
    # labels[labels == 0] = 10 # Remap 0 to 10
    images = images[:, :, :, 0:500]
    labels = labels[0:500]


    # Carrega base de dados MINST Teste
    testImages = loadMNISTImages(path + 'imagens/t10k-images-idx3-ubyte')
    testImages = np.reshape(testImages, (opts.imageDimX, opts.imageDimY, 1, -1), order = 'F')
    testLabels = loadMNISTLabels(path + 'imagens/t10k-labels-idx1-ubyte')
    # testLabels[testLabels == 0] = 10 # Remap 0 to 10

    # Camadas
    # c ---- convolucional
        #numfiltros, strider, dimFiltros, função ativação (sig, relu,)
    # p ---- pooling
        # strider, dimPool, Max ou Media
    # f ---- full conect 
        # fativ (sig, relu), numhidden 
        

#    cnnfull.camadas = {
    #    struct('tipo', 'c', 'numFiltros', 6,'strider',2,'dimFiltros', 2,'fativ','relu')   #camada convolução
    #    struct('tipo', 'p', 'strider',1,'dimPool', 2,'criterio','max')                    #camada subamostragem
    #	struct('tipo', 'c', 'numFiltros',8, 'strider',1, 'dimFiltros', 2, 'fativ', 'relu') #camada convolução
    #	struct('tipo', 'p', 'strider', 2, 'dimPool', 2, 'criterio', 'max')                 #camada subamostragem
     #   struct('tipo', 'f','fativ','sig','numhidden',100)                                 #camada totalmente conectada 
     #   struct('tipo', 'f','fativ','sig','numhidden',50)
#    }

    cnn = obj()
    cnn.camadas = {}
    #camada convolução
    cnn.camadas[0] = obj()
    cnn.camadas[0].tipo = 'c'           # c = convolucao
    cnn.camadas[0].numFiltros = 4       #
    cnn.camadas[0].strider = 1          # strider
    cnn.camadas[0].dimFiltros = 2       # tamanho do filtro
    cnn.camadas[0].fativ = 'relu'       # sig/relu: função de ativação.
    
    #camada subamostragem
    cnn.camadas[1] = obj()
    cnn.camadas[1].tipo = 'p'           # p = pooling
    cnn.camadas[1].strider = 2          # strider
    cnn.camadas[1].dimPool = 2          # tamanho do filtro
    cnn.camadas[1].criterio = 'max'     # max/mean
    # camada convolução
    # cnn.camadas[2] = obj()
    # cnn.camadas[2].tipo = 'c'  # c = convolucao
    # cnn.camadas[2].numFiltros = 4  #
    # cnn.camadas[2].strider = 1  # strider
    # cnn.camadas[2].dimFiltros = 2  # tamanho do filtro
    # cnn.camadas[2].fativ = 'relu'  # sig/relu: função de ativação.
    #
    # # camada subamostragem
    # cnn.camadas[3] = obj()
    # cnn.camadas[3].tipo = 'p'  # p = pooling
    # cnn.camadas[3].strider = 2  # strider
    # cnn.camadas[3].dimPool = 2  # tamanho do filtro
    # cnn.camadas[3].criterio = 'max'  # max/mean

    #camada totalmente conectada 
    # cnn.camadas[2] = obj()              #
    # cnn.camadas[2].tipo = 'f'           # f = full connected
    # cnn.camadas[2].fativ = 'sig'        # sig/relu: função de ativação
    # cnn.camadas[2].numhidden = 100      # numero de neuronios na camada escondida
    #

    #Inicializa parâmetros da CNN
    cnn = inicializa_parametros(cnn, opts)

    cnn = treinamento_cnn(cnn, images, labels)

    # predict(cnn, testImages, testLabels)
