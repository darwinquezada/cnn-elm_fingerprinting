import keras.backend as K
import tensorflow as tf
import numpy as np
import random
from numpy.random import seed, default_rng

from miscellaneous.misc import Misc


'''
Based on:

#  ELM MODULE
#================================================================================
#  Due to limited GPU RAM - input size should be smaller than 5000 on Kaggle
#  or arrange to have less input samples (e.g. 10000 input samples)
#  On other platforms it depends on the available resources.
#  For larger inputs size is better to use the latest cell (implementing a
#  typical multilayer perceptron  MLP in Keras)
# Copyright Radu & Ioana DOGARU - radu_d@ieee.org
# More details in paper
# R. Dogaru and Ioana Dogaru, "BCONV-ELM: Binary Weights Convolutional
# Neural Network Simulator based on Keras/Tensorflow for Low Complexity
# Implementations", in Proceedings ISEEE 2019, in Press.
# Please cite the above paper if you find this code useful
#
#--------------------------------------------------------------------------
'''
misc = Misc()
# implements the ELM training procedure with weight quantization

# For reproducibility
rnd_seed = 11
random.seed(rnd_seed)
seed(rnd_seed)
default_rng(rnd_seed)
tf.random.set_seed(
    rnd_seed
)

def elmTrain_fix(X, Y, h_Neurons, C, act_function, ni):
    # Training phase - emulated fixed point precision (ni bit quantization)
    # X - Samples (feature vectors) Y - Labels
    # ni - number of bits to quantize the inW weights
    Ntr = np.size(X, 1)
    in_Neurons = np.size(X, 0)

    targets = Y

    #   Generare inW
    #   Generate inW layer
    #   Takes care if h_Neurons==0
    if h_Neurons == 0:
        inW = np.eye(in_Neurons)
        h_Neurons = in_Neurons
    else:
        inW = -1 + 2 * np.random.rand(h_Neurons, in_Neurons).astype('float32')
        if ni > 0:
            Qi = -1 + pow(2, ni - 1)
            inW = np.round(inW * Qi)

    #  Compute hidden layer
    iw_ = K.variable(inW)
    x_ = K.variable(X)
    h_ = misc.activation_function(K.dot(iw_, x_), act_function)
    # ------------------------------------
    # Moore - Penrose computation of output weights (outW) layer
    ta_ = K.variable(targets)
    if h_Neurons < Ntr:
        outw_ = tf.linalg.solve(K.eye(h_Neurons) / C + K.dot(h_, K.transpose(h_)), K.dot(h_, K.transpose(ta_))) 
    else:
        outw_ = K.dot(h_, tf.linalg.solve(K.eye(Ntr) / C + K.dot(K.transpose(h_), h_), K.transpose(ta_))) 

    outW = K.elu(outw_)
    K.clear_session()
    return inW, outW, h_


def elmPredict_optim(X, inW, outW, act_function):
    # implements the ELM predictor given the model as arguments
    # model is simply given by inW, outW and type
    # returns a score matrix (winner class has the maximal score)
    x_ = K.variable(X)
    iw_ = K.variable(inW)
    ow_ = K.variable(outW)

    h_ = misc.activation_function(K.dot(iw_, x_), act_function)
    mul1 = K.dot(K.transpose(h_), ow_)
    sc_ = K.transpose(mul1)
    score = K.elu(sc_)
    K.clear_session()
    return score, h_