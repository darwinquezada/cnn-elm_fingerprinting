import keras.backend as K
import tensorflow as tf
import random
import numpy as np
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
# For reproducibility
rnd_seed = 11
seed(rnd_seed)
random.seed(rnd_seed)
default_rng(rnd_seed)
tf.random.set_seed(
    rnd_seed
)

def pool1d(x, pool_size, strides=1, padding='valid', data_format=None, pool_mode='max'):
    x = K.expand_dims(x, 1)
    x = K.pool2d(x, pool_size=(1, pool_size), strides=(1, strides), padding=padding, data_format=data_format,
                 pool_mode=pool_mode)
    return x[:, 0]

# definitie primul strat de convolutii (liniare)
def convlayer(input_data=None, cnn_config=None):
    # inlay este intrarea (multicanal si organizata [samples, x_size, y_size, channells])
    # ker_ este variabila kernel organizata in forma [x_size, x_sixe, channels, filtre ]
    misc = Misc()

    if input_data is None:
        misc.log_msg("ERROR", "Please, specify the input data.")
        exit(-1)

    if cnn_config is None:
        misc.log_msg("ERROR", "Please, specify the parameters of the CNN.")
        exit(-1)

    inp_chan = np.shape(input_data)[2]

    ker = np.sign(np.random.rand(cnn_config['kernel_size'], inp_chan, cnn_config['filter']).astype('float32'))
    # ker = np.random.rand(cnn_config['kernel_size'], inp_chan, cnn_config['filter']).astype('float32') - 0.5
    ker = 1*(ker)/(cnn_config['kernel_size']) #scaling
    ker = K.variable(ker)

    if cnn_config['type'] == 1:
        out_ = K.conv1d(input_data, ker, strides=cnn_config['strides'], padding=cnn_config['padding'],
                        data_format=cnn_config['data_format'])

    mp = pool1d(out_, 2, strides=2, padding='valid', data_format='channels_last', pool_mode='avg')
    nout__ = misc.activation_function(mp, cnn_config['act_funct'])
    out_ = K.batch_flatten(nout__)  # flatten all ELM
    return out_
