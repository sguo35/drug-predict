from __future__ import division

import numpy as np
np.random.seed(1234) # set our RNG seed
rng_seed = np.random.randint(2**30)

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD

from load_data import load_data
from load_data_test import load_data_test

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import RBM
from keras_extensions.dbn import DBN
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm

# model parameters
input_dim = 10468
hidden_dim = [400, 400, 400, 400]

dropouts = [0.0, 0.0, 0.0, 0.0]

batch_size = 128
# num epochs to train for each hidden layer
num_epoch = [50, 50, 50, 50]
num_epochs_SGD = 100
nb_gibbs_steps = 1

lr = 0.1

@log_to_file('predict_dbn.log')
def main():
    x, y = load_data()
    x_t, y_t = load_data_test()

    # split into test and train
    x_train = np.array(x)
    x_test = np.array(x_t)
    
    y_train = np.array(y)
    y_test = np.array(y_t)


    # setup model
    print('Creating training model...')

    # input layer
    rbm1 = RBM(
        hidden_dim[0], 
        input_dim=input_dim, 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=False,
        batch_size=batch_size,
        dropout=0.3
    )

    # hidden layer 1
    rbm2 = RBM(
        hidden_dim[1], 
        input_dim=hidden_dim[0], 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=False,
        batch_size=batch_size,
        dropout=0.3
    )

    # hidden layer 2
    rbm3 = RBM(
        hidden_dim[2], 
        input_dim=hidden_dim[1], 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=False,
        batch_size=batch_size,
        dropout=0.3
    )

    # hidden layer 3
    rbm4 = RBM(
        hidden_dim[3], 
        input_dim=hidden_dim[2], 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=False,
        batch_size=batch_size,
        dropout=0.3
    )

    rbms = [rbm1, rbm2, rbm3]
    dbn = DBN(rbms)

    # setup optimizer, loss
    def get_layer_loss(rbm, layer_no):
        return rbm.contrastive_divergence_loss
    def get_layer_optimizer(layer_no):
        return SGD(lr, 0., decay=0.0, nesterov=True)

    metrics=[]
    for rbm in rbms:
        metrics.append([rbm.reconstruction_loss])

    dbn.compile(layer_optimizer=get_layer_optimizer, layer_loss=get_layer_loss, metrics=metrics)


    # pretrain greedily
    print('Training...')
    dbn.fit(x_train, batch_size, num_epoch, verbose=1, shuffle=True)

    # create the Keras inference model
    print('Creating Keras inference model...')
    Flayers = dbn.get_forward_inference_layers()

    inference_model = Sequential()
    for layer in Flayers:
        inference_model.add(layer)
    # final layer takes in RBM 2k inputs and outputs + and - probabilities
    inference_model.add(Dense(2, input_dim=hidden_dim[3], activation='softmax'))

    inference_model.save('./model.h5')


    # inference_model = load_model('./model.h5')
    print('Finetuning parameters via SGD...')
    print(inference_model.summary())
    inference_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    inference_model.fit(x_train, y_train,
                batch_size=batch_size,
                nb_epoch=num_epochs_SGD,
                verbose=1,
                validation_data=(x_test, y_test))

    inference_model.save('./fine_tune_model.h5')
    print('Final model saved!')

if __name__ == '__main__':
    main()
