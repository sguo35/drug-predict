from __future__ import division

import numpy as np
np.random.seed(1234) # set our RNG seed
rng_seed = np.random.randint(2**30)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import RBM
from keras_extensions.dbn import DBN
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm

# model parameters
input_dim = 14564
hidden_dim = [2000, 2000, 2000, 2000]

dropouts = [0.0, 0.0, 0.0, 0.0]

batch_size = 128
# num epochs to train for each hidden layer
num_epoch = [100, 100, 100, 100]
num_epochs_SGD = 1000
nb_gibbs_steps = 10

lr = 0.01

@log_to_file('predict_dbn.log')
def main():
    # initialize our dataset
    dataset = [0]

    # split into test and train
    x_train
    x_test

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
        persistent=True,
        batch_size=batch_size,
        dropouts=dropouts[0]
    )

    # hidden layer 1
    rbm2 = RBM(
        hidden_dim[1], 
        input_dim=hidden_dim[0], 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=True,
        batch_size=batch_size,
        dropouts=dropouts[1]
    )

    # hidden layer 2
    rbm3 = RBM(
        hidden_dim[2], 
        input_dim=hidden_dim[1], 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=True,
        batch_size=batch_size,
        dropouts=dropouts[1]
    )

    # hidden layer 3
    rbm4 = RBM(
        hidden_dim[3], 
        input_dim=hidden_dim[2], 
        init=glorot_uniform_sigm,
        visible_unit_type='binary',
        hidden_unit_type='binary',
        nb_gibbs_steps=nb_gibbs_steps,
        persistent=True,
        batch_size=batch_size,
        dropouts=dropouts[1]
    )

    rbms = [rbm1, rbm2, rbm3, rbm4]
    dbn = DBN(rbms)

    # setup optimizer, loss
    def get_layer_loss(rbm, layer_no):
        return rbm.contrastive_divergence_loss
    def get_layer_optimizer(layer_no):
        return SGD((layer_no + 1) * lr, 0., decay=0.0, nesterov=False)

    metrics=[]
    for rbm in rbms:
        metrics.append([rbm.reconstruction_loss])

    dbn.compile(layer_optimizer=get_layer_optimizer, layer_loss=get_layer_loss, metrics=metrics)


    # pretrain greedily
    print('Training...')
    dbn.fit(X_train, batch_size, num_epoch, verbose=1, shuffle=False)

    # create the Keras inference model
    print('Creating Keras inference model...')
    Flayers = dbn.get_forward_inference_layers()
    Blayers = dbn.get_backward_inference_layers()

    inference_model = Sequential()
    for layer in Flayers:
        inference_model.add(layer)
        inference_model.add(SampleBernoulli(mode='random'))

    for layer in Blayers[:-1]:
        inference_model.add(layer)
        inference_model.add(SampleBernoulli(mode='random'))
    # final layer takes in RBM 2k inputs and outputs + and - probabilities
    inference_model.add(Dense(2, input_dim=hidden_dim[3]))

    print('Finetuning parameters via SGD...')
    opt = SGD()
    inference_model.compile(opt, loss='binary_crossentropy')
    h = inference_model.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=num_epochs_SGD,
                verbose=1,
                validation_data=(X_test, Y_test))

    print('Doing inference...')
    h = inference_model.predict(dataset)

    print(h)

if __name__ == '__main__':
    main()
