from __future__ import division

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers

from load_data import load_data
from load_data_test import load_data_test

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(1337)  # for reproducibility

from dbn import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

# Loading dataset
x, y = load_data()
X = np.asarray(x, dtype='float32')
Y = np.asarray(y, dtype='float32')

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[2000, 2000, 2000, 2000],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2,
                                         verbose=True)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

if __name__ == '__main__':
    main()
