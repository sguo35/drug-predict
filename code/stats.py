from keras.models import load_model

from load_data import load_data
from load_data_test import load_data_test

import numpy as np

inference_model = load_model('./fine_tune_model.h5')

x, y = load_data()
x_t, y_t = load_data_test()

# split into test and train
x_test = np.array(x_t)
y_test = np.array(y_t)

x_train = np.array(x)
y_train = np.array(y)

truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0


# train_results = inference_model.predict(x_train)

test_results = inference_model.predict(x_test)



for index, entry in enumerate(test_results):
    if entry[0] > 0.5 and y_test[index][0] == 1:
        truePositive += 1
    if entry[1] > 0.5 and y_test[index][1] == 1:
        trueNegative += 1
    if entry[0] > 0.5 and y_test[index][0] != 1:
        falsePositive += 1
    if entry[1] > 0.5 and y_test[index][1] != 1:
        truePositive += 1

print('True Positive')
print(truePositive)
print('True Negative')
print(trueNegative)
print('False Positive')
print(falsePositive)
print('False Negative')
print(falseNegative)

print('ACC')
print((truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive))

truePositive = 0
trueNegative = 0
falseNegative = 0
falsePositive = 0

train_results = inference_model.predict(x_train)



for index, entry in enumerate(train_results):
    if entry[0] > 0.5 and y_train[index][0] == 1:
        truePositive += 1
    if entry[1] > 0.5 and y_train[index][1] == 1:
        trueNegative += 1
    if entry[0] > 0.5 and y_train[index][0] != 1:
        falsePositive += 1
    if entry[1] > 0.5 and y_train[index][1] != 1:
        truePositive += 1

print('True Positive')
print(truePositive)
print('True Negative')
print(trueNegative)
print('False Positive')
print(falsePositive)
print('False Negative')
print(falseNegative)

print('ACC')
print((truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive))