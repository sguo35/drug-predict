from keras.models import load_model

from load_data import load_data
from load_data_test import load_data_test
from sklearn.metrics import roc_auc_score

import numpy as np

inference_model = load_model('./fine_tune_model.h5')

#x, y = load_data()
x_t, y_t = load_data_test()

# split into test and train
x_test = np.array(x_t)
y_test = np.array(y_t)

#x_train = np.array(x)
#y_train = np.array(y)

truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0
truePositive9 = 0
falsePositive9 = 0


print("Test data set")
test_results = inference_model.predict(x_test)

auc_result = []
auc_actual = []

for index, entry in enumerate(test_results):
    if entry[0] > 0.5 and y_test[index][0] == 1:
        truePositive += 1
    if entry[0] > 0.99 and y_test[index][0] == 1:
        truePositive9 += 1
    if entry[1] > 0.5 and y_test[index][1] == 1:
        trueNegative += 1
    if entry[0] > 0.5 and y_test[index][0] != 1:
        falsePositive += 1
    if entry[0] > 0.99 and y_test[index][0] != 1:
        falsePositive9 += 1
    if entry[1] > 0.5 and y_test[index][1] != 1:
        truePositive += 1
    auc_result.append(entry[0])
    auc_actual.append(y_test[index][0] == 1)

print('True Positive')
print(truePositive)
print('True Negative')
print(trueNegative)
print('False Positive')
print(falsePositive)
print('False Negative')
print(falseNegative)
print('True Positive >99%')
print(truePositive9)
print('False Positive >99%')
print(falsePositive9)

print('ACC')
print((truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive))

print('AUC')
print(roc_auc_score(auc_actual, auc_result))
"""
truePositive = 0
trueNegative = 0
falseNegative = 0
falsePositive = 0
truePositive9 = 0
falsePositive9 = 0

print("Train dataset")

auc_result = []
auc_actual = []
train_results = inference_model.predict(x_train)


for index, entry in enumerate(train_results):
    if entry[0] > 0.5 and y_train[index][0] == 1:
        truePositive += 1
    if entry[0] > 0.99 and y_train[index][0] == 1:
        truePositive9 += 1
    if entry[1] > 0.5 and y_train[index][1] == 1:
        trueNegative += 1
    if entry[0] > 0.5 and y_train[index][0] != 1:
        falsePositive += 1
    if entry[0] > 0.99 and y_train[index][0] != 1:
        falsePositive9 += 1
    if entry[1] > 0.5 and y_train[index][1] != 1:
        truePositive += 1
    auc_result.append(entry[0])
    auc_actual.append(y_train[index][0] == 1)

print('True Positive')
print(truePositive)
print('True Negative')
print(trueNegative)
print('False Positive')
print(falsePositive)
print('False Negative')
print(falseNegative)
print('True Positive >99%')
print(truePositive9)
print('False Positive >99%')
print(falsePositive9)

print('ACC')
print((truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive))

print('AUC')
print(roc_auc_score(auc_actual, auc_result))
"""