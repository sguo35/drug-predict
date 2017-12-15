from keras.models import load_model

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

p53list = pickle.load(open('../data/p53_mutations.pkl', 'rb'))
print('Loaded p53 list')

drugs = pickle.load(open('../data/chembl.pkl', 'rb'))
print('Loaded drugs')

min_max_scaler = pickle.load(open('./datasetScaler.pkl', 'rb'))
inference_model = load_model('./fine_tune_model.h5')

# Build list of drugs
drugList = []
drugLookup = []
print('Building drugs list')
for drugAttr, drugVal in drugs.items():
    drugList.append(drugVal)
    drugLookup.append(drugAttr)

drugResults = {}

for attribute, value in p53list.items():
    print("Testing " + str(attribute))
    print('Building network inputs')
    inputs = []
    for drug in drugList:
        # protein + drug
        inputs.append(value + drug)
    inputs_np = np.array(inputs)
    print('Predicting results')
    drugResults[attribute] = []
    results = inference_model.predict(min_max_scaler.transform(inputs_np))
    for key, value in enumerate(results):
        if value[0] > 0.5 and value[0] < 0.9:
            drugResults[attribute].append(drugLookup[key])

# Remove all that are shared in common with non mutated p53
for attribute, value in drugResults.items():
    if attribute != 'P53-REGULAR':
        for ind, val in enumerate(drugResults['P53-REGULAR']):
            if val in value:
                del value[value.index(val)]

commonDrugs = {}

for attribute, value in drugResults.items():
    if attribute != 'P53-REGULAR':
        for drug in value:
            if hasattr(commonDrugs, drug):
                commonDrugs[drug] += 1
            else:
                commonDrugs[drug] = 1


print(drugResults)
