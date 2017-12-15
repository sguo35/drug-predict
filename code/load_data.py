import numpy as np
import pickle
import random
from sklearn.preprocessing import MinMaxScaler

def load_data():
    # load proteins
    proteins = pickle.load(open('../data/approved_protein_descriptors.pkl', 'rb'))
    print('Loaded proteins...')
    # load drugs
    drugs = pickle.load(open('../data/approved_structures.pkl', 'rb'))
    print('Loaded drugs...')

    # load drug target pairs
    pairs = np.loadtxt('../data/approved_target_pairs.csv', dtype=str, delimiter=',')
    # remove the labels from the pairs
    pairs = np.delete(pairs, [0,1], 0)
    print(pairs)
    print('Loaded drug target pairs...')

    labeled_data_x = []
    labeled_data_y = []

    proteinList = []
    drugList = []

    i = 0

    print('Generating positive data...')
    # generate positive data
    for pair in pairs:
        prot = pair[0]
        drug = pair[1]
        prot = prot.replace("'", "")
        prot = prot.replace("b", "")
        drug = drug.replace("'", "")
        drug = drug.replace("b", "")
        try:
            x = proteins[prot] + drugs[drug]
            y = [1,0]
            labeled_data_x.append(x)
            labeled_data_y.append(y)

            proteinList.append(prot)
            drugList.append(drug)
            i += 1
        except:
            continue

    # Generate negative data
    print('Generating negative data...')
    j = 0
    while j < i:
        prot = random.choice(proteinList)
        drug = random.choice(drugList)

        x = proteins[prot] + drugs[drug]
        y = [0,1]
        labeled_data_x.append(x)
        labeled_data_y.append(y)
        j += 1

    print('Generated data!')

    min_max_scaler = MinMaxScaler()
    out_x = min_max_scaler.fit_transform(labeled_data_x)
    pickle.dump(min_max_scaler, open('./datasetScaler.pkl', 'wb'))

    return out_x, labeled_data_y
