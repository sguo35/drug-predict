from propy import PyPro
from Bio import SeqIO

import pickle

count = 0

proteins = {}

for seq_record in SeqIO.parse('../data/approved_protein.fasta', 'fasta'):
    print(seq_record.name)
    des = PyPro.GetProDes(str(seq_record.seq))
    print(count)
    count += 1

    # Tripeptide calculation
    TPCompObj = des.GetTPComp()
    TPComp = []
    for key, value in TPCompObj.iteritems():
        TPComp.append(value)


    # Dipeptide calculation
    DPCompObj = des.GetDPComp()
    DPComp = []
    for key, value in DPCompObj.iteritems():
        DPComp.append(value)

    # aminoacid calculation
    AACompObj = des.GetAAComp()
    AAComp = []
    for key, value in AACompObj.iteritems():
        AAComp.append(value)

    # Merge them to form protein sequence descriptor
    AAComp = AAComp + DPComp + TPComp
    proteins[seq_record.name.replace('drugbank_target|', '', 1000)] = AAComp

pickle.dump(proteins, open('../data/approved_protein_descriptors.pkl', 'wb'), protocol=2)
print('Saved descriptors')