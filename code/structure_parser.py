from __future__ import print_function
import openbabel as ob
import pybel

import numpy

molecules = pybel.readfile('sdf', '../data/experimental_structures.sdf')

count = 0
structArray = []

for mol in molecules:
    obmol = mol.OBMol
    ob_fingerprint1 = ob.OBFingerprint.FindFingerprint("ECFP2")
    fp1 = ob.vectorUnsignedInt()
    ob_fingerprint1.GetFingerprint(obmol, fp1, 1024)

    ob_fingerprint2 = ob.OBFingerprint.FindFingerprint("ECFP4")
    fp2 = ob.vectorUnsignedInt()
    ob_fingerprint2.GetFingerprint(obmol, fp2, 1024)

    ob_fingerprint3 = ob.OBFingerprint.FindFingerprint("ECFP6")
    fp3 = ob.vectorUnsignedInt()
    ob_fingerprint3.GetFingerprint(obmol, fp3, 1024)
    value1 = pybel.Fingerprint(fp1).bits
    value2 = pybel.Fingerprint(fp2).bits
    value3 = pybel.Fingerprint(fp3).bits

    bit1 = [0] * 2048
    bit2 = [0] * 2048
    bit3 = [0] * 2048
    for index in value1:
        bit1[index - 1] = 1
    for index in value2:
        bit2[index - 1] = 1
    for index in value3:
        bit3[index - 1] = 1

    bitArray = bit1 + bit2 + bit3
    structArray.append(bitArray)
    print(count)
    count += 1

exported = numpy.asarray(structArray)
numpy.savetxt('../data/experimental_structures.csv', exported, delimiter=',')
print('Saved structures')