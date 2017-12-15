from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
import pickle
 
mols = Chem.SDMolSupplier('../data/approved_structures.sdf')
act = [ ]
fps = {}
index = 0


for mol in mols:
    if mol is None: continue
    #print(mol.GetPropNames())
    #print(mol.GetProp('CHEMBL_ID'))
    print(mol.GetProp('DATABASE_ID'))
    index += 1
    arr2 = np.zeros( (1,) )
    arr4 = np.zeros( (1,) )
    arr6 = np.zeros( (1,) )
    fp2 = AllChem.GetMorganFingerprintAsBitVect( mol, 1, nBits=2048 )
    fp4 = AllChem.GetMorganFingerprintAsBitVect( mol, 2, nBits=2048 )
    fp6 = AllChem.GetMorganFingerprintAsBitVect( mol, 3, nBits=2048 )
    DataStructs.ConvertToNumpyArray( fp2, arr2 )
    DataStructs.ConvertToNumpyArray( fp4, arr4 )
    DataStructs.ConvertToNumpyArray( fp6, arr6 )
    #fps[mol.GetProp('CHEMBL_ID')] = arr.tolist()
    fps[mol.GetProp('DATABASE_ID')] = arr2.tolist() + arr4.tolist() + arr6.tolist()

pickle.dump(fps, open('../data/approved_structures.pkl', 'wb'), protocol=2)
print('Saved structures')