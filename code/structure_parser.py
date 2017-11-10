from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
import pickle
 
mols = Chem.SDMolSupplier('../data/experimental_structures.sdf')
act = [ ]
fps = {}
index = 0


for mol in mols:
    if mol is None: continue
    
    print(mol.GetProp('DATABASE_ID'))
    index += 1
    arr = np.zeros( (1,) )
    fp = AllChem.GetMorganFingerprintAsBitVect( mol, 6, nBits=2048 )
    DataStructs.ConvertToNumpyArray( fp, arr )
    fps[mol.GetProp('DATABASE_ID')] = arr.tolist()

pickle.dump(fps, open('../data/experimental_structures.pkl', 'wb'))
print('Saved structures')