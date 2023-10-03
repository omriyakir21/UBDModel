import sys,os
sys.path.append(os.getcwd())
sys.path.append('../ScanNet_Ub/')
import pandas as pd
from preprocessing import PDBio
import numpy as np
download_batch = 10
try:
    batch = int(sys.argv[1])
except:
    batch = 0
table = pd.read_csv('protein_classification/uniprotnamecsCSV.csv')
all_examples = []
for column in table.columns[:-1]:
    all_examples += list(table[column].dropna())

nexamples = len(all_examples)
batch_size = int(np.ceil( nexamples / download_batch) )
batch_examples = all_examples[batch_size*batch: batch_size * (batch+1)]
print(batch_examples)
pdblist = PDBio.myPDBList()
pdblist.download_pdb_files(batch_examples)