import os,sys
ScanNet_dir = '/Users/jerometubiana/Documents/GitHub/ScanNet_Ub/'
UBD_dir = '/Users/jerometubiana/Documents/GitHub/UBDModel/'
sys.path.append(ScanNet_dir)
sys.path.append(UBD_dir)
os.chdir(UBD_dir)
from utilities import dataset_utils
import pandas as pd
import numpy as np
import pickle
import copy

source_datasets = ['propagatedPssmFiles/PSSM%s.txt'%k for k in range(5)]
target_datasets = ['0608_dataset/labels_fold%s.txt'%k for k in range(1,6)]
Lmin = 10

for i in range(5):
    (list_origins,  # List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
     list_sequences,  # List of corresponding sequences.
     list_resids,  # List of corresponding residue identifiers.
     list_labels) = dataset_utils.read_labels(source_datasets[i])

    Lsequences = np.array([len(seq) for seq in list_sequences])
    subset = Lsequences >= Lmin
    subset = subset & (list_origins != '5u4p_0-B-2')
    print('Discarded examples',list_origins[~subset])

    list_origins = list_origins[subset]
    list_sequences = list_sequences[subset]
    list_resids = list_resids[subset]
    list_labels = list_labels[subset]

    for k in range(len(list_origins)):
        origin = list_origins[k]
        chains = origin.split('_')[1].split('+')
        chains_corrected = []
        for chain in chains:
            chain_split = chain.split('-')
            if origin[:4] == '2fid':
                print(origin)
                model_ = 1
                chain_  = 'B'
            elif len(chain_split) == 3:
                _,chain_,model_ = chain_split
                model_ = int(model_) - 1
            elif len(chain_split) == 2:
                model_,chain_ = chain_split
                model_ = int(model_)
                assert model_ == 0
            else:
                raise ValueError
            chains_corrected.append( (model_,chain_) )
        all_models = np.unique([chain[0] for chain in chains_corrected])
        # if (len(all_models) == 1) & (all_models[0] != 0): # Use the first model of the asymmetric unit.
        #     chains_corrected = [(0,chain_) for model_,chain_ in chains_corrected]

        chains = [chain[2:] for chain in chains]
        origin_corrected = origin.split('_')[0] + '_' + '+'.join( ['%s-%s' % (chain_,model_) for chain_,model_ in chains_corrected] )
        if origin_corrected != origin:
            print(k,origin,origin_corrected)
        resids = list_resids[k]
        new_resids = np.concatenate( (np.array([chains_corrected[chains.index(x)] for x in resids[:,0]]), resids[:,1:]),axis=-1)
        list_origins[k] = origin_corrected
        list_resids[k] = new_resids
    dataset_utils.write_labels(list_origins, list_sequences, list_resids, list_labels, target_datasets[i])