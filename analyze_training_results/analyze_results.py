import pickle
import os,sys
ScanNet_dir = '/Users/jerometubiana/Documents/GitHub/ScanNet_Ub/'
UBD_dir = '/Users/jerometubiana/Documents/GitHub/UBDModel/'
sys.path.append(ScanNet_dir)
sys.path.append(UBD_dir)
os.chdir(ScanNet_dir)
import matplotlib.pyplot as plt
from preprocessing import pipelines
from utilities import dataset_utils,paths,wrappers
import pandas as pd
import numpy as np
import pickle
import copy
from train import make_PR_curves


# env = pickle.load(open(os.path.join(UBD_dir,'analyze_training_results/','training_results.pkl'),'rb'))
# env = pickle.load(open(os.path.join(ScanNet_dir,'predictions/','predictions_ubiquitin_ScanNet_PUI_retrained_0708.pkl'),'rb'))
env = pickle.load(open(os.path.join(ScanNet_dir,'predictions/','predictions_ubiquitin_ScanNet_PUI_retrained_0108.pkl'),'rb'))
# l = 3

list_origins = env['list_origins']
list_sequences = env['list_sequences']
list_resids = env['list_resids']
list_labels = env['list_labels']
list_labels = np.array([(labels>=2).astype(int) for labels in list_labels],dtype=object)

list_weights = np.array(env['list_weights'])
list_source_dataset = ['Fold 1'] * 201 + ['Fold 2'] * 46 + ['Fold 3'] * 60 + ['Fold 4'] * 89 + ['Fold 5'] * 68
# list_models = env['list_models']
# similar_structures = env['similar_structures']
# all_list_predictions = env['all_list_predictions']

list_labels = []
list_files = ['pipelines/0608/UBS_fold%s_pipeline_ScanNet_aa-sequence_atom-valency_frames-triplet_sidechain_Beff-500.data'%k for k in range(1,6)]
for file in list_files:
    list_labels += list(pickle.load(open(file,'rb'))['outputs'])
list_labels = np.array([np.argmax(labels,axis=-1) for labels in list_labels],dtype=object)

# list_predictions = np.array(all_list_predictions[list_models[l]])
# model_name = list_models[l]
list_predictions = env['list_predictions']
model_name = 'ScanNet_Ub'

subsets = [np.array(list_source_dataset) == 'Fold %s'%k for k in range(1,6)]
subset_names = ['Fold %s'%k for k in range(1,6)]

sliced_predictions = [list_predictions[subsets[j]] for j in range(5)]
sliced_labels = [list_labels[subsets[j]] for j in range(5)]
sliced_weights = [list_weights[subsets[j]] for j in range(5)]

fig, ax = make_PR_curves(
    sliced_labels,
    sliced_predictions,
    sliced_weights,
    subset_names,
    title=model_name,
    figsize=(10, 10),
    margin=0.05, grid=0.1
    , fs=25)



baseline_proba = 0.12
eps = 1e-4
delta_cross_entropy = [( (labels>0) * np.log((eps+predictions)/(eps+baseline_proba) ) + (1-(labels>0) ) * np.log((1+eps-predictions)/(1+eps-baseline_proba) ) ).mean(-1) for labels,predictions in zip(list_labels,list_predictions)]
order = np.argsort(delta_cross_entropy)

bad_false_negatives = [predictions[labels>0].min() for labels,predictions in zip(list_labels,list_predictions)]
bad_false_positives = [predictions[labels==0].max() for labels,predictions in zip(list_labels,list_predictions)]
order = np.argsort(bad_false_negatives)
order = np.argsort(bad_false_positives)[::-1]

#%%

# visualize predictions
from predict_bindingsites import write_predictions
from utilities import chimera
from preprocessing import PDBio
for k in order[:5]:
    j = 0
    origin = env['list_origins'][k]
    resids = env['list_resids'][k]
    resids = np.stack([np.zeros(len(resids),dtype=int).astype(str),resids[:,-2], resids[:,-1]] ,axis=1)

    sequence = env['list_sequences'][k]
    labels = env['list_labels'][k]
    # predictions = env['all_list_predictions'][env['list_models'][j]][k]
    predictions = env['list_predictions'][k]
    # predictions = np.median(env['list_all_predictions'][k],axis=-1)
    # predictions = env['list_all_predictions'][k].mean(-1)
    # predictions = labels/4.

    predictions_csv_file = os.path.join(UBD_dir,'analyze_training_results/','predictions_%s.csv'%origin)
    pdb_file,_ = PDBio.getPDB(origin)
    output_file = os.path.join(UBD_dir,'analyze_training_results/','annotated_%s.pdb'%origin)
    write_predictions(predictions_csv_file, resids, sequence, predictions)
    chimera.annotate_pdb_file(pdb_file,predictions_csv_file,output_file,output_script=True,mini=0.0,maxi=0.5,version='default', field = 'Binding site probability')

    labels_csv_file = os.path.join(UBD_dir,'analyze_training_results/','labels_%s.csv'%origin)
    pdb_file,_ = PDBio.getPDB(origin)
    output_file = os.path.join(UBD_dir,'analyze_training_results/','annotated_labels_%s.pdb'%origin)
    write_predictions(predictions_csv_file, resids, sequence, labels)
    chimera.annotate_pdb_file(pdb_file,predictions_csv_file,output_file,output_script=True,mini=0.0,maxi=3.0,version='default', field = 'Binding site probability')
