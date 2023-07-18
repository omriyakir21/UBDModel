import os,sys
ScanNet_dir = '/Users/jerometubiana/Documents/GitHub/ScanNet_Ub/'
UBD_dir = '/Users/jerometubiana/Documents/GitHub/UBDModel/'
sys.path.append(ScanNet_dir)
sys.path.append(UBD_dir)
os.chdir(ScanNet_dir)
from preprocessing import pipelines
from utilities import dataset_utils,paths,wrappers
import pandas as pd
import numpy as np
import pickle
import copy




list_models = [
    'PUI_retrained',
    'PUI_retrained_noMSA',
    # 'PUI_old',
    # 'PUI_old_noMSA'
]

ncores = 8
mode = 'ubiquitin'

model_acronyms = {'epitope': 'PAI', 'idp': 'PIDPI', 'ubiquitin': 'PUI', 'ubiquitin_old': 'PUI_old'}
dataset_folders = {'epitope': 'BCE', 'idp': 'PIDPBS', 'ubiquitin': 'UBS', 'ubiquitin_old': 'UBS_old'}
full_names = {'epitope': 'B-cell epitopes', 'idp': 'intrinsically disordered protein binding sites',
              'ubiquitin': 'ubiquitin binding sites', 'ubiquitin_old': 'ubiquitin binding sites'}
Lmax_aas = {'epitope': 2120, 'idp': 1352, 'ubiquitin_old': 1024, 'ubiquitin': 2353}
Lmax_aa = Lmax_aas[mode]

list_datasets = ['fold1','fold2','fold3','fold4','fold5']
list_dataset_names = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5']

pipeline_MSA = pipelines.ScanNetPipeline(aa_features='pwm')
pipeline_noMSA = pipelines.ScanNetPipeline(aa_features='sequence')


list_dataset_locations = ['datasets/%s/labels_%s.txt' % (dataset_folders[mode], dataset) for dataset in list_datasets]
dataset_table = pd.read_csv('datasets/%s/table.csv' % dataset_folders[mode], sep=',')

list_inputs_MSA = []
list_inputs_noMSA = []
list_outputs = []
list_weights = []



list_origins_ = []
list_sequences_ = []
list_resids_ = []
list_labels_ = []


for dataset, dataset_name, dataset_location in zip(list_datasets, list_dataset_names, list_dataset_locations):
    # Parse label files
    (list_origins,  # List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
     list_sequences,  # List of corresponding sequences.
     list_resids,  # List of corresponding residue identifiers.
     list_labels) = dataset_utils.read_labels(dataset_location)  # List of residue-wise labels

    list_origins_ += list(list_origins)
    list_sequences_ += list(list_sequences)
    list_resids_ += list(list_resids)
    list_labels_ += list(list_labels)

    if dataset_folders[mode] == 'UBS':
        list_labels = np.array([labels >= 2 for labels in list_labels])

    '''
    Build processed dataset. For each protein chain, build_processed_chain does the following:
    1. Download the pdb file (biounit=True => Download assembly file, biounit=False => Download asymmetric unit file).
    2. Parse the pdb file.
    3. Construct atom and residue point clouds, determine triplets of indices for each atomic/residue frame.
    4. If evolutionary information is used, build an MSA using HH-blits and calculates a Position Weight Matrix (PWM).
    5. If labels are provided, aligns them onto the residues found in the pdb file.
    '''
    inputs_MSA, outputs, failed_samples = pipeline_MSA.build_processed_dataset(
        '%s_%s' % (dataset_folders[mode], dataset),
        list_origins=list_origins,  # Mandatory
        list_resids=list_resids,  # Optional
        list_labels=list_labels,  # Optional
        biounit=False,
        # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
        save=True,
        # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
        fresh=False,  # If fresh = False, attemps to load pickle files first.
        ncores=ncores
    )

    inputs_noMSA, outputs_noMSA, _ = pipeline_noMSA.build_processed_dataset(
        '%s_%s' % (dataset_folders[mode], dataset),
        list_origins=list_origins,  # Mandatory
        list_resids=list_resids,  # Optional
        list_labels=list_labels,  # Optional
        biounit=False,
        # Whether to use biological assembly files or the regular pdb files (asymmetric units). True for PPBS data set, False for BCE data set.
        save=True,
        # Whether to save the results in pickle file format. Files are stored in the pipeline_folder defined in paths.py
        fresh=False,  # If fresh = False, attemps to load pickle files first.
        ncores=ncores
    )

    # for output, output_noMSA in zip(outputs, outputs_noMSA):
    #     assert (output == output_noMSA).min()

    weights = np.array(dataset_table['Sample weight'][dataset_table['Set'] == dataset_name])
    weights = np.array([weights[b] for b in range(len(weights)) if not b in failed_samples])
    list_inputs_MSA.append(inputs_MSA)
    list_inputs_noMSA.append(inputs_noMSA)
    list_outputs.append(outputs)
    list_weights.append(weights)


all_predictions = [[] for _ in list_models]

for k in range(5): # 5-fold training/evaluation.
    test_inputs_MSA = list_inputs_MSA[k]
    test_inputs_noMSA = list_inputs_noMSA[k]
    test_outputs = list_outputs[k]
    test_weights = list_weights[k]

    for j,model in enumerate(list_models):
        full_name = os.path.join(paths.model_folder, 'ScanNet_%s_%s' % (model,k) )
        model_obj =  wrappers.load_model(full_name)
        test_predictions = model_obj.predict(
                test_inputs_noMSA if 'noMSA' in model else test_inputs_MSA,return_all=False,batch_size=1)
        all_predictions[j].append(test_predictions)


all_predictions = [np.concatenate(x) for x in all_predictions]


all_list_predictions = dict([(model, all_predictions) for model,all_predictions in zip(list_models,all_predictions)])
list_origins,list_resids,list_labels,list_sequences = list_origins_,list_resids_,list_labels_,list_sequences_


from create_tables_and_weights import cluster_sequences
cluster_indices, _ = cluster_sequences(list_sequences, seqid= 0.95, coverage = 0.8, covmode = '0')
similar_structures = {}
for k, origin in enumerate(list_origins):
    similar_structures[origin] = list(np.array(list_origins)[cluster_indices == cluster_indices[k]])


env = {
    'list_origins': list_origins,
    'list_resids': list_resids,
    'list_labels': list_labels,
    'list_sequences': list_sequences,
    'all_list_predictions':all_list_predictions,
    'similar_structures':similar_structures
}


pickle.dump(env,open(os.path.join(UBD_dir,'analyze_training_results/','training_results.pkl'),'wb'))


#%%



pkl_files = ['predictions/predictions_ubiquitin_ScanNet_PUI_retrained.pkl',
        'predictions/predictions_ubiquitin_ScanNet_PUI_retrained_noMSA.pkl',
        'predictions/predictions_ubiquitin_ScanNet_PUI_retrained_freeze.pkl',
        'predictions/predictions_ubiquitin_ScanNet_PUI_retrained_noMSA_freeze.pkl'
        ]

list_models = ['PUI','PUI_noMSA','PUI_freeze','PUI_noMSA_freeze']

env = {}
env['all_list_predictions'] = {}
for k,pkl_file in enumerate(pkl_files):
    env.update(pickle.load(open(pkl_file,'rb')))
    env['all_list_predictions'][list_models[k]]  = env['list_predictions']
del env['list_predictions']

from create_tables_and_weights import cluster_sequences
cluster_indices, _ = cluster_sequences(env['list_sequences'], seqid= 0.95, coverage = 0.8, covmode = '0')
similar_structures = {}
for k, origin in enumerate(env['list_origins']):
    similar_structures[origin] = list(np.array(env['list_origins'])[cluster_indices == cluster_indices[k]])
env['similar_structures'] = similar_structures
env['list_models'] = list_models
pickle.dump(env,open(os.path.join(UBD_dir,'analyze_training_results/','training_results.pkl'),'wb'))

#%%

# visualize predictions
from predict_bindingsites import write_predictions
from utilities import chimera
from preprocessing import PDBio
env = pickle.load(open(os.path.join(UBD_dir,'analyze_training_results/','training_results.pkl'),'rb'))
k = 32
j = 0
origin = env['list_origins'][k]
resids = env['list_resids'][k]
resids = np.stack([np.zeros(len(resids),dtype=int).astype(str),resids[:,-2], resids[:,-1]] ,axis=1)

sequence = env['list_sequences'][k]
labels = env['list_labels'][k]
predictions = env['all_list_predictions'][env['list_models'][j]][k]

predictions_csv_file = os.path.join(UBD_dir,'analyze_training_results/','predictions_%s.csv'%origin)
pdb_file,_ = PDBio.getPDB(origin)
output_file = os.path.join(UBD_dir,'analyze_training_results/','annotated_%s.pdb'%origin)
write_predictions(predictions_csv_file, resids, sequence, predictions)
chimera.annotate_pdb_file(pdb_file,predictions_csv_file,output_file,output_script=True,mini=0.0,maxi=0.5,version='default', field = 'Binding site probability')



#%% Regenerate overfitted predictions...

import pickle
import os,sys
ScanNet_dir = '/Users/jerometubiana/Documents/GitHub/ScanNet_Ub/'
UBD_dir = '/Users/jerometubiana/Documents/GitHub/UBDModel/'
sys.path.append(ScanNet_dir)
sys.path.append(UBD_dir)
os.chdir(ScanNet_dir)
from preprocessing import pipelines
from utilities import dataset_utils,paths,wrappers
import pandas as pd
import numpy as np
import pickle
import copy

use_evolutionary = True
aa_features = 'pwm' if use_evolutionary else 'sequence'
model_paths = [os.path.join(ScanNet_dir,'models/',f'ScanNet_PUI_retrained_{fold}') for fold in range(5)]
models = [wrappers.load_model(model_path,Lmax = 2353) for model_path in model_paths]

pipeline_folder = os.path.join(ScanNet_dir, 'pipelines/')
preprocessed_datasets = [os.path.join(pipeline_folder, f'UBS_fold{fold}_pipeline_ScanNet_aa-{aa_features}_atom-valency_frames-triplet_sidechain_Beff-500.data') for fold in range(1,6)]

all_predictions = [[] for _ in range(5)]
for preprocessed_dataset in preprocessed_datasets:
    env = pickle.load(open(preprocessed_dataset,'rb'))
    inputs,outputs,failed_samples = env['inputs'],env['outputs'],env['failed_samples']
    for k in range(5):
        all_predictions[k] += list(models[k].predict(inputs,batch_size=1))

all_predictions = [np.stack([all_predictions[k][l] for k in range(5)],axis=1) for l in range(len( all_predictions[0] ))]


env = pickle.load(open(os.path.join(UBD_dir,'analyze_training_results/','training_results.pkl'),'rb') )
env['overfitted_predictions'] = all_predictions
pickle.dump(env,open(os.path.join(UBD_dir,'analyze_training_results/','training_results_overfitted.pkl'),'wb') )