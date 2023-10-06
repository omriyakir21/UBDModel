import sys,os
sys.path.append(os.getcwd())
sys.path.append('../ScanNet_Ub/')
import predict_bindingsites
import pickle
import pandas as pd
import numpy as np
prediction_batch = 10
use_MSA = False
mode = 'ubiquitin'
timestamp = '0310'
try:
    batch = int(sys.argv[1])
except:
    batch = 0

table = pd.read_csv('protein_classification/uniprotnamecsCSV.csv')
all_entries = []
all_entry_names = []
all_entry_sources = []

for column in table.columns[:-1]:
    all_entries += list(table[column].dropna())
    all_entry_names += list(table[column].dropna())
    all_entry_sources += [column] * len( list(table[column].dropna())  )

human_proteome = list(table[table.columns[-1]].dropna())
human_proteome_files = []
human_proteome_folder = '/home/iscb/wolfson/jeromet/AFDB/Human/'
human_proteome_files_unsorted = [x for x in os.listdir(human_proteome_folder) if x.endswith('.cif')]
for id in human_proteome:
    files = [filename for filename in human_proteome_files_unsorted if id in filename]
    if len(files) == 1:
        all_entry_names.append(id)
        all_entries.append(files[0])
    else:
        order = np.argsort([ int(filename.split('-')[1][1:]) for filename in files])
        files = [files[o] for o in order]
        all_entry_names += [ id + f'-F{k}' for k in range(1,len(files) + 1)]
        all_entries += files

    for filename in human_proteome_files_unsorted:
        if id in filename:
            human_proteome_files.append(os.path.join(human_proteome_folder,filename))
            break

all_entries += human_proteome_files
all_entry_names += human_proteome
all_entry_sources += ['Human proteome'] * len(human_proteome)

nentries = len(all_entries)
batch_size = int(np.ceil( nentries / prediction_batch) )
batch_entries = all_entries[batch_size*batch: batch_size * (batch+1)]
batch_entry_names = all_entry_names[batch_size*batch: batch_size * (batch+1)]
batch_entry_sources = all_entry_sources[batch_size*batch: batch_size * (batch+1)]

model = predict_bindingsites.ubiquitin_model_MSA if use_MSA else predict_bindingsites.ubiquitin_model_noMSA
if mode[:-1] == 'ubiquitin':
    model = model[int(mode[-1])-1]

_,_,batch_predictions, batch_resids, batch_sequences = predict_bindingsites.predict_interface_residues(
    query_pdbs=batch_entries,
    query_names=batch_entry_names,
    query_chain_ids=None,
    query_sequences=None,
    pipeline= predict_bindingsites.pipeline_MSA if use_MSA else predict_bindingsites.pipeline_noMSA,
    model=model,
    model_name=mode,
    model_folder=predict_bindingsites.model_folder,
    structures_folder=predict_bindingsites.structures_folder,
    biounit=False,
    assembly=True,
    layer=None,
    use_MSA=use_MSA,
    overwrite_MSA=False,
    Lmin=1,
    output_predictions=False,
    aggregate_models=True,
    output_chimera='annotation',
    permissive=False,
    output_format='numpy')

dict_predictions = dict(zip(batch_entry_names,batch_predictions))
dict_resids = dict(zip(batch_entry_names,batch_resids))
dict_sequences =  dict(zip(batch_entry_names,batch_sequences))
dict_sources = dict(zip(batch_entry_names,batch_entry_sources))

env = {
    'dict_predictions':dict_predictions,
    'dict_resids':dict_resids,
    'dict_sequences':dict_sequences,
    'dict_sources':dict_sources
}
results_folder = f'protein_classification/predictions_{timestamp}/'
os.makedirs(results_folder,exist_ok=True)
pickle.dump(env,open(os.path.join(results_folder,f'batch_predictions_{batch}_{prediction_batch}.pkl'),'wb'))


#%%
if __name__ != '__main__':
    import pickle,os
    timestamp = '0310'
    prediction_batch = 10
    results_folder = f'protein_classification/predictions_{timestamp}/'
    files = [os.path.join(results_folder,f'batch_predictions_{batch}_{prediction_batch}.pkl') for batch in range(prediction_batch)]

    env = pickle.load(open(files[0],'rb'))
    for file in files[1:]:
        env_ = pickle.load(open(file,'rb'))
        for key in env.keys():
            env[key].update(env_[key])
            print(file, key, len(env[key]))
    pickle.dump(env,open(os.path.join(results_folder,f'all_predictions_{timestamp}.pkl'),'wb'))

    lengthes = [len(x) for x in env['dict_sequences'].values()]
    print(max(lengthes), min(lengthes) )