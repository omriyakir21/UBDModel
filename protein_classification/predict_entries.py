import sys,os
sys.path.append(os.getcwd())
sys.path.append('../ScanNet_Ub/')
def set_num_threads(num_threads=2):
    os.environ["MKL_NUM_THREADS"] = "%s"%num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = "%s"%num_threads
    os.environ["OMP_NUM_THREADS"] = "%s"%num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = "%s"%num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = "%s"%num_threads
    os.environ["NUMBA_NUM_THREADS"] = "%s"%num_threads
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return
set_num_threads(num_threads=8)
import predict_bindingsites
from preprocessing import PDBio,PDB_processing,sequence_utils
import pickle
import pandas as pd
import numpy as np
import copy
from functools import partial
from multiprocessing import Pool
prediction_batch = 20
'''
for k in range(20):
    print('python protein_classification/predict_entries.py 0304 interface 0 %s;'%k)
    print('python protein_classification/predict_entries.py 0304 ubiquitin 0 %s;'%k)
    print('python protein_classification/predict_entries.py 0304 interface 1 %s;'%k)
    print('python protein_classification/predict_entries.py 0304 ubiquitin 1 %s;'%k)

for k in range(20):
    print('sbatch submit_job_cpu.sh 0304 interface 0 %s;'%k)
    print('sbatch submit_job_cpu.sh 0304 ubiquitin 0 %s;'%k)
    print('sbatch submit_job_cpu.sh 0304 interface 1 %s;'%k)
    print('sbatch submit_job_cpu.sh 0304 ubiquitin 1 %s;'%k)
    print('sleep 25000;')    
    '''

try:
    timestamp = sys.argv[1]
    mode = sys.argv[2]
    use_MSA = bool(int(sys.argv[3]))
    batch = int(sys.argv[4])
except:
    timestamp = '0304'
    mode = 'interface'
    use_MSA = True
    batch = 0

if timestamp == '0304':
    sources = [
        'GO',
        'Human',
        'Yeast',
        'Ecoli',
        'Arabidopsis',
        'Celegans'
    ]
else:
    sources = ['GO','Human']

all_entries = []
all_entry_names = []
all_entry_sources = []
if 'GO' in sources:
    table = pd.read_csv('protein_classification/uniprotnamecsCSV.csv')
    for column in table.columns[:-1]:
        list_entries = list(table[column].dropna().unique())
        for entry in list_entries:
            if entry in all_entries:
                all_entry_sources[all_entries.index(entry)] += f';{column}'
            else:
                all_entries.append(entry)
                all_entry_names.append(entry)
                all_entry_sources.append(column)
    priority_order = ['E1','E2','E3','DUB','ubiquitinBinding']
    for i in range(len(all_entry_sources)):
        for k in range(len(priority_order)):
            if priority_order[k] in all_entry_sources[i]:
                break
        all_entry_sources[i] = copy.copy(priority_order[k])

for source in sources:
    if source != 'GO':
        proteome_folder =f'/home/iscb/wolfson/jeromet/AFDB/{source}/'
        proteome_files = []
        proteome_files_unsorted = [x for x in os.listdir(proteome_folder) if x.endswith('.cif')]

        if source == 'Human':
            table = pd.read_csv('protein_classification/uniprotnamecsCSV.csv')
            proteome = np.unique(list(table[table.columns[-1]].dropna()))
        else:
            proteome = np.unique(  [x.split('-')[1] for x in proteome_files_unsorted] )
            proteome = [x for x in proteome if not x in all_entry_names]
        for id in proteome:
            files = [filename for filename in proteome_files_unsorted if id in filename]
            if len(files) == 1:
                all_entry_names.append(id)
                all_entries.append(os.path.join(proteome_folder,files[0]) )
                all_entry_sources.append(f'{source} proteome')
            else:
                order = np.argsort([ int(filename.split('-')[-2][1:]) for filename in files])
                files = [files[o] for o in order]
                all_entry_names += [ id + f'-F{k}' for k in range(1,len(files) + 1)]
                all_entries += [os.path.join(proteome_folder,file) for file in files]
                all_entry_sources += [f'{source} proteome'] * len(files)

nentries = len(all_entries)
batch_size = int(np.ceil( nentries / prediction_batch) )
batch_entries = all_entries[batch_size*batch: batch_size * (batch+1)]
batch_entry_names = all_entry_names[batch_size*batch: batch_size * (batch+1)]
batch_entry_sources = all_entry_sources[batch_size*batch: batch_size * (batch+1)]

if mode.startswith('ubiquitin'):
    model = predict_bindingsites.ubiquitin_model_MSA if use_MSA else predict_bindingsites.ubiquitin_model_noMSA
    if mode[:-1] == 'ubiquitin':
        model = model[int(mode[-1])-1]
elif mode == 'interface':
    model = predict_bindingsites.interface_model_MSA if use_MSA else predict_bindingsites.interface_model_noMSA


if use_MSA:
    def _call_hhblits(x):
        sequence,target_location = x            
        return sequence_utils.call_hhblits(sequence, target_location, cores=4)

    batch_MSAs = [
    os.path.join( predict_bindingsites.MSA_folder,
    f'MSA_{entry_name}_0_A.fasta') for entry_name in batch_entry_names]

    MSAs2compute = []
    sequences2compute = []
    for entry,MSA in zip(batch_entries,batch_MSAs):
        if not os.path.exists(MSA):
            MSAs2compute.append(MSA)
            sequences2compute.append(PDB_processing.process_chain(PDBio.load_chains(file=entry)[1])[0])
    print('Need to calculate %s out of %s MSAs' % (len(MSAs2compute),len(batch_MSAs)) )    
    pool = Pool(10)
    list(pool.map( _call_hhblits, list(zip(sequences2compute,MSAs2compute)) ))
    pool.close()



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
    output_format='numpy',
    ncores=40)

dict_predictions = dict(zip(batch_entry_names,batch_predictions))
dict_resids = dict(zip(batch_entry_names,batch_resids))
dict_sequences =  dict(zip(batch_entry_names,batch_sequences))
dict_sources = dict(zip(batch_entry_names,batch_entry_sources))
dict_pdb_files = dict(zip(batch_entry_names, (PDBio.getPDB(entry)[0] for entry in batch_entries) ))


env = {
    'dict_predictions':dict_predictions,
    'dict_resids':dict_resids,
    'dict_sequences':dict_sequences,
    'dict_sources':dict_sources,
    'dict_pdb_files': dict_pdb_files,
}
results_folder = f'protein_classification/predictions_{timestamp}/'
os.makedirs(results_folder,exist_ok=True)
pickle.dump(env,open(os.path.join(results_folder,f'batch_predictions_{mode}_MSA_{use_MSA}_{batch}_{prediction_batch}.pkl'),'wb'))
#
#
# #%%
# if __name__ != '__main__':
    import pickle,os
    timestamp = '0304'
    prediction_batch = 20
    results_folder = f'protein_classification/predictions_{timestamp}/'
    for use_MSA in [True,False]:
        for mode in ['ubiquitin','interface']:
            files = [os.path.join(results_folder,f'batch_predictions_{mode}_MSA_{use_MSA}_{batch}_{prediction_batch}.pkl') for batch in range(prediction_batch)]
            if mode =='ubiquitin':
                env = pickle.load(open(files[0],'rb'))
                env[f'dict_predictions_{mode}'] = env.pop('dict_predictions')
                start = 1
            else:
                start = 0
            for file in files[start:]:
                env_ = pickle.load(open(file,'rb'))
                for key in env_.keys():
                    if key == 'dict_predictions':
                        key_ = f'{key}_{mode}'
                    else:
                        key_ = key
                    if key_ in env.keys():
                        env[key_].update(env_[key])
                    else:
                        env[key_] = env_[key]
                print(mode,file, key, len(env[key]))
        pickle.dump(env,open(os.path.join(results_folder,f'all_predictions_{timestamp}_MSA_{use_MSA}.pkl'),'wb'))

        lengthes = [len(x) for x in env['dict_sequences'].values()]
        print(max(lengthes), min(lengthes) )