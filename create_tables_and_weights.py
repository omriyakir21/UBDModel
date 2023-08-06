import os,sys
sys.path.append(os.getcwd())
import numpy as np
import subprocess,shutil
import pandas as pd

def read_labels(input_file, nmax=np.inf, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:-2])
                sequence += line_splitted[-2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                else:
                    labels.append(float(line_splitted[-1]))

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    list_origins = np.array(list_origins)
    list_sequences = np.array(list_sequences)
    list_labels = np.array(list_labels)
    list_resids = np.array(list_resids)
    return list_origins, list_sequences, list_resids, list_labels



def cluster_sequences(list_sequences, seqid= 1.0, coverage = 0.8, covmode = '0'):
    path2mmseqs = '/opt/anaconda3/bin/mmseqs'
    path2mmseqsdatabases = '/Users/jerometubiana/sequence_databases/'
    path2mmseqstmp = '/Users/jerometubiana/tmp/'

    rng = np.random.randint(0,high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp,'tmp_input_file_%s.fasta' % rng )
    tmp_output = os.path.join(path2mmseqstmp,'tmp_output_file_%s' % rng )

    with open(tmp_input,'w') as f:
        for k,sequence in enumerate(list_sequences):
            f.write('>%s\n'%k)
            f.write('%s\n' % sequence)

    command = ('{mmseqs} easy-cluster {fasta} {result} {tmp} --min-seq-id %s -c %s --cov-mode %s' % (seqid,coverage,covmode)).format(mmseqs=path2mmseqs,fasta=tmp_input,result=tmp_output,tmp=path2mmseqstmp)
    subprocess.run(command.split(' '))

    with open(tmp_output + '_rep_seq.fasta','r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]
    cluster_indices = np.zeros(len(list_sequences),dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv',sep='\t',header=None).to_numpy(dtype=int)
    for i,j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)
    for file in [tmp_output + '_rep_seq.fasta',tmp_output + '_all_seqs.fasta',tmp_output + '_cluster.tsv']:
        os.remove(file)
    return np.array(cluster_indices),np.array(representative_indices)


def calculate_weights(list_sequences,resolutions = [100,95,90,70],coverage=0.8, covmode = '0'):
    list_sequences = np.array(list_sequences,dtype=str)
    N = len(list_sequences)
    nresolutions = len(resolutions)
    hierarchical_cluster_indices = [np.arange(N)]
    hierarchical_representative_indices = [ np.arange(N)]
    hierarchical_representative_sequences = [list_sequences]
    hierarchical_cluster_sizes = [np.ones(N)]

    for k,resolution in enumerate(resolutions):
        cluster_indices, representative_indices = cluster_sequences(hierarchical_representative_sequences[k], seqid=resolution/100,
                                                                    coverage = coverage, covmode=covmode)
        cluster_sizes = np.array([(cluster_indices == k).sum() for k in range(len(representative_indices))])
        representative_sequences = hierarchical_representative_sequences[k][representative_indices]
        hierarchical_cluster_indices.append(cluster_indices)
        hierarchical_representative_indices.append(representative_indices)
        hierarchical_representative_sequences.append(representative_sequences)
        hierarchical_cluster_sizes.append(cluster_sizes)

    hierarchical_num_clusters = [len(representative_sequences) for representative_sequences in hierarchical_representative_sequences]
    hierarchical_weights = [np.ones(hierarchical_num_clusters[-1])]
    for k in range(1,nresolutions+1)[::-1]:
        num_neighbours = 1.0 / hierarchical_cluster_sizes[k]
        weights = (hierarchical_weights[-1] * num_neighbours)[hierarchical_cluster_indices[k]]
        hierarchical_weights.append(weights)
    hierarchical_weights =hierarchical_weights[::-1]
    return hierarchical_weights[0]


#%%

if __name__ == '__main__':

    input_folder =  '0608_dataset/'

    all_origins = []
    all_folds = []
    all_weights = []
    all_sequences = []

    for k in range(1,6):
        dataset_file = os.path.join(input_folder,'labels_fold%s.txt'%k)
        list_origins, list_sequences, list_resids, list_labels = read_labels(dataset_file)
        all_origins += list(list_origins)
        all_folds += ['Fold %s'%k] * len(list_origins)
        all_sequences += list(list_sequences)

    all_origins = np.array(all_origins)
    all_folds = np.array(all_folds)
    all_sequences = np.array(all_sequences)

    all_weights_v0 = np.ones(len(all_sequences))
    all_weights_v1 = calculate_weights(all_sequences,resolutions=[100,95,90,70])
    all_weights_v2 = calculate_weights(all_sequences,resolutions=[95])
    all_weights_v3 = calculate_weights(all_sequences,resolutions=[70])

    table = pd.DataFrame({
        'PDB ID': all_origins,
        'Length': [len(sequence) for sequence in all_sequences],
        'Set': all_folds,
        'Sample weight': all_weights_v1,
        'Sample weight none': all_weights_v0,
        'Sample weight flat95': all_weights_v2,
        'Sample weight flat70': all_weights_v3,
    })
    table.to_csv(os.path.join(input_folder,'table.csv'))

    #%%

    # dataset_file = 'FullPssmContent'
    # all_origins, all_sequences, all_resids, all_labels = read_labels(dataset_file)


