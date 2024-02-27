import pickle
import subprocess
import pandas as pd
import numpy as np
import os

def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object



# allInfoDict = loadPickle(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining1902\allInfoDict.pkl')
# allProteinsDict = dict()
# allProteinsDict['x'] = allInfoDict['x_train']+allInfoDict['x_cv']+allInfoDict['x_test']
# allProteinsDict['y'] = np.concatenate((allInfoDict['y_train'],allInfoDict['y_cv'],allInfoDict['y_test']))
allProteinsDict = loadPickle('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/allProteinInfo.pkl')
sequences = loadPickle('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/allProteinSequences.pkl')


def cluster_sequences(list_sequences, seqid=0.7, coverage=0.8, covmode='0'):
    path2mmseqs = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mmseqs'
    path2mmseqstmp = '/home/iscb/wolfson/omriyakir/UBDModel/mmseqs2'

    rng = np.random.randint(0, high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp, 'tmp_input_file_%s.fasta' % rng)
    tmp_output = os.path.join(path2mmseqstmp, 'tmp_output_file_%s' % rng)

    with open(tmp_input, 'w') as f:
        for k, sequence in enumerate(list_sequences):
            f.write('>%s\n' % k)
            f.write('%s\n' % sequence)

    command = ('{mmseqs} easy-cluster {fasta} {result} {tmp} --min-seq-id %s -c %s --cov-mode %s' % (
        seqid, coverage, covmode)).format(mmseqs=path2mmseqs, fasta=tmp_input, result=tmp_output, tmp=path2mmseqstmp)
    subprocess.run(command.split(' '))

    with open(tmp_output + '_rep_seq.fasta', 'r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]
    cluster_indices = np.zeros(len(list_sequences), dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv', sep='\t', header=None).to_numpy(dtype=int)
    for i, j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)
    for file in [tmp_output + '_rep_seq.fasta', tmp_output + '_all_seqs.fasta', tmp_output + '_cluster.tsv']:
        os.remove(file)
    saveAsPickle(cluster_indices, path2mmseqstmp + '/clusterIndices')
    return np.array(cluster_indices), np.array(representative_indices)

cluster_indices, representative_indices = cluster_sequences(sequences)

path2mafft = '/usr/bin/mafft'


def createClusterParticipantsIndexes(clusterIndexes):
    clustersParticipantsList = []
    for i in range(np.max(clusterIndexes) + 1):
        clustersParticipantsList.append(np.where(clusterIndexes == i)[0])
    return clustersParticipantsList

clustersParticipantsList = createClusterParticipantsIndexes(cluster_indices)
saveAsPickle(cluster_indices, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/' + 'clusterIndices')
saveAsPickle(representative_indices, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/' + 'representative_indices')
saveAsPickle(clustersParticipantsList, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/' + 'clustersParticipantsList')
