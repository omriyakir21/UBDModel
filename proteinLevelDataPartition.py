import pickle
import subprocess
import sys

import pandas as pd
import numpy as np
import os
import path
from Bio.PDB import MMCIFParser


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


# allProteinsDict = loadPickle('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/allProteinInfo.pkl')

# saveAsPickle(sequences,r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\allProteinSequences')

# cluster_indices = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\clusterIndices.pkl')
# clustersParticipantsList = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\clustersParticipantsList.pkl')
# representative_indices = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\representative_indices.pkl')


def cluster_sequences(list_sequences, seqid=0.5, coverage=0.4, covmode='0'):
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


def createClusterParticipantsIndexes(clusterIndexes):
    clustersParticipantsList = []
    for i in range(np.max(clusterIndexes) + 1):
        clustersParticipantsList.append(np.where(clusterIndexes == i)[0])
    return clustersParticipantsList


# saveAsPickle(cluster_indices, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/dataForTraining23_3/' + 'clusterIndices')
# saveAsPickle(representative_indices, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/dataForTraining23_3/' + 'representative_indices')
# saveAsPickle(clustersParticipantsList, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/dataForTraining23_3/' + 'clustersParticipantsList')

def divideClusters(clusterSizes):
    """
    :param clusterSizes: list of tuples (clusterIndex,size)
    :return:  sublists,sublistsSum
    divide the list into 5 sublists such that the sum of each cluster sizes in the sublist is as close as possible
    """
    sublists = [[] for i in range(5)]
    sublistsSum = [0 for i in range(5)]
    clusterSizes.sort(reverse=True, key=lambda x: x[1])  # Sort the clusters by size descending order.
    for tup in clusterSizes:
        min_cluster_index = sublistsSum.index(min(sublistsSum))  # find the cluster with the minimal sum
        sublistsSum[min_cluster_index] += tup[1]
        sublists[min_cluster_index].append(tup[0])
    return sublists, sublistsSum


path2mafft = '/usr/bin/mafft'
dirName = sys.argv[1]
dirPath = os.path.join(path.predictionsToDataSetDir, dirName)


def create_x_y_groups(allPredictionsPath):
    allPredictions = loadPickle(os.path.join(path.ScanNetPredictionsPath, allPredictionsPath))
    trainingDictsDir = os.path.join(dirPath, 'trainingDicts')
    allInfoDict = loadPickle(os.path.join(trainingDictsDir, 'allInfoDict.pkl'))

    allProteinsDict = dict()
    allProteinsDict['x'] = allInfoDict['x_train'] + allInfoDict['x_cv'] + allInfoDict['x_test']
    allProteinsDict['y'] = np.concatenate((allInfoDict['y_train'], allInfoDict['y_cv'], allInfoDict['y_test']))

    uniprots = [info[1] for info in allProteinsDict['x']]
    sequences = [allPredictions['dict_sequences'][uniprot] for uniprot in uniprots]
    cluster_indices, representative_indices = cluster_sequences(sequences)
    clustersParticipantsList = createClusterParticipantsIndexes(cluster_indices)
    clusterSizes = [l.size for l in clustersParticipantsList]
    clusterSizesAndInedxes = [(i, clusterSizes[i]) for i in range(len(clusterSizes))]
    sublists, sublistsSum = divideClusters(clusterSizesAndInedxes)
    groupsIndexes = []

    for l in sublists:
        groupsIndexes.append(np.concatenate([clustersParticipantsList[index] for index in l]))

    y_groups = []
    x_groups = []
    for indexGroup in groupsIndexes:
        x = [allProteinsDict['x'][index] for index in indexGroup]
        y = allProteinsDict['y'][indexGroup]
        x_groups.append(x)
        y_groups.append(y)

    saveAsPickle(x_groups, os.path.join(trainingDictsDir, 'x_groups'))
    saveAsPickle(y_groups, os.path.join(trainingDictsDir, 'y_groups'))
    return x_groups, y_groups


create_x_y_groups('all_predictions_22_3.pkl')
