import numpy as np
import pandas as pd
from Bio import pairwise2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pickle


def calculate_identity(seqA, seqB):
    """
    :param seqA: The sequence of amino acid from chain A
    :param seqB: The sequence of amino acid from chain B
    :return: percentage of identity between the sequences
    """
    score = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True, score_only=True)
    # min_len = min(len(seqA), len(seqB))
    # identity = score / min_len
    max_len = max(len(seqA), len(seqB))
    identity = score / max_len
    return identity


def keep_only_chars(string):
    return ''.join([char for char in string if char.isalpha()])


def listCreationUtil(fullName):
    pdbName = fullName[0:4]
    chainsString = fullName.split('_')[1]
    chainsStringsList = chainsString.split('+')
    chainsNamesList = [keep_only_chars(chainString) for chainString in chainsStringsList]
    pdbNamesWithChainsList = [pdbName + chainName for chainName in chainsNamesList]
    return pdbName, pdbNamesWithChainsList


def createDictionaries(namesList, sizesList, sequenceLists, fullNamesList, pdbNamesWithChainsLists):
    structuresDicts = {}
    for i in range(len(fullNamesList)):
        structureDict = {}
        structureDict['pdbName'] = namesList[i]
        structureDict['size'] = sizesList[i]
        structureDict['sequenceList'] = sequenceLists[i]
        structureDict['pdbNamesWithChainsList'] = pdbNamesWithChainsLists[i]
        structuresDicts[fullNamesList[i]] = structureDict
    return structuresDicts


def listCreation(filename):
    """
    :param filename: PSSM file
    :return: tuple(namesList,sizesList)
    namesList = list of all the chains's name in the file
    sizesList = list of all the chains's number of amino acids in the file
    """
    namesList = []
    fullNamesList = []
    pdbNamesWithChainsLists = []
    sizesList = []
    sequenceLists = [[]]
    file1 = open(filename, 'r')
    lastChainName = ''
    line = file1.readline().split()
    cnt = 0
    seq = ''
    while len(line) != 0:  # end of file
        cnt += 1
        if len(line) == 1:  # in chain header line
            sequenceLists[len(sequenceLists) - 1].append(seq)
            sizesList.append(cnt)
            fullName = line[0][1:]
            fullNamesList.append(fullName)
            pdbName, pdbNamesWithChainsList = listCreationUtil(fullName)
            namesList.append(pdbName)
            pdbNamesWithChainsLists.append(pdbNamesWithChainsList)
            try:
                if len(pdbNamesWithChainsLists) > 1:
                    assert (len(sequenceLists[len(sequenceLists) - 1]) == len(
                        pdbNamesWithChainsLists[len(pdbNamesWithChainsLists) - 2]))
                    assert (sizesList[len(sizesList) - 1]) == sum(
                        [len(seq) for seq in sequenceLists[len(sequenceLists) - 1]])
            except:
                print(pdbNamesWithChainsList)
                print(sequenceLists[len(sequenceLists) - 1])
                raise Exception(pdbName)
            sequenceLists.append([])
            cnt = -1
            seq = ''
            lastChainName = ''
        else:
            if lastChainName != line[0]:  # switching chains
                lastChainName = line[0]
                if len(seq) != 0:
                    sequenceLists[len(sequenceLists) - 1].append(seq)
                seq = ''
            seq = seq + line[2]  # not chain's name
        line = file1.readline().split()
    sizesList.append(cnt)
    sequenceLists[len(sequenceLists) - 1].append(seq)
    sizesList = sizesList[1:]  # first one is redundent
    sequenceLists = sequenceLists[1:]
    file1.close()
    return namesList, sizesList, sequenceLists, fullNamesList, pdbNamesWithChainsLists


def make_cath_df(filename, columns_number):
    """
    :param filename: cath-domain-list file
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: dataframe of all the chains in the file and their cath classification divide to 4 different columns
    """

    df = pd.read_csv(filename, skiprows=16, header=None, delimiter=r"\s+")
    df = df.iloc[:, 0:columns_number + 1]
    cath_columns = ["n" + str(i) for i in range(1, columns_number + 1)]
    df.columns = ['chain'] + cath_columns
    df['chain'] = df['chain'].apply(lambda x: x[0:5])
    return df


def make_cath_df_new(filename, columns_number):
    """
    :param filename: cath-domain-list file
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: dataframe of all the chains in the file and their cath classification divide to 4 different columns
    """
    file = open("cath_b.20230204.txt", 'r')
    lines = file.readlines()
    structuresNames = [line[0:5] for line in lines]
    structuresNumbers = [line[5:7] for line in lines]
    c0 = [line.split(" ")[2].split(".")[0] for line in lines]
    c1 = [line.split(" ")[2].split(".")[1] for line in lines]
    c2 = [line.split(" ")[2].split(".")[2] for line in lines]
    c3 = [line.split(" ")[2].split(".")[3] for line in lines]
    data = {
        'chain': structuresNames,
        'number': structuresNumbers,
        'c0': c0,
        'c1': c1,
        'c2': c2,
        'c3': c3,
    }
    df = pd.DataFrame(data)
    print(df)
    return df


def getAllCathClassificationsForChain(cath_df, chainName, columns_Number):
    df = cath_df[cath_df['chain'] == chainName]
    myList = df.values.tolist()
    onlyClassificationList = [l[2:2 + columns_Number] for l in myList]
    return onlyClassificationList


def addClassificationsForDict(cath_df, structuresDicts, columns_Number):
    for key in structuresDicts.keys():
        structureDict = structuresDicts[key]
        classificationsLists = []
        for i in range(len(structureDict['pdbNamesWithChainsList'])):
            if structureDict['inOrNotInCathList'][i]:
                classificationsLists.append(
                    getAllCathClassificationsForChain(cath_df, structureDict['pdbNamesWithChainsList'][i],
                                                      columns_Number))
            else:
                classificationsLists.append(None)
        structureDict['classificationsLists'] = classificationsLists


def findChainsInCath(cath_df, structuresDicts):
    setOfchainsNames = set(cath_df['chain'])
    for fullpdbName, structureDict in structuresDicts.items():
        inOrNotInCathList = [structureDict['pdbNamesWithChainsList'][j] in setOfchainsNames for j in
                             range(len(structureDict['pdbNamesWithChainsList']))]
        structureDict['inOrNotInCathList'] = inOrNotInCathList


def DivideToStructuresInAndNotInCath(cath_df, structuresDicts):
    setOfchainsNames = set(cath_df['chain'])
    inCath = []
    notInCath = []
    for fullpdbName, structureDict in structuresDicts.items():
        sequencesInCath = []
        pdbNamesWithChainsIn = []
        for i in range(len(structureDict['pdbNamesWithChainsList'])):
            if structureDict['pdbNamesWithChainsList'][i] in setOfchainsNames:
                sequencesInCath.append(structureDict['sequenceList'][i])
                pdbNamesWithChainsIn.append(structureDict['pdbNamesWithChainsList'][i])
        structureDict['sequencesInCath'] = sequencesInCath
        structureDict['pdbNamesWithChainsIn'] = pdbNamesWithChainsIn
        if len(pdbNamesWithChainsIn) >= 1:
            inCath.append(fullpdbName)
        else:
            notInCath.append(fullpdbName)
    return inCath, notInCath


def countInCath(cath_df, structuresDicts):
    setOfchainsNames = set(cath_df['chain'])
    cnt = 0
    inCath = []
    notInCath = []
    for fullpdbName, structureDict in structuresDicts.items():
        sequencesInCath = []
        pdbNamesWithChainsIn = []
        isInCath = False
        for i in range(len(structureDict['pdbNamesWithChainsList'])):
            if structureDict['pdbNamesWithChainsList'][i] in setOfchainsNames:
                isInCath = True
                break
        if isInCath:
            cnt += 1
            inCath.append(fullpdbName)
        else:
            notInCath.append(fullpdbName)
    return inCath, notInCath, cnt


def neighbor_mat(df, nameList, seqList, columns_number):
    """
    :param df: cath data frame as it return from the func make_cath_df
    :param lst: list of chains
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: matrix. mat[i][j] == 1 if there is connection between chain i and chain j
    """
    # generate the graph using CATH.
    cath_columns = ["n" + str(i) for i in range(1, columns_number + 1)]
    not_in_cath = set()
    n = len(nameList)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarity = df[df['chain'].isin([nameList[i], nameList[j]])]
            if len(similarity['chain'].unique()) == 1:
                if (similarity['chain'].unique()[0] == nameList[i]):
                    not_in_cath.add(nameList[j])
                else:
                    not_in_cath.add(nameList[i])
            else:
                similarity = similarity.groupby(by=cath_columns)
                for name, group in similarity:
                    if len(group['chain'].unique()) == 2:
                        mat[i][j] = mat[j][i] = 1
                        break
    # calculate the sequence identity
    for i in range(n):
        for j in range(i + 1, n):
            if (nameList[i] in not_in_cath):
                score = calculate_identity(seqList[i], seqList[j])
                mat[i][j] = mat[j][i] = 1
    return mat


def comapreClassifications(c1, c2):
    for i in range(len(c1)):
        if c1[i] != c2[i]:
            return False
    return True


def isSimiliarChains(structureDict1, structureDict2):
    connected = False
    for k in range(len(structureDict1['pdbNamesWithChainsList'])):
        for l in range(len(structureDict2['pdbNamesWithChainsList'])):
            if structureDict1['inOrNotInCathList'][k] and structureDict2['inOrNotInCathList'][l]:  # both chains in cath
                allClassifications1 = structureDict1['classificationsLists'][k]
                allClassifications2 = structureDict2['classificationsLists'][l]
                for c1 in allClassifications1:
                    for c2 in allClassifications2:
                        if comapreClassifications(c1, c2):
                            connected = True
            else:  # at least one of the chains not in cath
                if calculate_identity(structureDict1['sequenceList'][k], structureDict2['sequenceList'][l]) > 0.5:
                    connected = True
    return connected


def neighbor_mat_new(structuersDictionaries):
    """
    :param df: cath data frame as it return from the func make_cath_df
    :param lst: list of chains
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: matrix. mat[i][j] == 1 if there is connection between chain i and chain j
    """
    # generate the graph using CATH.
    n = len(structuersDictionaries)
    print(n)
    structuersDictionariesValues = [structuersDictionaries[key] for key in structuersDictionaries.keys()]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            structureDict1 = structuersDictionariesValues[i]
            structureDict2 = structuersDictionariesValues[j]
            if isSimiliarChains(structureDict1, structureDict2):
                mat[i][j] = mat[j][i] = 1
    return mat


def createRelatedChainslist(numberOfComponents, labels):
    """
    :param numberOfComponents: number of component = x => 0<=label values<x
    :param labels: labels
    :return: RelatedChainslist: RelatedChainslist[i] = list of all the chain index's which has the label i
    """
    relatedChainsLists = [[] for _ in range(numberOfComponents)]
    for i in range(len(labels)):
        relatedChainsLists[labels[i]].append(i)
    return relatedChainsLists


def createClusterSizesList(relatedChainslists, sizeList):
    """
    :param relatedChainslists:  relatedChainslist[i] = list of all the chain index's which has the label i
    :param relatedChainslists:  sizeList- list of all the chains's size
    :return: list of tuples (clusterIndex,size)
    """
    clusterSizes = []
    for i in range(len(relatedChainslists)):
        my_sum = 0
        for index in relatedChainslists[i]:
            my_sum += sizeList[index]
        clusterSizes.append((i, my_sum))
    return clusterSizes


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


def clusterToChainList(clusterId, relatedChainsLists, nameList):
    """
    :param clusterId: list of chain indexs
    :param relatedChainsLists: relatedChainslist[i] = list of all the chain index's in cluster i
    :param  nameList = list of all the chains's name in the file
    :return: chainList = list of all chain names in the cluster
    """
    cluster = relatedChainsLists[clusterId]  # get the chains in the cluster
    chainList = [nameList[i] for i in cluster]
    return chainList


def sublistsToChainLists(sublists, relatedChainsLists, nameList):
    """
    :param sublists: sublists[i] = all the clusters in sublist i
    :param relatedChainsLists: relatedChainslist[i] = list of all the chain index's in cluster i
    :return: chainLists: ChainLists[i] = list of all the chains in cluster i
    """
    chainLists = [[] for i in range(len(sublists))]
    for i in range(len(sublists)):
        for clusterId in sublists[i]:
            chainLists[i] += clusterToChainList(clusterId, relatedChainsLists, nameList)
    return chainLists


def chainListsToChainIndexDict(chainLists):
    """
    :param chainLists: chainLists[i] = list of all the chains in cluster i
    :return: chainDict: chainDict[chainName] = index of chain cluster(i if chain in ChainLists[i])
    """
    chainDict = {}
    for i in range(len(chainLists)):
        for chainName in chainLists[i]:
            chainDict[chainName] = i
    return chainDict


def dividePSSM(chainDict):
    """
    :param chainDict: chainDict[chainName] = index of chain cluster(i if chain in ChainLists[i])
    create len(chainLists) txt files. the i txt file contains the chains in chainLists[i]
    """
    dirPath = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\PssmFiles_propagated_asa\PssmFiles_propagated_asa_0_15'
    filesList = [open(dirPath + "\\PSSM{}.txt".format(i), 'w') for i in range(5)]
    fullPssmFilePath = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\FullPropagatedPssmWithAsa\\propagatedPssmWithAsaFile0_15.txt'
    pssmFile = open(fullPssmFilePath, 'r')
    lines = pssmFile.readlines()
    fillIndex = -1  # fillIndex = i -> we now write to PSSMi.txt
    for line in lines:
        if line[0] == '>':  # header line
            fillIndex = chainDict[line[1:len(line)-1]]
        filesList[fillIndex].write(line)
    for i in range(5):
        filesList[i].close()
    pssmFile.close()


def calclulateIndexesOfScc(homologousLabels, sccNumber):
    indexes = np.where(homologousLabels == sccNumber)
    return indexes[0]


def calculateRatioFromIndexes(matHomologous, indexes):
    selected_matrix = matHomologous[np.ix_(indexes, indexes)]
    avarageNumberOfEdgesForStructure = (np.sum(selected_matrix) / len(indexes))
    ratio = avarageNumberOfEdgesForStructure / len(indexes)
    return ratio


def calculateHomologousRatioForSCC(homologousLabels, matHomologous, sccNumber):
    indexes = calclulateIndexesOfScc(homologousLabels, sccNumber)
    return calculateRatioFromIndexes(matHomologous, indexes)


def calculateRatioForFold(homologousLabels, matHomologous, sublists, foldNum):
    indexesLists = [calclulateIndexesOfScc(homologousLabels, sccNumber) for sccNumber in sublists[foldNum]]
    totalIndexes = np.concatenate(indexesLists)
    ratio = calculateRatioFromIndexes(matHomologous, totalIndexes)
    return ratio


cath_df = make_cath_df_new("cath_b.20230204.txt", 4)
namesList, sizesList, sequenceList, fullNamesList, pdbNamesWithChainsLists = listCreation("propagatedFullPssmFile")
structuresDicts = createDictionaries(namesList, sizesList, sequenceList, fullNamesList, pdbNamesWithChainsLists)
inCath, notInCath, cnt = countInCath(cath_df, structuresDicts)

# print(structuresDicts)
print(len(structuresDicts))
print(cnt)
print(inCath)
print(notInCath)
# print(cath_df)
findChainsInCath(cath_df, structuresDicts)
addClassificationsForDict(cath_df, structuresDicts, 4)
matHomologous = neighbor_mat_new(structuresDicts)
graphHomologous = csr_matrix(matHomologous)
homologous_components, homologousLabels = connected_components(csgraph=graphHomologous, directed=False,
                                                               return_labels=True)
print(namesList)
print(sizesList)
print(sum(sizesList))
print(homologous_components)
print(homologousLabels)
print("Done2")
relatedChainsLists = createRelatedChainslist(homologous_components, homologousLabels)
print("Done3")
clusterSizes = createClusterSizesList(relatedChainsLists, sizesList)
print("Done4")
sublists, sublistsSum = divideClusters(clusterSizes)
print("Done5")

print(relatedChainsLists)
print(clusterSizes)
print(sublists)
print(sublistsSum)

chainLists = sublistsToChainLists(sublists, relatedChainsLists, fullNamesList)
chainDict = chainListsToChainIndexDict(chainLists)
print(chainLists)
print(chainDict)
# pickleDirPath = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel'
# with open(pickleDirPath + "\\receptorsFoldsDict.pkl" , "wb") as f:
#     # pickle the list to the file
#     pickle.dump(chainDict, f)
#



# print(calculateHomologousRatioForSCC(homologousLabels, matHomologous, 0))
# for i in range(5):
#     print("avarage ratio of fold : ", i + 1)
#     print(calculateRatioForFold(homologousLabels, matHomologous, sublists, i))
dividePSSM(chainDict)
