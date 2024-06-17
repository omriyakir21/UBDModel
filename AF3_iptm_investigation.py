import csv
import pickle
import sys
from itertools import chain
from plistlib import load

import networkx as nx
import numpy as np
import os
import networkx
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import pandas as pd
import aggregateScoringMLPUtils as utils
from Bio.PDB import MMCIFParser
import path
import proteinLevelDataPartition
import seaborn as sns
from sklearn.metrics import auc
import aggregateScoringMLPUtils as utils

allPredictions = utils.loadPickle(os.path.join(path.ScanNetPredictionsPath, 'all_predictions_0304_MSA_True.pkl'))
# allPredictions = utils.loadPickle(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\Predictions\batch_predictions_interface_MSA_True_0_20.pkl')
NegativeSources = utils.NegativeSources
dict_sources = allPredictions['dict_sources']
dict_predictions_ubiquitin = allPredictions['dict_predictions_ubiquitin']
# labels = np.array([0 if dict_sources[key] in NegativeSources else 1 for key in dict_sources.keys()])
ubiqString = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'


def select_n_random_indices(n):
    keys = np.array(list(dict_sources.keys()))
    indices = np.random.choice(keys.size, n, replace=False)
    selected_keys = keys[indices]
    return selected_keys


def createInputFiles(n):
    selected_keys = utils.loadPickle(os.path.join(path.AF2_multimerDir, 'selected_keys_'+str(n)+'.pkl'))
    for uniprot_id in selected_keys:
        sequence = allPredictions['dict_sequences'][uniprot_id]
        fasta_content = f">{uniprot_id}\n{sequence}:\n{ubiqString}\n"
        inputDir = os.path.join(path.AF2_multimerDir, 'input')
        filename = os.path.join(inputDir, f"{uniprot_id}.fasta")
        with open(filename, "w") as f:
            f.write(fasta_content)


# selected_keys = select_n_random_indices(200)
# utils.saveAsPickle(selected_keys, os.path.join(path.AF2_multimerDir, 'selected_keys_200'))
createInputFiles(200)