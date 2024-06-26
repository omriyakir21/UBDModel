import csv
import pickle
import sys
from itertools import chain
from plistlib import load
import json
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

def get_iptm_value(uniprotId):
    # Define the directory path
    dir_path = os.path.join(path.AF2_multimerDir,f"output/{uniprotId}")

    # Check if the directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist")

    # Search for the JSON file containing "rank_001" in its filename
    json_file = None
    for file_name in os.listdir(dir_path):
        if "rank_001" in file_name and file_name.endswith(".json"):
            json_file = file_name
            break

    if json_file is None:
        raise FileNotFoundError("No JSON file containing 'rank_001' found in the directory")

    # Construct the full path to the JSON file
    json_file_path = os.path.join(dir_path, json_file)

    # Read the JSON file and retrieve the "iptm" value
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        iptm_value = data.get("iptm")

        if iptm_value is None:
            raise ValueError("The 'iptm' property does not exist in the JSON file")

    return float(iptm_value)

def plotAF2IptmPredictorPlot():
    selected_keys = utils.loadPickle(os.path.join(path.AF2_multimerDir, 'selected_keys_'+str(200)+'.pkl'))
    dict_sources = allPredictions['dict_sources']
    labels = np.array([0 if dict_sources[key] in NegativeSources else 1 for key in selected_keys])
    predictions =  np.array(get_iptm_value(key) for key in selected_keys)
    precision, recall, thresholds = utils.precision_recall_curve(labels, predictions)
    sorted_indices = np.argsort(recall)
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    aucScore = auc(sorted_recall, sorted_precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision Recall curve (auc = {aucScore:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AF2 iptm based predictor')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path.AF2_multimerDir, 'iptmPredictor'))
    plt.close()


# Example usage:
# uniprotId = "your_uniprot_id_here"
# print(get_iptm_value(uniprotId))

# selected_keys = select_n_random_indices(200)
# utils.saveAsPickle(selected_keys, os.path.join(path.AF2_multimerDir, 'selected_keys_200'))
# createInputFiles(200)