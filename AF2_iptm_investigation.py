import json
import numpy as np
import os
from matplotlib import pyplot as plt
import path
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

def get_max_iptm_value(uniprotId):
    # Define the directory path
    dir_path = os.path.join(path.AF2_multimerDir, f"output/{uniprotId}")

    # Check if the directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist")

    # Search for all JSON files containing "seed" in their filenames
    json_files = [file_name for file_name in os.listdir(dir_path) if "seed" in file_name and file_name.endswith(".json")]

    if not json_files:
        raise FileNotFoundError(f"No JSON files containing 'seed' found in the directory for {uniprotId}")

    # Initialize a variable to store the maximum iptm value
    max_iptm_value = float('-inf')

    # Iterate through the found JSON files and update the max iptm value
    for json_file in json_files:
        json_file_path = os.path.join(dir_path, json_file)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            iptm_value = data.get("iptm")

            if iptm_value is None:
                raise ValueError(f"The 'iptm' property does not exist in the JSON file {json_file}")

            max_iptm_value = max(max_iptm_value, float(iptm_value))

    # Check if any valid iptm value was found
    if max_iptm_value == float('-inf'):
        raise ValueError("No valid 'iptm' values found in the JSON files")

    return max_iptm_value

# Example usage:
# uniprotId = "your_uniprot_id_here"
# print(get_max_iptm_value(uniprotId))


def plotAF2IptmPredictorPlot():
    selected_keys = utils.loadPickle(os.path.join(path.AF2_multimerDir, 'selected_keys_'+str(200)+'.pkl'))
    dict_sources = allPredictions['dict_sources']
    labels = np.array([0 if dict_sources[key] in NegativeSources else 1 for key in selected_keys])
    predictions =  np.array([get_iptm_value(key) for key in selected_keys])
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
plotAF2IptmPredictorPlot()