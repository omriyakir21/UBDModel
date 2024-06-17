linux = True
if linux:
    mainProjectDir = '/home/iscb/wolfson/omriyakir/UBDModel/'
    aggregateFunctionMLPDir = '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP'
    GoPath = '/home/iscb/wolfson/omriyakir/UBDModel/GO'
    predictionsToDataSetDir = '/home/iscb/wolfson/omriyakir/UBDModel/predictionsToDataSet'
    ScanNetPredictionsPath = '/home/iscb/wolfson/omriyakir/UBDModel/model_predictions/'
    gridSearchDir = ('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP'
                     '/MLP_MSA_val_AUC_stoppage_with_evolution_50_plddt_all_organizems_15_4')
    trainingDir = ('/home/iscb/wolfson/omriyakir/UBDModel/predictionsToDataSet'
                   '/with_evolution_50_plddt_all_organizems_15_4/trainingDicts/')
    ScanNetDB = '/home/iscb/wolfson/omriyakir/UBDModel/ScanNetDB/'
    AF2_multimerDir = '/home/iscb/wolfson/omriyakir/UBDModel/AF2_multimer/'

else:
    mainProjectDir = r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel'
    aggregateFunctionMLPDir = r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP'
    GoPath = r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\GO'
    predictionsToDataSetDir = r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\predictionsToDataSetDir'
    ScanNetPredictionsPath = r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\Predictions'
    gridSearchDir = (
        r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP'
        r'\MLP_MSA_val_AUC_stoppage_with_evolution_50_plddt_all_organizems_15_4')
    trainingDir = (r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\predictionsToDataSet'
                   r'\with_evolution_50_plddt_all_organizems_15_4\trainingDicts')
