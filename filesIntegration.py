import os

def batchIntegrator(index):
    dir = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\asaBatches\\asaBatch' + str(index)
    filesList = [open(dir + '\\' + file_path, 'r') for file_path in os.listdir(dir) if file_path != ('summaryLog.txt')]
    # pssmfile = open(
    #    dir + '\\' + 'pssmFile' + str(index), 'w')
    asafile = open(
       dir + '\\' + 'asaFile' + str(index), 'w')
    allFilesContent = []
    for file in filesList:
        fileContent = file.read()
        allFilesContent += fileContent
    allFilesString = ''.join(allFilesContent)
    # pssmfile.write(allFilesString)
    # pssmfile.close()
    asafile.write(allFilesString)
    asafile.close()

    for file in filesList:
        file.close()



# batchIntegrator(36)


def integrateDifferentBatches(i):
    dir = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\asaBatches'
    pssmFiles = [open(dir + '\\asaBatch' + str(index) + '\\' + 'asaFile' + str(index)) for index in range(i)]
    # pssmFiles = [open(dir + '\\Batch' + str(index) + '\\' + 'pssmFile' + str(index)) for index in range(i)]
    pssmContents = [file.read() for file in pssmFiles]
    for file in pssmFiles:
        file.close()
    # summaryFiles = [open(dir + '\\asaBatch' + str(index) + '\\' + 'summaryLog.txt') for index in range(i)]
    # summaryContents = [file.read() for file in summaryFiles]
    # for file in summaryFiles:
    #     file.close()
    allPssmContent = ''.join(pssmContents)
    # allSummaryContent = '\n'.join(summaryContents)
    allPssmContentFile = open('FullAsaPssmContent', 'w')
    allPssmContentFile.write(allPssmContent)
    allPssmContentFile.close()
    # allsummaryContentFile = open('FullSummaryContent', 'w')
    # allsummaryContentFile.write(allSummaryContent)
    # allsummaryContentFile.close()


integrateDifferentBatches(40)
