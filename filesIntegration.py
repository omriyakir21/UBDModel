import os

def batchIntegrator(index):
    dir = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\Batch' + str(index)
    filesList = [open(dir + '\\' + file_path, 'r') for file_path in os.listdir(dir) if file_path != ('summaryLog.txt')]
    pssmfile = open(
       dir + '\\' + 'pssmFile' + str(index), 'w')
    allFilesContent = []
    for file in filesList:
        fileContent = file.read()
        allFilesContent += fileContent
    allFilesString = ''.join(allFilesContent)
    pssmfile.write(allFilesString)
    pssmfile.close()
    for file in filesList:
        file.close()


# for i in range(0,7):
#     batchIntegrator(i)
# for i in range(8,13):
#     batchIntegrator(i)
for i in range(40):
    batchIntegrator(i)


def integrateDifferentBatches(i):
    dir = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel'
    pssmFiles = [open(dir + '\\Batch' + str(index) + '\\' + 'pssmFile' + str(index)) for index in range(i)]
    pssmContents = [file.read() for file in pssmFiles]
    for file in pssmFiles:
        file.close()
    summaryFiles = [open(dir + '\\Batch' + str(index) + '\\' + 'summaryLog.txt') for index in range(i)]
    summaryContents = [file.read() for file in summaryFiles]
    for file in summaryFiles:
        file.close()
    allPssmContent = ''.join(pssmContents)
    allSummaryContent = '\n'.join(summaryContents)
    allPssmContentFile = open('FullPssmContent', 'w')
    allPssmContentFile.write(allPssmContent)
    allPssmContentFile.close()
    allsummaryContentFile = open('FullSummaryContent', 'w')
    allsummaryContentFile.write(allSummaryContent)
    allsummaryContentFile.close()


integrateDifferentBatches(40)
