import re


def plotUbiquitinBindingResiduesHistogram(fileName):
    file = open(fileName, 'r')
    lines = file.readlines()
    cntList = [0 for _ in range(75)]
    for line in lines:
        onlyUbiqline = line.split("$")[3]
        if onlyUbiqline == '\n':
            continue
        ubiqs = onlyUbiqline.split('//')
        for ubiq in ubiqs:
            residues = ubiq.split('+')
            residueNumbers = [''.join(char for char in residue if char.isdigit()) for residue in residues]
            for num in residueNumbers:
                cntList[int(num) - 1] += 1
    print(cntList)

plotUbiquitinBindingResiduesHistogram('NewAllSummaryContent')