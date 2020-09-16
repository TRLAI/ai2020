import LR
import pdb

filename = 'testSet.txt'
dataMat, labelsMat = LR.loadData(filename)
weights = LR.gradAscent(dataMat, labelsMat)
LR.plotBestFit(weights, filename) 