import LRgradAscent
import pdb

filename = 'testSet.txt'
dataMat, labelsMat = LRgradAscent.loadData(filename)
weights = LRgradAscent.gradAscent(dataMat, labelsMat)
LRgradAscent.plotBestFit(weights, filename)