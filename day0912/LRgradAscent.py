from numpy import *


def loadData(filepath):
    dataMat = []
    labels = []
    fr = open(filepath)
    for line in fr.readlines():
        str = line.strip().split('\t')
        dataMat.append([1.0, float(str[0]), float(str[1])])
        labels.append(int(str[2]))
    return mat(dataMat), mat(labels).transpose()


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatrix, labelsMatrix):
    n, m = shape(dataMatrix)
    weights = ones((m, 1))
    step = 0.001
    iter = 500
    for k in range(iter):
        value = sigmoid(dataMatrix * weights)
        chazhi = labelsMatrix - value
        grad = dataMatrix.transpose() * chazhi
        weights = weights + step * grad
    return weights


def plotBestFit(weights, filepath):
    import matplotlib.pyplot as plt
    # illustrate the samples
    dataMatrix, labelsMatrix = loadData(filepath)
    n, m = shape(dataMatrix)
    xcord1 = [];
    ycord1 = []  # store the coordination of sample having label 1
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelsMatrix[i]) == 1:
            xcord1.append(dataMatrix[i, 1])
            ycord1.append(dataMatrix[i, 2])
        else:
            xcord2.append(dataMatrix[i, 1])
            ycord2.append(dataMatrix[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # illustrate the classifying line
    min_x = min(dataMatrix[:, 1])[0, 0]
    max_x = max(dataMatrix[:, 2])[0, 0]
    y_min_x = (-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = (-weights[0] - weights[1] * max_x) / weights[2]  # here, sigmoid(wx = 0) so wo + w1*x1 + w2*x2 = 0
    plt.plot([min_x, max_x], [y_min_x[0, 0], y_max_x[0, 0]], '-g')
    plt.show()