# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import os
path = 'D:\dasishixun\LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1','Exam 2','Admitted'])
pdData.head()

pdData.shape


positive = pdData[pdData['Admitted']==1]
negative = pdData[pdData['Admitted']==0]

fig,ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=30,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=30,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

def sigmoid(z):
    return 1/(1 + np.exp(-z))

nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(nums, sigmoid(nums),'r')

def model(X,theta):
    return sigmoid(np.dot(X, theta.T))

pdData.insert(0, 'Ones', 1)
##将表格转换为矩阵
orig_data = pdData.values
cols = orig_data.shape[1]#cols=4列
X = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]
theta = np.zeros([1,3])

X[:5]
y[:5]
theta
X.shape, y.shape, theta.shape

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# model 函数 - 预测函数实现
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))
#损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)  # 因为会有三个结果, 因此这里进行预先的占位
    error = (model(X, theta) - y).ravel()  # h0(xi)-yi 象形... 这里吧前面的m分之一的那个负号移到了这里来去除负号
    for j in range(len(theta.ravel())):  # 求三次
        term = np.multiply(error, X[:, j])  # : 表示所有的行, j 表示第 j 列
        grad[0, j] = np.sum(term) / len(X)
    return grad


STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:   return value > threshold
    elif type == STOP_COST: return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold


import numpy.random
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y


def descent(data, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解
    init_time = time.time()
    i = 0#迭代次数
    k = 0#batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)#计算的梯度
    costs = [cost(X, y, theta)]#损失值
    
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize#取batch数量个数据
        if k>=n:
            k = 0
            X, y = shuffleData(data)#重新洗牌
        theta = theta - alpha*grad#参数更新
        costs.append(cost(X, y, theta))#计算新的损失
        i +=1
        
        if stopType == STOP_ITER:   value = i
        elif stopType == STOP_COST: value = costs
        elif stopType == STOP_GRAD: value = grad
        if stopCriterion(stopType, value, thresh):break
    return theta, i-1, costs, grad, time.time() - init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += "data - learning rate:{} -".format(alpha)
    if batchSize==n:strDescType = "Gradient"
    elif batchSize == 1: strDescType = "Stochastic"
    else: strDescType = "Mini-batch({})".format(batchSize)
    name += strDescType + "descent - Stop:"
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_ITER: strStop = "costs change < {}".format(thresh)
    else: strStop ="gradient norm < {}".format(thresh)
    name+=strStop
    print ("***{}\nTheta:{} - Iter:{} - Last cost:{:03.2f} - Duration:{:03.2f}s".format(
        name, theta, iter, costs[-1],dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)),costs,'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


n = 100


runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

#设定阈值1E-6，差不多需要110000次迭代
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
#设定阈值0.05，差不多需要40000次迭代
runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)

runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)

runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)

from sklearn import preprocessing as pp

scaled_data = orig_data.copy()
scaled_data[:,1:3] = pp.scale(orig_data[:,1:3])

runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)

runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)

runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)

runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)

#设定阈值
def predict (X, theta):
    return [1 if x >=0.5 else 0 for x in model(X, theta)]

scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
