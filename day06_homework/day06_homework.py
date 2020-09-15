#引包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

import os
# 读取data文件夹下txt文件
path = 'data' + os.sep + 'LogiReg_data.txt'
#按照老师提供的txt文件，pandas读取，转化为DataFrame结构
#设置列头：第一次成绩、第二次成绩、录取与否
pdData = pd.read_csv(path,header=None,names=['第一次成绩','第二次成绩','录取否'])
print(pdData.head())

#维度
print(pdData.shape)

# 正例
positive = pdData[pdData['录取否'] == 1]
# 负例
negative = pdData[pdData['录取否'] == 0]
plt.style.use(['ggplot','bmh'])
# 指定画图区域大小
fig, ax = plt.subplots(figsize=(10,5))
# 正例散点图，设置：标记大小，颜色，标记样式，示例名
ax.scatter(positive['第一次成绩'], positive['第二次成绩'], s=30, c='b', marker='o', label='录取')
# 负例散点图，设置：标记大小，颜色，标记样式，示例名
ax.scatter(negative['第一次成绩'], negative['第二次成绩'], s=30, c='r', marker='x', label='没录取')
# 显示示例
ax.legend()
# x,y轴的名称
print(ax.set_xlabel('Exam 1 Score'))
print(ax.set_ylabel('Exam 2 Score'))
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(X, theta):
    # 做一个矩阵乘法
    return sigmoid(np.dot(X, theta.T))


# 插入一列，全为1的值
pdData.insert(0, 'Ones', 1)
# print(pdData.values)
orig_data = pdData.values
cols = orig_data.shape[1]  # 第一维的长度
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
theta = np.zeros([1, 3])


# print(X)
# print(y)
# print(theta)


# 损失函数，X样本数据，y是标签，theta参数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


# print(cost(X, y, theta))


# 计算梯度函数
def gradient(X, y, theta):
    # 声明梯度的变量
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    # 每次的偏导，要求3此theta的偏导
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        # 计算梯度
        grad[0, j] = np.sum(term) / len(X)
    return grad


# 按照迭代次数停止
STOP_ITER = 0
# 根据损失停止
STOP_COST = 1
# 根据梯度停止
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    # 设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold  # 指定一个次数
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold  # 指定阈值
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold  # 指定阈值


import numpy.random


# 洗牌，打乱顺序，让有规律的数据无规律，增强函数得泛化能力
def shuffleData(data):
    # 使用random中的shuffle
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


import time


# 不同的梯度下降策略，消耗的时间不同，要看一下，时间对结果的影响
# data数据，theta参数，batchSize梯度策略，stopType停止策略，thresh策略对应的阈值，alpha学习率
def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解，先做初始化
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        # 根据不同策略，使用实际的x,y，进行梯度计算
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1
        # 判断，是否停止
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        # break，进行停止
        if stopCriterion(stopType, value, thresh): break

    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # import pdb; pdb.set_trace();
    # 初始化，求解
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    # 定义如何显示信息，根据参数，选择梯度下降方式以及停止策略
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    # 停止
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    # 展示与画图
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    plt.style.use(['ggplot', 'bmh'])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta


# 选择的梯度下降方法是基于所有样本的


n = 100
# 按照迭代次数STOP_ITER，迭代5000次thresh=5000，学习率alpha=0.000001
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

# 损失函数STOP_COST，两次之间小于thresh，
# runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)

# 梯度STOP_GRAD，阈值小于thresh=0.05
# runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

# 梯度STOP_GRAD，阈值小于thresh=0.05
# runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

# 1个样本
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)

# runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)

# 样本数16
# runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)




# 处理后的数据scaled_data
scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])


#
# runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)

# 样本100，使用梯度，阈值调整到thresh=0.02
# runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)

# theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
# runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)

# 设定阈值
def predict(X, theta):
    # 大于0.5被录取，小于0.5不被录取
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
# 精度，预测对了多少个
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))