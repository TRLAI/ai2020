import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy.random
import time

STOP_ITER = 0 # 根据迭代次数, 达到迭代次数及停止
STOP_COST = 1 # 根据损失, 损失差异较小及停止
STOP_GRAD = 2 # 根据梯度, 梯度变化较小则停止
#sigmoid 函数
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
#停止策略
def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold

#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

#核心操作函数
def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解

    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh): break

    return theta, i - 1, costs, grad, time.time() - init_time
#辅助工具函数
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)  # 这是最核心的代码

    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    # 根据传入参数选择下降方式
    if batchSize==n: strDescType = "Gradient"  #  批量
    elif batchSize==1:  strDescType = "Stochastic"  # 随机
    else: strDescType = "Mini-batch ({})".format(batchSize) # 小批量
    name += strDescType + " descent - Stop: "
    # 根据 停止方式进行选择处理生成 name
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    # 画图
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta

path = 'data/LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
pdData.head()
pdData.shape   #(100, 3) 数据维度 100 行, 3列

positive = pdData[pdData['Admitted'] == 1] # returns the subset of rows such Admitted = 1, i.e. the set of *positive* examples
negative = pdData[pdData['Admitted'] == 0] # returns the subset of rows such Admitted = 0, i.e. the set of *negative* examples

fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle('图形展示 - 散点图',fontproperties='SimHei')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend() # 展示图形标识
ax.set_xlabel('Exam 1 Score') # x 标识
ax.set_ylabel('Exam 2 Score') # y 标识

nums = np.arange(-10, 10, step=1) #creates a vector containing 20 equally spaced values from -10 to 10
fig, ax = plt.subplots(figsize=(12,4))
fig.suptitle('sigmoid 函数实现',fontproperties='SimHei')
ax.plot(nums, sigmoid(nums), 'r')

pdData.insert(0, 'Ones', 1) # 偏置项参数的处理


# set X (training data/训练数据) and y (target variable/变量参数)
orig_data = pdData.iloc[:,:].values # 转换为 数组形式
cols = orig_data.shape[1]
X = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
#X = np.matrix(X.values)
#y = np.matrix(data.iloc[:,3:4].values) #np.array(y.values)
theta = np.zeros([1, 3])  # 参数占位, 先用 0 填充

#批量下降 - 迭代次数停止策略
n=100
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

#批量下降 - 损失值停止策略
runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)

#批量下降 - 梯度值停止策略
runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)


plt.show()