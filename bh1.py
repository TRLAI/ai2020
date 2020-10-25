import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost_func(theta, x, y):
    x = np.matrix(x)
    y = np.matrix(y)
    theta = np.matrix(theta)
    z = x*theta.T
    pos = np.multiply(y, np.log(sigmoid(z)))
    neg = np.multiply(1-y, np.log(1-sigmoid(z)))
    return np.sum(-pos-neg)/len(x)


def grident(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    para_num = x.shape[1]
    grad = np.zeros(para_num)
    error = sigmoid(x*theta.T)-y
    for i in range(para_num):
        term = np.multiply(error, x[:, i])
        grad[i] = np.sum(term)/len(x)
    return grad


def f(x, y, theta, alpha, iters):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    for i in range(iters):
        theta = theta - alpha*grident(theta, x, y)
    return theta


def predict(theta, x):
    probability = sigmoid(x*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


path = os.getcwd()
data = pd.read_csv(path+'\\LogiReg_data.txt', names = ['exam1', 'exam2', 'admit'])

data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:,:cols-1]
x = np.array(x.values)
y = data.iloc[:,cols-1:cols]
y = np.array(y.values)
th = np.zeros(3)
th = np.array([-20,0.21,0.7])
th = f(x, y, th, 0.0001, 1000)
print(cost_func(th, x, y))
cnt = 0
predictions = predict(th, x)
for i in range(len(x)):
    if predictions[i] == y[i]:
        cnt+=1
print("正确率:",cnt/len(x)*100,'%')