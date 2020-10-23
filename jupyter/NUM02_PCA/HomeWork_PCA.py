import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('iris.data')
# print(df.head())

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
print(df.head())

# 分割数据集
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# print(X)
# print(y)

# 标签索引
# label_dict = {1: 'Iris-Setosa',
#               2: 'Iris-Versicolor',
#               3: 'Iris-Virgnica'}
feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}
plt.figure(figsize=(8, 6))
# 绘图
for cnt in range(4):
    plt.subplot(2, 2, cnt + 1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        plt.hist(X[y == lab, cnt],
                 label=lab,
                 bins=10,
                 alpha=0.3, )
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()
print(plt.show())

# 原始矩阵
X_std = StandardScaler().fit_transform(X)
# print (X_std)

# 协方差矩阵
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('Covariance matrix: \n%s' % cov_mat)

print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))

# 求解特征值和特征向量
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors: \n%s' % eig_vecs)
print('Eigenvalues: \n%s' % eig_vals)

# 降序排列特征值
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
print(eig_pairs)
print('----------')
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)

# 主成分分析
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 选取最大的个特征值所对应的特征向量组成构成矩阵
matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                      eig_pairs[1][1].reshape(4, 1)))

print('Matrix W:\n', matrix_w)

# 与原始矩阵相乘
Y = X_std.dot(matrix_w)
print(Y)

# 绘图
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(X[y==lab, 0],
                X[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
