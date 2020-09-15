# 学习要点

# 文本数据分析

## 相似度计算

### 数据预处理

1. 语料清洗
   1. 删除停用词
   2. 去重（好、顶、赞、感谢楼主）
2. 分词（针对中文）

# 分类与聚类

|                    | 标签（目标）                                                 | 参数                                                       |
| ------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| 监督学习（分类）   | 1.可以监督，尝试各种算法，让预测值不断逼近真实值<br/>2.可以使用各类的评估，基于预测值和真实值，比如交叉验证 | 有“标准答案”，可以往标准答案的方向，进行调参               |
| 无监督学习（聚类） | 1.无标签，需要根据相似度，把数据分类<br/>2.由于没有标签，评估就比较难做了 | 没有“标准答案”，每次调节参数之后，都会出现不同的分类（簇） |

# 聚类算法对比

![Cluster](images\Cluster.png)

# 降维

1. 维度过多，上千维的数据中，当要运算出一个结果，需要大量的算力和时间。在有限的时间里，不一定能跑出一个结果。
2. 上千个维度中，不是所有的维度，都有用。留下有用的维度。

## 讲维度的视频

https://youtu.be/gg85IH3vghA

https://www.bilibili.com/video/BV1hs41147Ae

https://www.bilibili.com/video/BV1Ja4y1t7zL

# 机器学习相关算法小结

1. 监督
   1. 决策树（Decision Trees）
   2. XGBoost
   3. 朴素贝叶斯分类(Naive Bayesian classification)
   4. 最小二乘法（Ordinary Least Squares Regression）
   5. 逻辑回归(Logistic Regression)
   6. 线性回归（Linear Regression）
   7. 支持向量机（Support Vector Machine，SVM）
   8. 集成算法（Ensemble methods）
   9. 随机森林 (Random Forest)
   10. 线性判别分析(Linear Discriminant Analysis)
   11. Adaboost
   12. CNN (Convolutional Neural Network)
2. 无监督
   1. 聚类算法（Clustering Algorithms）
      1. 基于密度聚类(Mean Shift)
      2. 基于密度聚类(DBSCAN)
   2. 高斯混合模型(GMM)与EM
   3. 主成分分析（Principal Component Analysis，PCA）
   4. 奇异值分解（Singular Value Decomposition，SVD）
   5. 独立成分分析（Independent Component Analysis，ICA）