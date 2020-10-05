# 深度学习

## The CIFAR-10 dataset

数据集的链接：http://www.cs.toronto.edu/~kriz/cifar.html

### 流程

1. 读取数据
   1. 随机5000训练集
   2. 500验证集
   3. 随机500测试集
2. 建立模型
   1. 第一部分，data层，读取数据
   2. 第二部分，隐层，使用ReLU作为激活函数
   3. 第三部分，直接输出，使用Softmax，进行多分类
3. 得到结果之后，使用反向传播，求出W，b最适合的值。
   1. 结果中，包含每个图片，在10个分类中的分数。
   2. 结果中的loss值，进行反向传播
   3. 通过迭代，得到最佳的W，b值

### 代码

完整的做一个项目，分模块来编写。包含：前向传播、反向传播、计算操作、模型函数更新

1. fc_net，搭建基础模型
2. data_utils，数据读取及处理
3. solver，实际工作代码，包括模型更新
4. two_layer_fc_net_start，最终进行训练和可视化展示
5. 其他几个文件，定义了一些工具函数

## NLP

### 使用wiki数据得步骤

1. 从https://dumps.wikimedia.org/backup-index.html，找到zhwiki，点击进入。
2. 下载zhwiki-20201001-pages-articles.xml.bz2，或者，下载下面的分包文件。
3. 使用process.py，进行转换，把xml文件转换成txt文件。
4. 使用opencc，把繁体转换成简体
5. 使用jieba，进行分词

```
# 使用process.py，目标文件为bz2文件，输出文件为text文件
python process 目标文件 输出文件

# 使用opencc，目标和输出文件都为text文件
opencc -i 目标文件 -o 输出文件 -c t2s.json
```