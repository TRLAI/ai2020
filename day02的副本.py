import numpy as np
import sys

# 1. 打印当前Numpy版本
# print(np.__version__)
#
# 2. 构造一个全零的矩阵，并打印其占用的内存大小
# py_matirx = np.zeros((3,3))
# print(py_matirx.nbytes)
#
# 3. 打印一个函数的帮助文档，比如numpy.add
# print(help(np.info(np.prod)))
#
# 4. 创建一个10-49的数组，并将其倒序排列
# a4 = np.arange(10,50)
# print(a[::-1])
#
# 5. 找到一个数组中不为0的索引
# a5 = np.arange(-5,5)
# print(np.nonzero(a5))
#
# 6. 随机构造一个3*3矩阵，并打印其中最大与最小值
# py_m6 = np.random.random((3,3))
# print(py_m6)
# print("max=",py_m6.max())
# print("min=",py_m6.min())
#
# 7. 构造一个5*5的矩阵，令其值都为1，并在最外层加上一圈0
# a7 = np.ones((5, 5))
# print(np.pad(a7, pad_width=1, mode='constant', constant_values=0))

# 8. 构建一个shape为（6，7，8）的矩阵，并找到第100个元素的索引值
# a8 = np.random.random((6,7,8))
# print(np.unravel_index(100,(6,7,8)))

# 9. 对一个5*5的矩阵做归一化操作
# a9 = np.random.randint(1,10,(5,5))
# print(a9)
# print(a9-a9.min()/a9.max()-a9.min())

# 10. 找到两个数组中相同的值
# a10 = np.random.randint(0,5,6)
# b10 = np.random.randint(0,5,6)
# print(a10,"\n",b10)
# print([x for x in a10 if x in b10])

# 11. 得到今天 明天 昨天的日期
# today = np.datetime64('today','D')
# tommorow = np.datetime64('today','D') + np.timedelta64('1','D')
# yesterday = np.datetime64('today','D') - np.timedelta64('1','D')
# print(today)
# print(tommorow)
# print(yesterday)

# 12. 得到一个月中所有的天
# a12 = np.arange(2020-1,2020-2, dtype="datetime64[D]")
# print(a12)

# 13. 得到一个数的整数部分
# a13 = np.random.random(5)
# print(np.floor(a13))

# 14. 构造一个数组，让它不能被改变
# a14 = np.arange(1,10)
# a14.flags.writeable = False

# 15. 打印大数据的部分值，全部值
# np.set_printoptions(threshold=5)
# a15 = np.zeros((10,10))
# print(a15)

# 16. 找到在一个数组中，最接近一个数的索引
# a16 = np.arange(100)
# b16 = np.random.uniform(100)
# print(b16)
# index = np.abs(a16-b16).argmin()
# print(a16[index])

# 17. 32位float类型和32位int类型转换
# a17 = np.arange(10,dtype = np.int64)
# print(a17.dtype)
# a17 = a17.astype(np.float64)
# print(a17.dtype)

# 18. 打印数组元素位置坐标与数值
# a18 = np.arange(9).reshape(3,3)
# for index,value in np.ndenumerate(a18):
#     print(index,value)

# 19. 按照数组的某一列进行排序
# a19 = np.random.randint(0,10,(3,3))
# print(a19)
# print(a19[a19[:,0].argsort()])

# 20. 统计数组中每个数值出现的次数
# a20 = np.random.randint(0,10,10)
# print(a20)
# print(np.bincount(a20))

# 21. 如何对一个四维数组的最后两维来求和
# a21 = np.random.randint(0, 10, (4, 4, 4, 4))
# print(a21.sum(axis=(-2, -1)))

# 22. 交换矩阵中的两行
# a22 = np.arange(36).reshape(6, 6)
# print(a22)
# a22[:, [0, 1]] = a22[:, [1, 0]]
# a22[0, 1] = a22[1, 0]
# print(a22)

# 23. 找到一个数组中最常出现的数字
# a23 = np.random.randint(0,10,5)
# print(a23)
# print(np.bincount(a23).argmax())

# 24. 快速查找TOP K
# a24 = np.arange(10000)
# np.random.shuffle(a24)
# print(a24)
# k = 5
# print(a24[np.argpartition(-a24, k)[:k]])

# 25. 去除掉一个数组中，所有元素都相同的数据
np.set_printoptions(threshold=sys.maxsize)
a25 = np.random.randint(0, 4, (10, 3))
print(a25)
s = np.all(a25[:, 1:] == a25[:, :-1], axis=1)
print(s)
print(a25[~s])
