import numpy as np
# 1. 打印当前Numpy版本

print('当前Numpy版本为：',np.__version__,'\n')

# 2. 构造一个全零的矩阵，并打印其占用的内存大小

py_array = np.array([[1,2,3],[4,5,6]])
size = py_array.size * py_array.itemsize
print('全零矩阵为：\n',np.zeros_like(py_array),'\n其占用大小为：\n',size,'\n')

# 3. 打印一个函数的帮助文档，比如numpy.add

print(help(np.info(np.add)),'\n')

# 4. 创建一个10-49的数组，并将其倒序排列

py_array0 = np.arange(10,50)
py_array1 = py_array0[::-1]
print('倒序矩阵为：\n',py_array1,'\n')

# 5. 找到一个数组中不为n0的索引

py_array2 = np.arange(10)
py_array3 = np.nonzero(py_array2)
print('不为0的索引数组：\n',py_array3,'\n')

# 6. 随机构造一个3*3矩阵，并打印其中最大与最小值

py_array4 = np.empty([3,3])
print('\n',py_array4,py_array4.min(),py_array4.max(),'\n')

# 7. 构造一个5*5的矩阵，令其值都为1，并在最外层加上一圈0

py_array5 = np.ones([5,5])
py_array5 = np.pad(py_array5,pad_width=1, mode='constant',constant_values = 0)
print(py_array5,'\n')

# 8. 构建一个shape为（6，7，8）的矩阵，并找到第100个元素的索引值

print(np.unravel_index(100,(6,7,8)),'\n')
# py_array40 = np.random.randint(10,size=(6,7,8))
# print(py_array40)

# 9. 对一个5*5的矩阵做归一化操作

# py6_array = np.random.random((5,5))
# yu_max = py6_array.max()
# yu_min = py6_array.min()
# py6_array = (py6_array-yu_min)/(yu_max-yu_min)
# py6_array

# 10. 找到两个数组中相同的值

yu_array = np.random.randint(0,5,8)
xu_array = np.random.randint(0,9,8)
print(yu_array)
print(xu_array)
print(np.intersect1d(yu_array,xu_array),'\n')

# 11. 得到今天 明天 昨天的日期

today = np.datetime64('today','D')
yesterday = today - np.timedelta64('1','D')
tomorrow = today + np.timedelta64('1','D')
print('今天：\n',today,'昨天：\n',yesterday,'明天\n',tomorrow,'\n')

# 12. 得到一个月中所有的天

py_array11 = np.arange('2020-09','2020-10',dtype = 'datetime64[D]')
print('九月：\n',py_array11,'\n')

# 13. 得到一个数的整数部分

py_array12 = np.random.uniform(0,10,10)
print(np.floor(py_array12),'\n')

# 14. 构造一个数组，让它不能被改变

# py_array13 = np.ones(10)
# py_array13.flags.writeable = False
# py_array13[0] = 1
# print(py_array13,'\n')

# 15. 打印大数据的部分值，全部值

import sys
np.set_printoptions(threshold=5) #打印部分
np.set_printoptions(threshold=sys.maxsize) #打印全部
py_array14 = np.ones((10,10))
print(py_array14)

# 16. 找到在一个数组中，最接近一个数的索引

py_array15 = np.arange(1000)
py_array16 = np.random.uniform(0,1000) #指定一个随机数
print(py_array16)
index = (np.abs(py_array15-py_array16)).argmin()
print(py_array15[index])

# 17. 32位float类型和32位int类型转换

py_array17 = np.arange(10,dtype=np.int32)
py_array17 = py_array17.astype(np.float32)
print(py_array17.dtype)

# 18. 打印数组元素位置坐标与数值

py_array18 = np.arange(9).reshape(3,3)
for index,value in np.ndenumerate(py_array18):
    print(index,value)

# 19. 按照数组的某一列进行排序

py_array19 = np.random.randint(0,10,(3,3))
print(py_array19)
print(py_array19[py_array19[:,0].argsort()])

# 20. 统计数组中每个数值出现的次数

py_array20 = np.array([1,7,1,1,6,5,0,2,2,2])
print(np.bincount(py_array20))

# 21. 如何对一个四维数组的最后两维来求和

py_array21 = np.random.randint(0,100,(8,8,4,8))
py_array22 = py_array21.sum(axis = (-2,-1))
print(py_array22)

# 22. 交换矩阵中的两行

py_array23 = np.arange(16).reshape(4,4)
print('原矩阵：\n',py_array23,'\n')
py_array23[:,[0,2]] = py_array23[:,[2,0]]#交换两列
py_array23[[0,2]] = py_array23[[2,0]]#交换两行
print('交换后的矩阵：\n',py_array23)

# 23. 找到一个数组中最常出现的数字

py_array24 = np.random.randint(0,10,10)
print(py_array24)
print(np.bincount(py_array24).argmax())

# 24. 快速查找TOP K

py_array25 = np.arange(50)
# np.random.shuffle(yu_array)
py_array25.sort()
k = 2
print(py_array25[np.argpartition(-py_array25,k)[:k]])

# 25. 去除掉一个数组中，所有元素都相同的数据

np.set_printoptions(threshold=sys.maxsize)
py_array26 = np.random.randint(0,5,(10,3))
print(py_array26)
m = np.all(py_array26[:,1:] == py_array26[:,:-1],axis = 1)
#矩阵比对true去掉
print(py_array26[~m])