import numpy as np


print(np.version)
yu_array = np.zeros((5,4))
print('%d bytes'%(yu_array.size * yu_array.itemsize))
print(help(np.info(np.add)))
yu_array = np.arange(10,50)
yu_array[::-1]
'''
array([49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
       32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
       15, 14, 13, 12, 11, 10])
'''
yu_array = np.arange(5)
np.nonzero(yu_array)
yu_array = np.random.random((3,3))
yu_array.min()
yu_array.max()
yu_array = np.ones((5,5))
yu_array = np.pad(yu_array,pad_width=1, mode='constant',constant_values = 0)
yu_array
'''
array([[0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 1., 1., 1., 0.],
       [0., 1., 1., 1., 1., 1., 0.],
       [0., 1., 1., 1., 1., 1., 0.],
       [0., 1., 1., 1., 1., 1., 0.],
       [0., 1., 1., 1., 1., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0.]])
'''
np.unravel_index(100,(6,7,8))
yu_array = np.random.random((5,5))
yu_max = yu_array.max()
yu_min = yu_array.min()
yu_array = (yu_array-yu_min)/(yu_max-yu_min)
yu_array
'''
array([[0.48397315, 0.91683086, 0.47196491, 1.        , 0.36369439],
       [0.93613393, 0.25536399, 0.92875343, 0.31834473, 0.59442395],
       [0.6512316 , 0.68949923, 0.70313301, 0.8288644 , 0.82369357],
       [0.27935272, 0.41088302, 0.78561219, 0.3246256 , 0.79369801],
       [0.21235418, 0.        , 0.13606257, 0.30524044, 0.22690585]])
'''
yu_array = np.random.randint(0,10,8)
xu_array = np.random.randint(0,10,8)
print(yu_array)
print(xu_array)
np.intersect1d(yu_array,xu_array)
'''
[1 7 6 7 1 7 1 9]
[9 9 1 6 3 3 8 3]
array([1, 6, 9])
'''
today = np.datetime64('today','D')
yesterday  = np.datetime64('today','D')-np.timedelta64('1','D')
tommorow = np.datetime64('today','D')+np.timedelta64('1','D')
'''
numpy.datetime64('2020-01-31')
numpy.datetime64('2020-01-30')
numpy.datetime64('2020-02-01')
'''
'''
array(['2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04',
       '2019-10-05', '2019-10-06', '2019-10-07', '2019-10-08',
       '2019-10-09', '2019-10-10', '2019-10-11', '2019-10-12',
       '2019-10-13', '2019-10-14', '2019-10-15', '2019-10-16',
       '2019-10-17', '2019-10-18', '2019-10-19', '2019-10-20',
       '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24',
       '2019-10-25', '2019-10-26', '2019-10-27', '2019-10-28',
       '2019-10-29', '2019-10-30', '2019-10-31'], dtype='datetime64[D]')
'''
yu_array = np.random.uniform(0,10,10)
np.floor(yu_array)
yu_array = np.zeros(5)
yu_array.flags.writeable = False
yu_array[0] = 1
'''
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-18-6a8d2efcd9b8> in <module>
      1 yu_array = np.zeros(5)
      2 yu_array.flags.writeable = False
----> 3 yu_array[0] = 1

ValueError: assignment destination is read-only
'''
import sys
np.set_printoptions(threshold=5) #打印部分
np.set_printoptions(threshold=sys.maxsize) #打印全部
yu_array = np.zeros((15,15))
yu_array
'''
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
'''
yu_array = np.arange(100)
xu_array = np.random.uniform(0,100) #指定一个随机数
print(xu_array)
index = (np.abs(yu_array-xu_array)).argmin()
print(yu_array[index])
'''
66.82479960861905
67
'''
yu_array = np.arange(10,dtype=np.int32)
yu_array = yu_array.astype(np.float32)
yu_array.dtype
yu_array = np.arange(9).reshape(3,3)
for index,value in np.ndenumerate(yu_array):
    print(index,value)
'''
(0, 0) 0
(0, 1) 1
(0, 2) 2
(1, 0) 3
(1, 1) 4
(1, 2) 5
(2, 0) 6
(2, 1) 7
(2, 2) 8
'''
yu_array = np.random.randint(0,10,(3,3))
print(yu_array)
print(yu_array[yu_array[:,0].argsort()])#第一列
'''
[[5 4 4]
 [1 9 6]
 [9 6 9]]
[[1 9 6]
 [5 4 4]
 [9 6 9]]
'''
yu_array = np.array([1,1,1,2,2,3,3,4,5,8])
np.bincount(yu_array)
yu_array = np.random.randint(0,10,(4,4,4,4))
res = yu_array.sum(axis = (-2,-1))
res
'''
array([[58, 63, 85, 87],
       [81, 70, 64, 60],
       [64, 79, 76, 91],
       [63, 78, 66, 89]])
'''
yu_array = np.arange(25).reshape(5,5)
yu_array[:,[0,1]] = yu_array[:,[1,0]]#交换两列
yu_array[[0,1]] = yu_array[[1,0]]#交换两行
yu_array
'''
array([[ 1,  0,  2,  3,  4],
       [ 6,  5,  7,  8,  9],
       [11, 10, 12, 13, 14],
       [16, 15, 17, 18, 19],
       [21, 20, 22, 23, 24]])
'''
yu_array = np.random.randint(0,10,5)
print(yu_array)
print(np.bincount(yu_array).argmax())
'''
[5 7 5 1 4]
5
'''
yu_array = np.arange(5000)
np.random.shuffle(yu_array)
k = 5
print(yu_array[np.argpartition(-yu_array,k)[:k]])
import sys
np.set_printoptions(threshold=sys.maxsize)
yu_array = np.random.randint(0,5,(10,3))
yu_array
'''
array([[3, 2, 3],
       [2, 2, 2],
       [0, 3, 2],
       [4, 3, 3],
       [0, 1, 4],
       [1, 0, 1],
       [4, 2, 1],
       [4, 4, 1],
       [1, 2, 4],
       [2, 1, 3]])
'''
m = np.all(yu_array[:,1:] == yu_array[:,:-1],axis = 1)
print(m)

yu_array[~m]
'''
array([[3, 2, 3],
       [0, 3, 2],
       [4, 3, 3],
       [0, 1, 4],
       [1, 0, 1],
       [4, 2, 1],
       [4, 4, 1],
       [1, 2, 4],
       [2, 1, 3]])
'''

