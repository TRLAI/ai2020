# 1
# import numpy
# print(numpy.__version__)


# 2
import numpy as np
# z=np.zeros((4,4))
# print(z)
# print('%d bytes ' %(z.size*(z.itemsize)))

# 3
# import numpy as np
# print(help(np.info(np.add)))


# 4
# a_array=np.arange(10,50,1)
# print(a_array)
# b_array=a_array[::-1]
# print(b_array)


# 5
# a=[1,2,3,4,5,6,0,7,8,0,9]
# b=np.nonzero(a)
# print(b)


# 6
# a_array=np.random.random((3,3))
# print(a_array.min())
# print(a_array.max())


# 7
# a_array=np.ones((5,5))
# a_array2=np.pad(a_array,pad_width=1,mode='constant',constant_values=0)
# print(a_array2)


# 8
# a=np.unravel_index(100,(6,7,8))
# print(a)


# 9
# a_array=np.random.random((5,5))
# print(a_array)
# a_max=a_array.max()
# a_min=a_array.min()
# a_array=(a_array-a_min)/(a_max-a_min)
# print(a_array)


# 10
# a1=np.random.randint(0,10,10)
# a2=np.random.randint(0,10,10)
# print(a1)
# print(a2)
# print(np.intersect1d(a1,a2))


# 11
# yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
# today = np.datetime64('today', 'D')
# tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
# print(yesterday)
# print(today)
# print(tomorrow)


# 12
# print(np.arange('2020-09','2020-10',dtype='datetime64[D]'))


# 13
# a=np.random.uniform(0,10,10)
# print(a)
# print(np.floor(a))


# 14
# a=np.zeros(5)
# a.flags.writeable=False
# print(a)


# 15?
# np.set_printoptions(threshold=3)
# a=np.zeros((7,7))
# print(a)


# 16
# z=np.arange(100)
# v=np.random.uniform(0,100)
# print(v)
# index=(np.abs(z-v)).argmin()
# print(z[index])


# 17
# z=np.arange(10,dtype=np.int32)
# print(z.dtype)
# z=z.astype(np.float32)
# print(z.dtype)


# 18
# z=np.arange(9).reshape(3,3)
# for index,value in np.ndenumerate(z):
#     print(index, value)


# 19
# z=np.random.randint(0,10,(3,3))
# print(z)
# print(z[z[:1].argsort()])


# 20?
# z=np.array([1,1,1,2,2,3,3,4,5,8,])
# a=np.bincount(z)
# print(a)


# 21
# z=np.random.randint(0,10,(4,4,4,4))
# print(z)
# res=z.sum(axis=(-2,-1))
# print(res)


# 22
# z=np.arange(25).reshape(5,5)
# print(z)
# z[[0,1]]=z[[1,0]]
# print(z)


# 23
# z=np.random.randint(0,10,50)
# print(z)
# print(np.bincount(z).argmax())


# 24
# z=np.arange(1000)
# np.random.shuffle(z)
# n=5
# print(z[np.argpartition(-z,n)[:n]])


# 25
# np.set_printoptions(threshold=100000)
# z=np.random.randint(0,2,(10,3))
# print(z)