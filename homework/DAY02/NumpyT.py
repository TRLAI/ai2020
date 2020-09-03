import numpy as np

# T01
print(np.__version__)

# T02
matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
print(matrix)
print(matrix.size*matrix.itemsize,"bytes")

# T03
print(help(np.info(np.add)))

# T04
array_1 = np.arange(10, 50, 1)
print(array_1)
array_1 = array_1[::-1]
print(array_1)

# T05
print(np.nonzero([0, 12, 0, 145, 14, 0, 0, 0, 186, 0]))

# T06
array_2 = np.random.random((3, 3))
print(array_2)
print(array_2.max())
print(array_2.min())

# T07
array_3 = np.ones([5, 5], dtype=int)
array_4 = np.pad(array_3, ((1, 1), (1, 1)), 'constant', constant_values=0)
print(array_4)

# T08
array_5 = np.zeros([6, 7, 8],dtype=int)
print(array_5.shape)
value_1 = np.unravel_index(100, (6, 7, 8))
print(value_1)

# T09
array_6 = np.random.randint(20,100,size=(5,5))
print(array_6)
array_6_max = array_6.max()
array_6_min = array_6.min()
array_6 = (array_6-array_6_min)/(array_6_max-array_6_min)
print(array_6)

# T10
array_7 = np.random.randint(0, 10, 10)
array_8 = np.random.randint(0, 10, 10)
print(array_7)
print(array_8)
array_9 = np.intersect1d(array_7, array_8)
print(array_9)

# T11
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday)
print(today)
print(tomorrow)

# T12
print(np.arange('2019-09', '2019-10', dtype ='datetime64[D]'))

# T13
array_10 = np.random.uniform(0, 100, 3)
print(array_10)
array_11 = np.floor(array_10)
print(array_11)

# T14
array_20 = np.zeros(5)
array_20.flags.writeable=False
print(array_20)

# T15
array_12 = np.ones([15, 15])
print(array_12)
np.set_printoptions(threshold=5)
print(array_12)

# T16
array_13 = np.arange(100)
v = np.random.uniform(0, 100)
print(v)
index=(np.abs(array_13-v)).argmin()
print(array_13[index])

# T17
array_14 = np.arange(10,dtype=np.int32)
print(array_14.dtype)
array_15 = array_14.astype(np.float32)
print(array_15.dtype)

# T18
array_16 = np.random.random((3, 3))
for index,value in np.ndenumerate(array_16):
    print(index,value)

# T19
array_17 = np.random.randint(0, 100, (5,5))
print(array_17)
print(array_17[:, 0])
print(array_17[:, 0].argsort())
print(array_17[array_17[:, 0].argsort()])

# T20
array_18 = np.random.randint(0,100,10)
c = 0
for i in np.bincount(array_18):
    print("%d的数量:" %(c),i)
    c += 1
print(array_18)
print(np.bincount(array_18))

# T21
array_19 = np.random.randint(0,10,(2,2,2,2))
print(array_19)
res = array_19.sum(axis=(2,3))
print(res)

# T22
array_21 = np.random.randint(0, 100, [5, 5])
print(array_21)
array_21[[0, 4]] = array_21[[4, 0]]
print(array_21)

# T23
array_22 = np.random.randint(0,10,74)
print(array_22)
print(np.bincount(array_22))
print(np.bincount(array_22).argmax())

# T24
array_23 = np.random.randint(1,100,10)
k = 5
print(array_23)
for i in range(k):
    print(array_23.max())
    array_23 = np.delete(array_23, array_23.argmax())

# T25
array_24 = np.random.randint(0,2,[10,3])
print(array_24)
index_1 = 0
for array_25 in array_24:
    if array_25[0] == array_25[1] and array_25[1] == array_25[2]:
        array_24 = np.delete(array_24, index_1, axis=0)
        index_1 -= 1
    index_1 += 1
print(array_24)