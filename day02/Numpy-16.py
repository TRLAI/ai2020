import numpy as np
array1 = np.arange(100)
array2 = np.random.uniform(1,100)
print(array2)
index = (np.abs(array1-array2)).argmin()
print(array1[index])