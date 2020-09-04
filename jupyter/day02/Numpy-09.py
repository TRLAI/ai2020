import numpy as np
array = np.random.random((5,5))
max = array.max()
min = array.min()
array = (array - min)/(max - array)
print(array)