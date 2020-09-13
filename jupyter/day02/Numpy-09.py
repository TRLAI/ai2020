import numpy as np
array = np.random.random((4,4))
max = array.max()
min = array.min()
array = (array - min)/(max - array)
print(array)