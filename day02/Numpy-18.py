import numpy as np
array = np.arange(9).reshape(3,3)
for index,value in np.ndenumerate(array):
    print(index,value)