import numpy as np
array = np.zeros(5)
array.flags.writeable = False
array[0] = 1
print(array)