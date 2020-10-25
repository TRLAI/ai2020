import numpy as np
array = np.random.randint(0,10,(3,3))
print(array)
print(array[array[:,0].argsort()])