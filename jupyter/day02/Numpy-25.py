import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
yu_array = np.random.randint(0,5,(10,3))
print(yu_array)
m = np.all(yu_array[:,1:] == yu_array[:,:-1],axis = 1)
print(m)
print(yu_array[~m])
