import numpy as np
yu_array = np.arange(25).reshape(5,5)
yu_array[:,[0,1]] = yu_array[:,[1,0]]
yu_array[[0,1]] = yu_array[[1,0]]
print(yu_array)