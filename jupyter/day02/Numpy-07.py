import numpy as np
array = np.ones((4,4))
array = np.pad(array,pad_width=1 ,mode='constant',constant_values= 0)
print(array)