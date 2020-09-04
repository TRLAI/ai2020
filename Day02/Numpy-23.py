import numpy as np
array = np.random.randint(0,10,5)
print(array)
print(np.bincount(array).argmax())