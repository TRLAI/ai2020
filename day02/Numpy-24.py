import numpy as np
array = np.arange(5000)
np.random.shuffle(array)
k = 5
print(array[np.argpartition(-array,k)[:k]])