import numpy as np
array = np.random.randint(0,10,(4,4,4,4))
res = array.sum(axis=(-2,-1))
print(res)