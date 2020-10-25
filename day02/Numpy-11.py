import numpy as np
today = np.datetime64('today','D')
yesterday = np.datetime64('today','D') - np.timedelta64('1','D')
tommorrow = np.datetime64('today','D') + np.timedelta64('1','D')
print(today)
print(yesterday)
print(tommorrow)