
import os
b = 0
print("输入当月利润：")
i = int(input())
if i <= 10:
    b = 0.1*i
elif i <= 20:
    b = 1+(i-10)*0.075
elif i <= 40:
    b = 1.75+(i-20)*0.05
elif i <= 60:
    b = 2.75+(i-40)*0.03
elif i <= 100:
    b = 3.35+(i-60)*0.015
else:
    b = 3.95+(i-100)*0.01
print(b)