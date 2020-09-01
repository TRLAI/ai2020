high = 100
sum_meter = 0
for i in range(10):
    if i == 0:
        sum_meter += high
        high /= 2
else:
    sum_meter += (2*high)
    high /= 2
    print('-------------------------------')
    print('第十次落地可以反弹的高度为：%f米' %(high))
    print('一共经过了%f米' %(sum_meter))
