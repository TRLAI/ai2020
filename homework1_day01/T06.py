#打印出所有的"水仙花数"，所谓"水仙花数"是指一个三位数，其各位数字立方和等于该数本身。例如：153是一个"水仙花数"，因为153=1的三次方＋5的三次方＋3的三次方。
list = []
for b in range(1,10):
    for s in range(10):
        for g in range(10):
            if b ** 3 + s ** 3 + g ** 3 == b*100 + s * 10 + g:
                num = b * 100 + s * 10 + g
                list.append(num)
            else:
                continue
print('水仙花数为：',list)