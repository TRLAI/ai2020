a = 0
b = 0
c = 0
print('水仙花数如下：')
for i in range(100, 1000):
    a = (i // 100) ** 3
    b = ((i // 10) % 10) ** 3
    c = (i % 10) ** 3
    k = a + b + c
    if (k == i):
        print(i)
