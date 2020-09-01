# 1
# count = 0
# for i in range(1, 5):
#     for j in range(1, 5):
#         for n in range(1, 5):
#             if (i != j) and (i !=n ) and (j != n):
#                 print("%d%d%d" % (i, j, n), end='  ')
#                 count += 1
#     print('')
# print('共有：%s个' % count)


# 2
# I = int(input("当月利润，单位为万元："))
# if I <= 10n
#     money = 10 * 0.1
# elif 10 < I <= 20:
#     money = 10 * 0.1 + (I-10) * 0.075
# elif 20 < I <= 40:
#     money = 10 * 0.1 + 10 * 0.075 + (I-20) * 0.05
# elif 40 < I <= 60:
#     money = 10 * 0.1+ 10 * 0.075 + 20 * 0.05 + (I-40) * 0.03
# elif 60 < I <= 100:
#     money = 10 * 0.1+ 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + (I-60) * 0.015
# elif I > 100:
#     money = 10 * 0.1 + 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + 40 * 0.015 + (I-100) * 0.015
# print(money,'万元')


# 3
# i = int(input("第一个数: "))
# j = int(input("第二个数: "))
# n = int(input("第三个数: "))
#
# if i > j:
#     i, j = i, j
# if i > n:
#     i, n = i, n
# if j > z:
#     j, n = j, n
#
# print(i, j, n)


# 4
# def copylist(list1):
#     list_copy = list(list1)
#     return list_copy
#
#
# list1 = [1, 2, 3, 4, 5]
# list2 = copylist(list1)
# print(list2)


# 5
# import time
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# time.sleep(1)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# 6
# for i in range(100,1000):
#     a = int(i/100)
#     b = int(i/10)-a*10
#     c = i - a*100 - b*10
#     if i == a**3+b**3+c**3:
#         print(i)


# 7
# n = input("输入一行字符：")
# space = 0
# word = 0
# num = 0
# other = 0
# for i in n:
#     if i.isalpha():
#         word += 1
#     elif i.isdigit():
#         num += 1
#     elif i.isspace():
#         space += 1
#     else:
#         other += 1
# print('space=',space,'word=', word, 'num=',num, 'other=',other)


# 8
# a = [100]
# h = 100
# print('第1次从%s米高落地，走过%s米，之后又反弹至%s米。' % (h, a[0], h/2))
# for i in range(2,11):
#     a.append(h)
#     h = h / 2
#     print('第%s次从%s米高落地，共走过%s米，之后又反弹至%s米。' % (i, h, sum(a), h / 2))
