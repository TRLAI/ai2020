# 1
# count = 0
# for x in range(1, 5):
#     for y in range(1, 5):
#         for z in range(1, 5):
#             if (x != y) and (x !=z ) and (y != z):
#                 print("%d%d%d" % (x, y, z), end='  ')
#                 count += 1
#     print('')
# print('最终结果为：%s个' % count)


# 2
# I = int(input("请输入当月利润，单位为万元："))
# if I <= 10:
#     jiangjin = 10 * 0.1
# elif 10 < I <= 20:
#     jiangjin = 10 * 0.1 + (I-10) * 0.075
# elif 20 < I <= 40:
#     jiangjin = 10 * 0.1 + 10 * 0.075 + (I-20) * 0.05
# elif 40 < I <= 60:
#     jiangjin = 10 * 0.1+ 10 * 0.075 + 20 * 0.05 + (I-40) * 0.03
# elif 60 < I <= 100:
#     jiangjin = 10 * 0.1+ 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + (I-60) * 0.015
# elif I > 100:
#     jiangjin = 10 * 0.1 + 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + 40 * 0.015 + (I-100) * 0.015
# print(jiangjin,'万元')


# 3
# x = int(input("请输入第一个数: "))
# y = int(input("请输入第二个数: "))
# z = int(input("请输入第三个数: "))
#
# if x > y:
#     x, y = y, x
# if x > z:
#     x, z = z, x
# if y > z:
#     y, z = z, y
#
# print(x, y, z)


# 4
# def copyli(li1):
#     li_copy = list(li1)
#     return li_copy
#
#
# li1 = [1, 2, 3, 4, 5]
# li2 = copyli(li1)
# print(li2)


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
# a = input("请输入一行字符：")
# space = 0
# alpha = 0
# num = 0
# etc = 0
# for i in a:
#     if i.isalpha():
#         alpha += 1
#     elif i.isdigit():
#         num += 1
#     elif i.isspace():
#         space += 1
#     else:
#         etc += 1
# print('space=',space,'alpha=', alpha, 'num=',num, 'etc=',etc)


# 8
# a = [100]
# h = 100
# print('第1次从%s米高落地，走过%s米，之后又反弹至%s米。' % (h, a[0], h/2))
# for i in range(2,11):
#     a.append(h)
#     h = h / 2
#     print('第%s次从%s米高落地，共走过%s米，之后又反弹至%s米。' % (i, h, sum(a), h / 2))
