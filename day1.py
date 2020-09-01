"""
T01
有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
"""
# count = 0
# for i in range(1, 5):
#     for j in range(1, 5):
#         for k in range(1, 5):
#             if i != j and j != k and i != k:
#                 count += 1
#                 print(i * 100 + j * 10 + k)
# print("结果是%d" % count)

"""
T02
企业发放的奖金根据利润提成。利润低于或等于10万元时，奖金可提10%；利润高于10万元，低于20万元时，低于10万元的部分按10%提成，
高于10万元的部分，可提成7.5%；20万到40万之间时，高于20万元的部分，可提成5%；40万到60万之间时高于40万元的部分，
可提成3%；60万到100万之间时，高于60万元的部分，可提成1.5%，高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，
求应发放奖金总数？    
"""

# bonus = 0
# profit = int(input("profit:"))
# if profit <= 10:
#     bonus += profit*0.1
# if 10 < profit <= 20:
#     bonus += 10*0.1 + (profit-10)*0.075
# if 20 < profit <= 40:
#     bonus += 10*0.1 + (20-10)*0.0075 + (profit-20)*0.005
# if 40 < profit <= 60:
#     bonus += 10*0.1 + (20-10)*0.075 + (40-20)*0.005 + (profit-40)*0.003
# if 60 < profit <= 100:
#     bonus += 10*0.1 + (20-10)*0.075 + (40-20)*0.005 + (60-40)*0.003 + (profit-60)*0.0015
# else:
#     bonus += 10*0.1 + (20-10)*0.075 + (40-20)*0.005 + (60-40)*0.003 + (100-60)*0.0015 + (profit-100)*0.001

"""
T03
输入三个整数x,y,z，请把这三个数由小到大输出
"""
# x = input("x=")
# y = input("y=")
# z = input("z=")
# list = []
# list.append(x)
# list.append(y)
# list.append(z)
# list.sort()
# print(list)

"""
T04
将一个列表的数据复制到另一个列表中
"""
# list1 = [5, 7, 9]
# list2 = list1
# print(list2)
# list3 = list1.copy()
# print(list3)
# list4 = [x for x in list1]
# print(list4)

"""
T05
暂停一秒输出,并格式化当前时间。使用 time 模块的 sleep() 函数。
"""
# import time
#
# for i in range(10):
#     if i == 3:
#         time.sleep(1)
#     print(i)

"""
T06
打印出所有的"水仙花数"，所谓"水仙花数"是指一个三位数，其各位数字立方和等于该数本身。
例如：153是一个"水仙花数"，因为153=1的三次方＋5的三次方＋3的三次方。
"""
# for i in range(100, 1000):
#     x = i // 100
#     y = (i // 10) % 10
#     z = i % 10
#     if x**3 + y**3 + z**3 == i:
#         print(i)

"""
T07
输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。
"""

# s = input("输入字符串：")
# letter = 0
# intger = 0
# space = 0
# other = 0
# for i in s:
#     if 'a' <= i <= 'z' or 'A' <= i <= 'Z':
#         letter += 1
#     elif i in '0123456789':
#         intger += 1
#     elif i == ' ':
#         space += 1
#     else:
#         other += 1
# print("英文字母为：%d，空格为：%d，数字为：%d，其他为：%d" % (letter, space, intger, other))

"""
T08
一球从100米高度自由落下，每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？
"""

# meter = 100
# s = meter # 第一次下落的距离
# for i in range(9):
#     s += meter
#     meter /= 2
# print(s)  # 总距离
# print(meter/2)  # 第十次
