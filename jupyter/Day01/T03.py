'''
T03
输入三个整数x,y,z，请把这三个数由小到大输出
'''
list = []
print("输入第一个整数：")
list.append(int(input()))
print("输入第二个整数：")
list.append(int(input()))
print("输入第三个整数：")
list.append(int(input()))
list.sort()
print(list)