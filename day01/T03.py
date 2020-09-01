#输入三个整数x,y,z，请把这三个数由小到大输出

list = []
num = input('请输入第一个数字：')
list.append(num)
num0 = input('请输入第二个数字：')
list.append(num0)
num1 = input('请输入第三个数字：')
list.append(num1)
list = sorted(list)
print(list)