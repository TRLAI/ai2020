#将一个列表的数据复制到另一个列表中
list1 = [x for x in range(1,5)]
list2 = []
print(list1)
print(list2) #初始化

print('地址赋值------------------')
list2 = list1
print (list2)

print('调用函数-------------------')
list3 = []
list3 = list1.copy()
print(list3)
print('遍历列表赋值-------------\n')
list4 = []
for i in range(len(list1)): #数组直接用 list4[i] = list1[i] 是不可以的，因为数组的长度已经被限制
    list4.append(list1[i])
print(list4)

print('列表生成发---------------------')
a = [x for x in range(0,5)]
b = [i for i in a]
print(b)