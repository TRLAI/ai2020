list1 = [x for x in range(1,5)]
list2 = []
print(list1)
print(list2)

print('地址赋值------------------')
list2 = list1
print (list2)

print('调用函数-------------------')
list3 = []
list3 = list1.copy()
print(list3)
print('遍历列表赋值-------------\n')
list4 = []
for i in range(len(list1)):
    list4.append(list1[i])
print(list4)

print('列表生成---------------------')
a = [x for x in range(0,5)]
b = [i for i in a]
print(b)