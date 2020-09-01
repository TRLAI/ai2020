
a=int(input('第一个数字:'))
b=int(input('第二个数字:'))
c=int(input('第三个数字:'))
print(str.format('输入的三个数为：{0} {1} {2}', a, b, c))
list1 = [a, b, c]
print(list1)
for i in range(len(list1)-1):
    for j in range(len(list1)-1-i):
        if list1[j] > list1[j+1]:
            list1[j],list1[j+1] = list1[j+1],list1[j]
print(list1)