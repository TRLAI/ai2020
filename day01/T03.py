lst = []
x = int(input("请输入x的值为："))
y = int(input("请输入y的值为："))
z = int(input("请输入y的值为："))
def minNumbwe(x,y,z):
    if x < y :
        if x < z:
            lst.append(x)
        else:
            lst.append(z)
    else:
        if y < z:
            lst.append(y)
        else:
            lst.append(z)
def maxNumber(x,y,z):
    if x > y:
        if x > z:
            lst.append(x)
        else:
            lst.append(z)
    else:
        if y > z:
            lst.append(y)
        else:
            lst.append(z)
minNumbwe(x,y,z)
maxNumber(x,y,z)
if lst[0] == x:
    if lst[1] == y:
        lst.insert(1,z)
    else:
        lst.insert(1,y)
elif lst[0] == y:
    if lst[1] == z:
        lst.insert(1,x)
    else:
        lst.insert(1,z)
elif list[0] == z:
    if lst[1] == x:
        lst.insert(1,y)
    else:
        lst.insert(1,x)
print(lst)