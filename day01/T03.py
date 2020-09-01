#输入三个整数x,y,z，请把这三个数由小到大输出
x = int(input("Enter 1st Number: "))
y = int(input("Enter 2nd Number: "))
z = int(input("Enter 3rd Number: "))

if x > y:
    x, y = y, x
if x > z:
    x, z = z, x
if y > z:
    y, z = z, y

print(x, y, z)