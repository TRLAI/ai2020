#输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。

a = 0
b = 0
num = 0
space = ' '
other = 0
s = list()
s = input('请输入任意一段字符串：')
list = list(s)#将字符串转换为list
print(list)
for i in list:
    if i >= 'a' and i <= 'z' or i >='A' and i<= 'Z':
        a += 1
    elif i == space :
        b += 1
    elif i.isdigit():
        num +=1
    else:
        other += 1
print('英文字母的个数为：',a,'空格的个数为：',b,'数字的个数为：',num,'其他符号的个数为：',other)