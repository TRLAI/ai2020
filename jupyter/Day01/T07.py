'''
T07
输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。
'''
s = input("输入字符串：")
c = {'letter': 0, 'integer': 0, 'space': 0, 'other': 0}
for i in s:
    if i > 'a' and i < 'z' or i > 'A' and i < 'Z':
        c['letter'] += 1
    elif i in '0123456789':
        c['integer'] += 1
    elif i == ' ':
        c['space'] += 1
    else:
        c['other'] += 1

print('统计字符串：', s)
print(c)