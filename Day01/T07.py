s = input('请输入字符串：')
dic = {'letter': 0, 'integer': 0, 'space': 0, 'other': 0}
for i in s:
    if i > 'a' and i < 'z' or i > 'A' and i < 'Z':
        dic['letter'] += 1
    elif i in '0123456789':
        dic['integer'] += 1
    elif i == ' ':
        dic['space'] += 1
    else:
        dic['other'] += 1

print('统计字符串：', s)
print(dic)
print('------------显示结果2---------------')
for i in dic:
    print('{0}={1}'.format(i, dic[i]))
print('------------显示结果3---------------')
for key, value in dic.items():
    print('{0}={1}'.format(key, value))