len_letters = 0
len_blanks = 0
len_nums = 0
len_others = 0
input_strings = input('请输入一串字符：')
for i in input_strings:
    if ((i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z')):
        len_letters += 1
    elif (i >= '0' and i <= '9'):
        len_nums += 1
    elif (i == ' '):
        len_blanks += 1
    else:
        len_others += 1
print('此字符串有：\n字母：{0}个\n数字：{1}个\n空格：{2}个\n其他字符：{3}个'.format(len_letters, len_nums, len_blanks, len_others))
