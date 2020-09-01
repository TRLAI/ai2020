input_nums = []
while True:
    i = input('请输入数字,按q退出：')
    if i == 'q':
        break
    else:
        k = int(i)
        input_nums.append(k)
order_nums = sorted(input_nums)
print(order_nums)


