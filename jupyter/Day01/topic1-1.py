nums = [1, 2, 3, 4]
count = 0
for i in nums:
    for j in nums:
        for k in nums:
            if (i != j and i != k and j != k):
                print(i * 100 + j * 10 + k)
                count += 1
print('能组{0}个互不成相同且无重复数字的三位数'.format(count))
