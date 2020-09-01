bounce = 0
down = 100
long = 100
input_count = int(input('请输入要查询的次数：'))
if input_count == 1:
    long = down
for i in range(2, input_count + 1):
    bounce = down / 2
    down = bounce
    long = long + bounce + down
    i += 1
print('第{0}次落地时，共经过{1}米，第{2}次反弹{3}米'.format(input_count, long, input_count, bounce / 2))
