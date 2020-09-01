#一球从100米高度自由落下，每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？
init_high = 100
total_times = 10
total_distance = []
high = init_high

while len(total_distance) < total_times:
    total_distance.append(high * 2)
    high /= 2

print(sum(total_distance) - init_high)
print(high)