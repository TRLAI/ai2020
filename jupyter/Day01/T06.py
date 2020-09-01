

for i in range (100,1000):

    a = int(i/100)
    b = int(i/10%10)
    c = int(i%10)

    sum = a*a*a+b*b*b+c*c*c

    if  sum==i:
        print(i)
