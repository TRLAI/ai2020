#水仙花数
for a in range(1,10):
    for b in range(0,10):
        for c in range(0,10):
            s1= a*100+b*10+c
            s2= pow(a,3)+pow(b,3)+pow(c,3)
            if s1==s2:
                print('水仙花:%4d'%s1)