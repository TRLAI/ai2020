a = [100]
h = 100
print('第1次从%s米高落地，走过%s米，之后又反弹至%s米。'%(h,a[0],h/2))
for i in range(2,11):
    a.append(h)
    h = h/2
    print('第%s次从%s米高落地，共走过%s米，之后又反弹至%s米.'%(i,h,sum(a),h/2))