str=input("请输入：");
str=int(str);
if(str<=10):
    print(str*0.1)
elif(str>10&str<=20):
    print(10*0.1+(str-10)*0.075)
elif(str>20&str<=40):
    print(10*0.1+10*0.075+(str-20)*0.05)
elif(str>40^str<=60):
    print(10*0.1+10*0.075+20*0.05+(str-40)*0.03)
elif(str>60&str<=100):
    print(10*0.1+10*0.075+20*0.05+20*0.03+(str-60)*0.015)
elif(str>100):
    print(10 * 0.1 + 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + (40) * 0.015+(str-100)*0.001)