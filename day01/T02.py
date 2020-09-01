#企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，奖金可提10%；利润高于10万元
# -*- coding: utf-8 -*-
"""
Spyder Editor


This is a temporary script file.
"""
num=input("Please input your number:(unit is million)\n")
num2=float(num)
if num2<=0:
    print("Your number is error!")
elif num2<=10:
    print ("You have %f million RMB."%(num2*1.1),end=" ")
elif num2<=20:
     print ("You have %f million RMB."%(11+(num2-10)*1.075),end=" ")
elif num2<=40:
    print ("You have %f million RMB."%(21.75+(num2-20)*1.05),end=" ")
elif num2<=60:
    print ("You have %f million RMB."%(42.75+(num2-40)*1.03),end=" ")
elif num2<=100:
    print ("You have %f million RMB."%(63.35+(num2-60)*1.015),end=" ")
else:
    print ("You have %f million RMB."%(103.95+(num2-100)*1.01),end=" ")