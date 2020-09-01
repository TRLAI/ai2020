for num in range(100,1000):

 a=num%10;
 b=int(num/10%10);
 c=int(num/100);
 if(a*a*a+b*b*b+c*c*c==num):
     print(num);
