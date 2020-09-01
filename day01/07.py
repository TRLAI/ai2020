str=input();
a=0;
b=0;
c=0;
d=0;
for i in range  (len(str)):
    if(str[i].isalpha()):
             a+=1;
    elif(str[i].isdigit()):
        b+=1;
    elif(str[i].isspace()):
        c+=1;
    else:d+=1;
print(a);
print(b);
print(c);
print(d);

