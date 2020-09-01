T01

void num()
{ int i,j,k;
  for(i=0;i<4;i++)
    for(j=0;j<4;j++)
      if(j!=i)
        for(k=0;k<4;k++)
          if(k!=i&&k!=j)  print(k+(j*10)+(i*100));
}


T02

void calcaus(int l,int s)
{ if(l<100000) return(s*0.1);
  elif(l<200000) return((s-100000)*0.075+10000);
  elif(l<400000) return((s-200000)*0.05);
  elif(l<600000) return((s-400000)*0.03);
  elif(l<1000000) return((s-600000)*0.015);
  else return((s-1000000)*0.01);
}


T03

void sort(int x,int y,int z)
{
int a[3],i,min,k;
a[0]=x;a[1]=y;a[2]=z;
min=x;
for(i=0;i<3;i++)
  if(a[i]<min) min=a[i];k=i;
print(min);
switch(min):
case(x): if(y<z) {print(y);print(z);} else {print(z);print(y);}
case(y): if(x<z) {print(x);print(z);} else {print(z);print(x);}
case(z): if(y<x) {print(y);print(x);} else {print(x);print(y);}
}

 T04

void copy(int a[],int b[],int n)
{
 int i;
for(i=0;i<n;i++)
{
b[i]=a[i];
}
}


T05

void pause()
{
sleep(1);
print(time.localtime(time.time()))
}

T06

void shuixianhua()
{
 int i,n,m,k;
for(i=0;i<=999;i++)
 {
   n=i%10;
   m=(i-n)%100;
   k =i/100;
   if(((i*i*i)+(k*k*k)+(n*n*n))==i) print(i);
 }
}

T07

void suan(char a)
{int y=0,k=0,s=0,o=0；
 for i in a
   if(i>=a&&i<=z||i>=A&&i<=Z) y++;
   elif(i== ) k++；
   elif(i>=0&&i<=9) s++;
   else o++;
   print(y,k,s,o);
}

T08

void gaodu(int gaodu,int cishu)
{
if (cishu==10) return(gaodu);
else
  gaodu(gaodu/2,++cishu);
}