#有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
numberList=[1,2,3,4]
complexList=[]
def permutationNum():
    for i in numberList:
        for j in numberList:
                for k in numberList:
                    if i!=j and k != j and i!=k:
                        complexList.append(str(i)+str(j)+str(k))
    print("共有{}种组合，分别为{}".format(len(complexList),complexList))

permutationNum()