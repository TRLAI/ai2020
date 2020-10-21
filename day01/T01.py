counter = 0
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if i != j and j != k and k != i:
                print(" {}{}{} ".format(i,j,k),end = "")
                counter += 1
print("")
print("共{}种组合".format(counter))