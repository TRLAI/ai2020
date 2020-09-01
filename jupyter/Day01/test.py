
t1 = 10

def py_sum(py_list):
    py_sumnum = 0
    for i in range(len(py_list)):
        py_sumnum += py_list[i]
    return py_sumnum
temp_list = [1,2,3,4,5]
print (py_sum(temp_list))
