input_list = [1, 2, 3, 4]
a = []


def copy_list(n1, n2):
    n1.extend(n2)


copy_list(a, input_list)
print(a)
