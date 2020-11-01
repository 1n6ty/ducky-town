import numpy as np

arr_np = np.array([[1, 0, 2], [2, -1, 1], [1, 3, -1]])
array = [[1, 0, 2], [2, -1, 1], [1, 3, -1]]

print(np.linalg.det(arr_np), '- determinant of array')
print(np.linalg.det(np.linalg.inv(arr_np)), '- determinant of inverse array')

def det2D(arr):
    return arr[0][0] * arr[1][1] - arr[0][1] * arr[1][0]

def crop(arr, z):
    return [[arr[i][j] for j in range(len(arr[i])) if j != z] for i in range(1, len(arr))]

def det(arr):
    if len(arr[0]) == 2:
        return det2D(arr)
    else:
        flag = True
        var = 0
        for i in range(len(arr[0])):
            if flag:
                var += det(crop(arr, i)) * arr[0][i]
            else:
                var -= det(crop(arr, i)) * arr[0][i]
            flag = not flag
        return var

print(det(array))