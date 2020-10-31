import numpy as np

matrix = np.array([list(map(int, input().split())) for i in range(3)])

print(np.linalg.det(matrix), '- determinant')
print(np.linalg.inv(matrix), '- inverse matrix')
print(np.linalg.det(np.linalg.inv(matrix)), '- determinant of inverse matrix')
print(matrix.dot(np.linalg.inv(matrix)), '= 1')
print(np.flip(np.transpose(matrix)))