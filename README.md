# v1
import numpy as np

#input matrix and vector
A = np.array([[2, 1],
              [5, 3]])
b = np.array([8, 18])
print("Matrix A:\n", A)
print("Vector b:\n", b)

#gauss elimination method
x = np.linalg.solve(A, b)
print("\nSolution using Gaussian Elimination method:")
print(f"x = {x[0]}, y = {x[1]}\n")

#inverse matrix method
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b
print("Solution using inverse matrix method :\nA=", x_inv)


#performing lu decompostion 
n = A.shape[0]
L = np.eye(n)
U = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
    for j in range(i+1, n):
        L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

print("\nLower Triangular Matrix\n L:", L)
print("\nUpper Triangular Matrix \nU:", U)
#transpose of matrix
A_T = A.T
print("\nTranspose of A:\n", A_T)
