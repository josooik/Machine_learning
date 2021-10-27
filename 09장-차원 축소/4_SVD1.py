from numpy.linalg import svd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

print("SVD (Singular Value Decomposition)")

print("\n")

np.random.seed(121)
A = np.random.randn(4, 4)
print("A : \n", A)

print("\n")

U, Sigma, Vt = svd(A)
print("U.shape : %s, Sigma.shape : %s, Vt.shape : %s" % (U.shape, Sigma.shape, Vt.shape))

print("\n")

print("U : \n", U)

print("\n")

print("Sigma : ", Sigma)

print("\n")

print("Vt : \n", Vt)

print("\n")

Sigma_mat = np.diag(Sigma)
A2 = np.dot(np.dot(U, Sigma_mat), Vt)
print("A2 : \n", A2)

print("\n")

A[2] = A[0] + A[1]
A[3] = A[0]
print("A : \n", A)

print("\n")

U, Sigma, Vt = svd(A)
print("U.shape : %s, Sigma.shape : %s, Vt.shape : %s" % (U.shape, Sigma.shape, Vt.shape))

print("\n")

print("U : \n", U)

print("\n")

print("Sigma : ", Sigma)

print("\n")

print("Vt : \n", Vt)

print("\n")

U_ = U[:, :2]
Sigma_ = np.diag(Sigma[:2])     # np.diag : 대각선 행렬
Vt_ = Vt[:2, :]
print("U.shape : %s, Sigma.shape : %s, Vt.shape : %s" % (U.shape, Sigma.shape, Vt.shape))

print("\n")

A_ = np.dot(np.dot(U_, Sigma_), Vt_)
print("A_ : \n", A_)

print("\n")

np.random.seed(121)
matrix = np.random.random((6, 6))
print("matrix : \n", matrix)

print("\n")

U, Sigma, Vt = svd(matrix, full_matrices=False)
print("U.shape : %s, Sigma.shape : %s, Vt.shape : %s" % (U.shape, Sigma.shape, Vt.shape))

print("\n")

print("Sigma : ", Sigma)

print("\n")

U_tr, Sigma_tr, Vt_tr = svds(matrix, k=4)
print("U_tr.shape : %s, Sigma_tr.shape : %s, Vt_tr.shape : %s" % (U_tr.shape, Sigma_tr.shape, Vt_tr.shape))

print("\n")

print("Sigma_tr : ", Sigma_tr)

print("\n")

matrix_tr = np.dot(np.dot(U_tr, np.diag(Sigma_tr)), Vt_tr)
print("matrix_tr : \n", matrix_tr)