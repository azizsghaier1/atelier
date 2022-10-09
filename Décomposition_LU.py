import numpy as np
def Décompositin(A):
    n=np.shape(A)[0]
    L = np.identity(n)
    U = np.zeros(shape=(n,n))
    for i in range(0, n-1):
        for j in range(i+1, n):
            L[j,i] = A[j,i]/A[i,i]
            for k in range(i + 1, n):
                A[j:k] -= L[j, i] * U[i, k]
        for j in range(i, n):
            U[i,j] = A[i,j]
    U[-1,-1] = A[-1,-1]
    return L, U


A = np.random.random(size=(3,3)) #générer une matrice au hazard
print('A=',A, "\n")
l, u = Décompositin(A)
print('L=',l, "\n\n",'U=', u, "\n")
print('LU=',np.matmul(l, u)) # tester s'il verife bien T=LU
