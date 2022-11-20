import numpy as np
from vector_util import *

def QRdecom(A):
    Q = np.empty(A.shape)
    for i in range(A.shape[1]):
        u = A[:,i]
        for j in range(0,i):
            u = u - projection(A[:,i],Q[:,j])[0]
        Q[:,i] = u
    for i in range(Q.shape[0]):
        Q[:,i] = norm(Q[:,i])
    R = np.zeros(A.shape)
    for i in range(R.shape[0]):
        for j in range(i,R.shape[1]):
            R[i][j] = dot(Q[:,i],A[:,j])
    return Q, R

def eigen(A):
    V = np.identity(A.shape[0])
    A1 = A
    for i in range(100):
        Q, R = QRdecom(A1)
        V = V @ Q
        A1 = R @ Q
    return A1.diagonal(), V

        
