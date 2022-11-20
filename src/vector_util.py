import numpy as np
import math as mt

def dot(a,b):
    """
    Menghitung dot product dari vektor a dan b
    a, b : numpy array 1D
    """
    res = 0
    for i in range(a.shape[0]):
        res +=  a[i]*b[i]
    return res

def projection(a,b):
    """"
    Menghitung proyeksi vektor a kepada b
    a, b : numpy array 1D
    
    Kembalian : vektor proyeksi, skalar proyeksi
    """
    scalar = dot(a,b)/dot(b,b)
    return scalar*b, scalar

def magnitude(a):
    """
    Menghitung panjang vektor
    a : numpy array 1D
    """
    res = 0
    for i in range(a.shape[0]):
        res += a[i]**2
    return mt.sqrt(res)

def norm(a):
    """
    Mengembalikan vektor normalisasi dari a
    a : numpy array 1D
    """
    return a / magnitude(a)
