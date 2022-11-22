import math as mt
import numpy as np

def dot(a,b):
    """
    Menghitung dot product dari vektor a dan b
    a, b : numpy array 1D
    """
    return np.dot(a,b)

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
    return mt.sqrt(np.dot(a,a))

def norm(a):
    """
    Mengembalikan vektor normalisasi dari a
    a : numpy array 1D
    """
    return a / magnitude(a)
