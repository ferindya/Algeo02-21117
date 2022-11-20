import cv2
import numpy as np
from eigen import *
from img_utils import *
from vector_util import *

def_size = 256

def getFaceImage(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img,1.1,4)
    if (len(face) == 0):
        return None
    for i in range(len(face)):
        x,y,h,w = face[i]
        if (h < 20 or w < 20):
            continue
        img = img[y:y+h,x:x+w]
        img = image_resize(img,width=def_size,height=def_size)
        break
    return img

def calcEigenFaces(listfoto, k):
    # Matriks rata-rata foto
    mean = np.zeros(def_size)
    for img in range(len(listfoto)):
        mean = mean + img
    mean = mean/len(listfoto)

    # Membuat matriks foto hasil normalisasi
    matfoto =  np.empty((def_size*def_size,0))
    for i in range(len(listfoto)):
        vectorFoto = np.reshape((listfoto[i]-mean), (def_size**2,1))
        matfoto = np.append(matfoto,vectorFoto, axis=1)
    
    # kovarian
    kov = matfoto.T @ matfoto

    # eigvector dari kovarian
    eigenValues, eigenVectors = eigen(kov)
    # Mengurutkan eigenvector dari eigenvalue terbesar
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # Mendapatkan eigenface sebanyak K
    eigenFaces = np.empty((def_size*def_size,k))
    for i in range(k):
        eigenFaces[:,[i]] = matfoto @ eigenVectors[:,[i]]

    # Mendapatkan matriks berat untuk masing-masing foto
    weight = np.empty((k,matfoto.shape[1]))
    for i in range(matfoto.shape[1]):
        for j in range(eigenFaces.shape[1]):
            weight[j,i] = projection(matfoto[:,i],eigenFaces[:,j])[1]
    return eigenFaces, mean, weight


def train(imgList):
    grayImgList = []
    # konversi tiap foto menjadi grayscale dan hanya mengambil bagian wajah
    for i in range(len(imgList)):
        tmp = getFaceImage(imgList[i])
        if tmp is not None:
            grayImgList.append(tmp)
    # mendapatkan eigenface, foto rata-rata, dan matriks berat
    eigenFaces, mean, weight = calcEigenFaces(grayImgList,(int)(0.75*len(grayImgList)))
    return eigenFaces, mean, weight

def test(img, eigenFaces, mean, weightTrainingData):
    reducedImg = getFaceImage(img) - mean
    weightTest = np.empty((eigenFaces.shape[1],1))
    vectorImg = np.reshape(reducedImg, (def_size**2,1))
    for i in range(eigenFaces.shape[1]):
        weightTest[i,0] = projection(vectorImg,eigenFaces[:,i])[1]
    distMin = 9999999999
    idx_min = -1
    for i in range(weightTrainingData.shape[1]):
        print(weightTest.shape)
        print(weightTrainingData[:,i].shape)
        dist = magnitude(weightTest[:,0]-weightTrainingData[:,i])
        if (dist < distMin):
            distMin = dist
            idx_min = i
    return idx_min, distMin






    