from matplotlib import pyplot as plt
import random
import numpy as np

def displayImgList(imgList, isRandom, cmap='gray'):
    row, column = (3,3)
    if (row*column > len(imgList)):
        row = 1
        column = len(imgList)
    fig = plt.figure(figsize=(row,column))
    for i in range(1,row*column+1):
        fig.add_subplot(row,column,i)
        if (isRandom):
            idx = random.randint(0,len(imgList)-1)
        else:
            idx = i-1
        plt.imshow(imgList[idx],cmap=cmap)
    plt.show()

def displayImg(img,cmap='gray'):
    plt.imshow(img,cmap=cmap)
    plt.show()
    
def displayReconstructionImg(weight, eigenFaces, mean, size):
    listReconImg = []
    reducMat = (eigenFaces @ weight)
    for i in range(reducMat.shape[1]):
        img = reducMat[:,i].reshape(size,size)
        img = img+mean
        listReconImg.append(img)
    displayImgList(listReconImg,False,cmap='gray')

def matToImgList(mat,size):
    listImg = []
    for i in range(mat.shape[1]):
        img = np.reshape(mat[:,i],(size,size))
        listImg.append(img)
    return listImg
