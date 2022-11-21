import numpy as np
import cv2
import os
from img_utils import *

def extractImg(image_path, max_length = 256):
    img = cv2.imread(image_path)
    if (img.shape[0] >= img.shape[1] and img.shape[0] > max_length):
        img = image_resize(img, height=max_length)
    if (img.shape[1] > img.shape[0] and img.shape[1] > max_length):
        img = image_resize(img, height=max_length)
    return img

def data_extractor(images_path):
    print("Loading test images...")
    imgList= []
    faceimgList = []
    filename = []
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    for f in files:
        name = f.split("\\")[-1].lower()
        img = extractImg(f)
        face = getFaceImage(img)
        if face is not None :
            imgList.append(img)
            faceimgList.append(face)
            filename.append(name)
    return imgList, faceimgList, filename
