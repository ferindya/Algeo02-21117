import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    h = image.shape[0]
    w = image.shape[1]
    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    elif height is None:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    else:
        dim = (width,height)
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def getFaceImage(img,size=100):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
    face = face_cascade.detectMultiScale(img,1.1,3)
    cntFace = 0
    for i in range(len(face)):
        x,y,h,w = face[i]
        if (h < 20 or w < 20):
            continue
        img = img[y:y+h,x:x+w]
        img = image_resize(img,width=size,height=size)
        cntFace += 1
        break
    if (cntFace == 0):
        return None, None, None, None, None
    else:
        return img, x, y, h, w