import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)#width
cap.set(4,480)#height
cap.set(10,100)#brightness
while True:
    ret, img = cap.read()
    cv2.imshow('a',img)
