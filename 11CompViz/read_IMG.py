import cv2 as cv

img = cv.imread("0DATA/Resources/Photos/cat.jpg")
cv.imshow('cat', img)
cv.waitKey(0)
