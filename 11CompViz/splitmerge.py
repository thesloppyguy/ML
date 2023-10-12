import cv2
import numpy as np

img = cv2.imread('0DATA/Resources/Photos/cat.jpg')
cv2.imshow('Cats', img)

b, g, r = cv2.split(img)
# images will be black and white
cv2.imshow("blue", b)
cv2.imshow("green", g)
cv2.imshow("red", r)

merged_img = cv2.merge([b, g, r])
cv2.imshow("merged", merged_img)

cv2.waitKey(0)
