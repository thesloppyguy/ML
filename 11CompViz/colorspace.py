import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('0DATA/Resources/Photos/cat.jpg')
cv2.imshow('Cats', img)

#BGR - BW
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("grey", img_grey)

#BGR -HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", img_hsv)

#BGR - LAB
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab", img_lab)

#BGR - RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.imshow(img)
plt.show()


cv2.waitKey(0)
