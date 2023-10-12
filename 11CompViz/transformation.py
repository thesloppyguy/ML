import cv2
import numpy as np

img = cv2.imread("0DATA/Resources/Photos/cat.jpg")


# translation - shift img in x or y
def translate(image, x, y):
    transmet = np.float32([[1, 0, x], [0, 1, y]])
    dimesion = (img.shape[1], image.shape[1])
    return cv2.warpAffine(img, transmet, dimesion,)


def rotate(image, angle, rotation_point=None):

    height, width = img.shape[:2]

    if rotation_point is None:
        rotation_point = (width//2, height//2)

    rotmat = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimesion = (width, height)
    return cv2.warpAffine(image, rotmat, dimesion)


trans = translate(img, 100, 100)
cv2.imshow("trans", trans)

rotated_img = rotate(img, 45)
cv2.imshow("rotate", rotated_img)


cv2.waitKey(0)
