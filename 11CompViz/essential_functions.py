import cv2

img = cv2.imread("0DATA/Resources/Photos/cat.jpg")

# converting to greyscale
grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("grey", grey)

# applying blur
blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
cv2.imshow("blur", blur)

# applying Edge Cascade (detection)
canny = cv2.Canny(img, 125, 175)
cv2.imshow("canny", canny)

# applying blur and edge cascade
blur_canny = cv2.Canny(blur, 125, 175)
cv2.imshow("canny_blur", blur_canny)

# dilating the image
dilate = cv2.dilate(canny, (3, 3), iterations=1)
cv2.imshow("dilate", dilate)

# erode the image (opposite of dilate)
erode = cv2.erode(dilate, (3, 3), iterations=1)
cv2.imshow("erode", erode)

# resizing an image
# inter_cubic is for making the picture bigger but is slow with good results
# inter_linear is for making the picture bigger but is fast with moderate results
# inter_area is for making image smaller
resize = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow("resize", resize)

# cropping the image
cropped = img[50:100, 200:250]
cv2.imshow("cropped", cropped)

cv2.waitKey(0)
