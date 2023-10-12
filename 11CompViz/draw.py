import cv2
import numpy as np

# 1. making a blank page
blank = np.zeros((500, 500), dtype="uint8")
# cv2.imshow('blank', blank)

# 2. making a box of any size
box = np.zeros((500, 500, 3), dtype="uint8")
box[100:200, 300:400] = 0, 255, 255
# cv2.imshow("box", box)

# 3. making a rectangle outline
cv2.rectangle(box, (0, 0), (250, 250), (0, 255, 255), 2)
cv2.imshow('rectangle_ouline', box)

# 4. making a filled rectangle
rectangle = cv2.rectangle(box, (0, 0), (250, 250), (0, 255, 255), cv2.FILLED)
cv2.imshow('rectangle_filled', box)

# 5. making a circle outline
cv2.circle(box, (blank.shape[1]//2, blank.shape[0]//2), 40, (255, 0, 255), 3)
cv2.imshow('circle_outline', box)

# 5. making a circle filled
cv2.circle(box, (blank.shape[1]//4, blank.shape[0]//4), 40, (255, 0, 255), -1)
cv2.imshow('circle_outline', box)

# 6. making a line
cv2.line(box, (10, 20), (100, 200), (255, 255, 255), 2)
cv2.imshow('line', box)

# 7. display text
cv2.putText(box, "hellow", (255, 255),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 0), 2)
cv2.imshow('text', box)
cv2.waitKey(0)
cv2.destroyAllWindows()
