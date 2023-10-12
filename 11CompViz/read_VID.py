
import cv2

# we read a video frame by frame using

video = cv2.VideoCapture('0DATA/Resources/Videos/dog.mp4')

while True:
    isTrue, frame = video.read()
    cv2.imshow('video', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

video.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
