import cv2


def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


video = cv2.VideoCapture('0DATA/Resources/Videos/dog.mp4')

while True:
    isTrue, frame = video.read()
    frame_resize = rescale(frame)

    cv2.imshow('video', frame)
    cv2.imshow('video resize', frame_resize)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

video.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
