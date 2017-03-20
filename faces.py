import numpy as np
import cv2
from imutils import paths
import imutils

face_cascade = cv2.CascadeClassifier('ocv2/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('ocv2/opencv/data/haarcascades/haarcascade_eye.xml')

image_dir = "images/"
test_img = "images/G0061813.jpg"

for imagePath in paths.list_images(image_dir):

    img = cv2.imread(imagePath)
    img = imutils.resize(img, width=min(600, img.shape[1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()