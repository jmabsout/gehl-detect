import numpy as np
import cv2
from imutils import paths
import imutils
import os

face_cascade = cv2.CascadeClassifier('/Users/marioag/Documents/GitHub/gehl-detect/ocv2/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('/Users/marioag/Documents/GitHub/gehl-detect/ocv2/opencv/data/haarcascades/haarcascade_mcs_upperbody.xml')

image_dir = '/Users/marioag/Dropbox (Personal)/documents/mit-github-projects/gehl/Gehl/img'


i = 0
for imagePath in os.listdir(image_dir):
    if imagePath.endswith(".jpg"):
        print "processing image {}".format(i)
        imagePath = os.path.join(image_dir, imagePath)
        img = cv2.imread(imagePath)
        # print img
        img = imutils.resize(img, width=min(2500, img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        
        cv2.imwrite("/Users/marioag/Documents/GitHub/gehl-detect/out/faces/from_img2/image{}.png".format(i), img)
        i +=1



