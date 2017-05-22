import numpy as np
import imutils
import cv2
import datetime
from imutils.object_detection import non_max_suppression
import os
import glob
import gehl_detect as gd

cwd = os.getcwd()
print cwd
imageDir = "/Users/Mario/Desktop/faces/"
# print imageDir
images = []

os.chdir(imageDir)
cwd = os.getcwd()
print cwd
# for file in os.listdir(imageDir):
#     # print file, "<filename"
#     images.append(file)
#     # print images
images = gd.get_jpgs(imageDir)
print images

fface_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
pface_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades/haarcascade_profileface.xml")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print "starting loop..."

for each in images:

    image = cv2.imread(each)
    image = imutils.resize(image, width=min(1000, image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fface = fface_cascade.detectMultiScale(gray, 1.3, 5)
    pface = pface_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in fface:
        frect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        print len(frect)
        fface_gray = gray[y:y + h, x:x + w]
        fface_color = image[y:y + h, x:x + w]

    for (x, y, w, h) in pface:
        prect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        print len(prect)
        pface_gray = gray[y:y + h, x:x + w]
        pface_color = image[y:y + h, x:x + w]

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(2, 2), scale=1.5)
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        i = 0
        i < 1000
        nRect = [xA, yA, xB, yB]
        nImg = cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    image = imutils.resize(image, width=min(1000, image.shape[1]))
    cv2.imshow('img', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
