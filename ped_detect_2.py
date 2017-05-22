
from collections import deque
import numpy as np
import imutils
import cv2
import datetime
from imutils.object_detection import non_max_suppression
import gehl_detect as gd

import json
# same as process_video.py but also using alternate haarcascades classifiers... should be extended to use different classifiers to measure different features and finding overlap using nms
# Macintosh:opencv-haar-classifier-training marioag$ opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000 -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024 -precalcIdxBufSize 1024
images = "/Users/Mario/Desktop/originalPics/2002/07/28/big/"

imgs = gd.get_jpgs(images)
print imgs
Lear_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades/haarcascade_mcs_leftear.xml")
face_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
# bod_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/lbpcascades/lbpcascade_frontalface.xml")
# vidPath = "/Users/Mario/Documents/mit-github-projects/gehl/Gehl/t3old/GOPR2201.MP4"
vidPath = "/Users/Mario/Desktop/IMG_6452.m4v"
video = cv2.VideoCapture(vidPath)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# dictionary that will hold timestamp:[lx, ly, ux, uy]
coords = {}

# dictionary to hold the number of nms boxes per timestamp
count = {}

while True:

    start = datetime.datetime.now()  # sets up a timer
    tstamp = str((datetime.datetime.now() - start).total_seconds())
    # grab the current frame
    (grabbed, frame) = video.read()
    # detect people in the image

    image = imutils.resize(frame, width=min(800, frame.shape[1]))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    legs = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in legs:
        rect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print len(rect)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

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
        # cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # writes tstamp and bounding box coordinates to coords
        coords.update({tstamp: nRect})

        # writes timestamp and # of boxes to count
        count.update({tstamp: len(pick)})

        print count
        # print "lower center point X is: ", xA, "lower center point Y is: ", (yB + yA) / 2

    cv2.imshow('img', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()