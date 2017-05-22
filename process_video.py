from collections import deque
import numpy as np
import imutils
import cv2
import datetime
from imutils.object_detection import non_max_suppression
import json


# TODO:
# think about the site in terms of 5'x5' grid cells (don't need to necessarily set one up but think about how data will be represented relative to that)
# first objective is to be able to count the number of people in the space at a given time...
# set up a counter or something, create a list of all the counts at a given interval, maybe average that? and have that
# be the count of people for a given period of time

vidPath = "./t3old/GOPR2201.MP4"
video = cv2.VideoCapture(vidPath)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque([])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# dictionary that will hold timestamp:[lx, ly, ux, uy]
coords = {}

# dictionary to hold the number of nms boxes per timestamp
count = {}

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = video.read()
    # detect people in the image

    image = imutils.resize(frame, width=min(1000, frame.shape[1]))

    start = datetime.datetime.now()  # sets up a timer
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(2, 2), scale=1.5)

    tstamp = (datetime.datetime.now() - start).total_seconds()

    print("[INFO] detection took: {}s".format(
        (datetime.datetime.now() - start).total_seconds()))

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)



    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        i = 0
        i < 1000
        nRect = [xA, yA, xB, yB]
        nImg = cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # attempt to find contours within image;
        # first extract image within bounding box to grayscale, then find contours
        # if len(cnts) > 0:
        #     # find the largest contour in the mask, then use
        #     # it to compute the minimum enclosing circle and
        #     # centroid
        #     c = max(cnts, key=cv2.contourArea)
        #     ((x, y), radius) = cv2.minEnclosingCircle(c)
        #     M = cv2.moments(c)
        #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # dictKey = "t" +str(tstamp)

        #writes tstamp and bounding box coordinates to coords
        coords.update({tstamp:nRect})

        # writes timestamp and # of boxes to count
        count.update({tstamp: len(pick)})

        print count
        # print "lower center point X is: ", xA, "lower center point Y is: ", (yB + yA) / 2

    print len(coords)

    with open("coordinates.json", 'w') as fp:
        json.dump(coords, fp)

    with open("count.json", 'w') as fp:
        json.dump(count, fp)

    # show some information on the number of bounding boxes
    filename = vidPath[vidPath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))



    # show the output images
    # cv2.imshow("Before NMS", frame)
    cv2.imshow("After NMS", image)

    cv2.waitKey(1)

    cv2.destroyAllWindows()
    # print coords
